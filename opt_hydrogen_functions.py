"""Linearer technoökonomischer Basiskern für die H2-Produktion.

Der Basiskern dimensioniert Photovoltaik, Onshore-Wind, PEM-Elektrolyseur,
Kompressor und Druckspeicher für einen bereits validierbaren stündlichen
Datensatz. Die Zielfunktion minimiert annualisierte Kosten. Die monatlichen und
stündlichen RED-III-Szenarien ergänzen eine mengenbasierte zeitliche
Stromzuordnung; weitere RED-III-Kriterien, Stapelläufe und Ergebnisgrafiken
folgen getrennt.

Kurze Testzeiträume werden als zyklisch wiederkehrende Perioden behandelt.
Variable Kosten, Emissionen und H2-Mengen werden deshalb mit dem Verhältnis
von 8.760 Stunden zur Länge der Testperiode auf ein Jahr hochgerechnet.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final, Mapping, Sequence

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from config_h2 import DEFAULT_CONFIG, HOURS_PER_YEAR, ModelConfig, Scenario
from h2_input_data import validate_hourly_input
from red_iii_data import (
    DEFAULT_RED_III_PARAMETERS,
    EligibilityEvidence,
    TemporalCorrelation,
    assess_red_iii_compliance,
    build_correlation_periods,
    calculate_ghg_savings_fraction,
    operation_from_optimizer_output,
)


NUMERICAL_ZERO_TOLERANCE: Final[float] = 1e-8
BALANCE_ABSOLUTE_TOLERANCE: Final[float] = 1e-6
COST_BALANCE_RELATIVE_TOLERANCE: Final[float] = 1e-9
RED_TEMPORAL_SCENARIOS: Final[frozenset[Scenario]] = frozenset(
    {Scenario.RED_MONTHLY, Scenario.RED_HOURLY}
)


class HydrogenOptimizationError(RuntimeError):
    """Basisklasse für verständliche Fehler des H2-Optimierungskerns."""


class HydrogenModelInfeasibleError(HydrogenOptimizationError):
    """Das H2-Modell kann die Nachfrage unter den Vorgaben nicht decken."""


@dataclass(frozen=True, slots=True)
class HydrogenOptimizationResult:
    """Kompaktes Ergebnis eines optimal gelösten H2-Modelllaufs."""

    solver_status: str
    solver_name: str
    runtime_seconds: float
    optimality_gap_fraction: float
    annualization_factor: float
    objective_eur_per_year: float
    lcoh_eur_per_kg_h2: float
    annual_h2_delivered_kg: float
    annual_h2_produced_kg: float
    annual_grid_emissions_kg_co2e: float
    operational_emission_intensity_kg_co2e_per_kg_h2: float
    annual_regulatory_non_renewable_electricity_mwh: float
    annual_regulatory_emissions_kg_co2e: float
    regulatory_emission_intensity_kg_co2e_per_kg_h2: float
    regulatory_emission_intensity_g_co2e_per_mj_h2: float
    red_iii_ghg_savings_fraction: float
    red_iii_ghg_compliant: bool
    red_iii_maximum_product_intensity_kg_co2e_per_kg_h2: float
    red_iii_maximum_product_intensity_g_co2e_per_mj: float
    max_electricity_balance_residual_mwh: float
    max_hydrogen_balance_residual_kg: float
    annual_cost_balance_residual_eur_per_year: float
    red_iii_temporal_mode: str | None
    red_iii_temporal_compliant: bool | None
    red_iii_correlation_periods: int
    max_red_iii_temporal_deficit_mwh: float
    capacities: dict[str, float]
    annual_costs: dict[str, float]
    hourly_operation: pd.DataFrame


def capital_recovery_factor(
    real_wacc_fraction: float,
    lifetime_years: float,
) -> float:
    """Berechne den Annuitätenfaktor einer Investition.

    Bei einem WACC von null wird der Grenzfall ``1 / lifetime_years``
    verwendet. WACC und Lebensdauer werden geprüft, bevor die Formel
    ausgewertet wird.
    """

    if isinstance(real_wacc_fraction, bool) or not isinstance(
        real_wacc_fraction, (int, float)
    ):
        raise TypeError("real_wacc_fraction muss eine reelle Zahl sein.")
    if isinstance(lifetime_years, bool) or not isinstance(
        lifetime_years, (int, float)
    ):
        raise TypeError("lifetime_years muss eine reelle Zahl sein.")
    rate = float(real_wacc_fraction)
    lifetime = float(lifetime_years)
    if not isfinite(rate) or not isfinite(lifetime):
        raise ValueError("WACC und Lebensdauer müssen endlich sein.")
    if not 0.0 <= rate < 1.0:
        raise ValueError("Der reale WACC muss zwischen 0 und 1 liegen.")
    if lifetime <= 0.0:
        raise ValueError("Die Lebensdauer muss positiv sein.")
    if rate == 0.0:
        return 1.0 / lifetime
    growth = (1.0 + rate) ** lifetime
    return rate * growth / (growth - 1.0)


def optimize_hydrogen_system(
    hourly_input: pd.DataFrame,
    *,
    config: ModelConfig = DEFAULT_CONFIG,
    allow_grid_import: bool | None = None,
    solver_output: bool = False,
    solver_backend: str = "auto",
) -> HydrogenOptimizationResult:
    """Dimensioniere und betreibe das H2-System kostenminimal.

    Unterstützt werden ``Scenario.REFERENCE``, ``Scenario.OFF_GRID``,
    ``Scenario.RED_MONTHLY`` und ``Scenario.RED_HOURLY``.
    """

    config.validate()
    _validate_scenario(config.scenario)
    grid_import_allowed = _resolve_grid_import_setting(
        config.scenario, allow_grid_import
    )
    if not isinstance(solver_output, bool):
        raise TypeError("solver_output muss ein boolescher Wert sein.")
    selected_solver = _validate_solver_backend(solver_backend)
    if config.study.time_step_hours.value != 1.0:
        raise ValueError("Der aktuelle Basiskern erwartet stündliche Zeitschritte.")

    data = validate_hourly_input(hourly_input)
    number_of_hours = len(data)
    time_step_hours = config.study.time_step_hours.value
    modeled_hours = number_of_hours * time_step_hours
    annualization_factor = HOURS_PER_YEAR / modeled_hours

    if selected_solver == "scipy_highs":
        return _optimize_hydrogen_system_scipy(
            data=data,
            config=config,
            grid_import_allowed=grid_import_allowed,
            solver_output=solver_output,
        )

    technology = config.technologies
    electrolyzer = technology.electrolyzer
    compressor = technology.compressor
    storage = technology.h2_storage
    water = technology.water

    electrolyzer_specific_energy = (
        electrolyzer.specific_electricity_mwh_per_kg_h2
    )
    compressor_specific_energy = compressor.specific_electricity_mwh_per_kg_h2
    h2_survival_fraction = 1.0 - compressor.h2_loss_fraction.value

    model = gp.Model("single_site_h2_base")
    model.Params.OutputFlag = int(solver_output)

    pv_capacity_mw = model.addVar(lb=0.0, name="pv_capacity_mw")
    wind_capacity_mw = model.addVar(lb=0.0, name="wind_capacity_mw")
    electrolyzer_capacity_mw = model.addVar(
        lb=0.0, name="electrolyzer_capacity_mw"
    )
    compressor_capacity_mw = model.addVar(
        lb=0.0, name="compressor_capacity_mw"
    )
    h2_storage_capacity_kg = model.addVar(
        lb=0.0, name="h2_storage_capacity_kg"
    )

    hours = range(number_of_hours)
    pv_generation_mwh = model.addVars(hours, lb=0.0, name="pv_generation_mwh")
    wind_generation_mwh = model.addVars(
        hours, lb=0.0, name="wind_generation_mwh"
    )
    pv_self_consumption_mwh = model.addVars(
        hours, lb=0.0, name="pv_self_consumption_mwh"
    )
    wind_self_consumption_mwh = model.addVars(
        hours, lb=0.0, name="wind_self_consumption_mwh"
    )
    grid_upper_bound = GRB.INFINITY if grid_import_allowed else 0.0
    grid_import_mwh = model.addVars(
        hours, lb=0.0, ub=grid_upper_bound, name="grid_import_mwh"
    )
    electrolyzer_electricity_mwh = model.addVars(
        hours, lb=0.0, name="electrolyzer_electricity_mwh"
    )
    compressor_electricity_mwh = model.addVars(
        hours, lb=0.0, name="compressor_electricity_mwh"
    )
    h2_production_kg = model.addVars(hours, lb=0.0, name="h2_production_kg")
    h2_storage_level_kg = model.addVars(
        hours, lb=0.0, name="h2_storage_level_kg"
    )

    pv_capacity_factors = data["pv_capacity_factor"].to_numpy(dtype=float)
    wind_capacity_factors = data["wind_capacity_factor"].to_numpy(dtype=float)
    h2_demand = data["h2_demand"].to_numpy(dtype=float)

    for hour in hours:
        model.addConstr(
            pv_generation_mwh[hour]
            <= pv_capacity_mw * pv_capacity_factors[hour] * time_step_hours,
            name=f"pv_availability_{hour}",
        )
        model.addConstr(
            wind_generation_mwh[hour]
            <= wind_capacity_mw * wind_capacity_factors[hour] * time_step_hours,
            name=f"wind_availability_{hour}",
        )
        model.addConstr(
            pv_self_consumption_mwh[hour] <= pv_generation_mwh[hour],
            name=f"pv_self_consumption_limit_{hour}",
        )
        model.addConstr(
            wind_self_consumption_mwh[hour] <= wind_generation_mwh[hour],
            name=f"wind_self_consumption_limit_{hour}",
        )
        if config.scenario not in RED_TEMPORAL_SCENARIOS:
            model.addConstr(
                pv_self_consumption_mwh[hour] == pv_generation_mwh[hour],
                name=f"pv_no_external_allocation_{hour}",
            )
            model.addConstr(
                wind_self_consumption_mwh[hour] == wind_generation_mwh[hour],
                name=f"wind_no_external_allocation_{hour}",
            )
        model.addConstr(
            electrolyzer_electricity_mwh[hour]
            <= electrolyzer_capacity_mw * time_step_hours,
            name=f"electrolyzer_capacity_{hour}",
        )
        model.addConstr(
            h2_production_kg[hour]
            == electrolyzer_electricity_mwh[hour]
            / electrolyzer_specific_energy,
            name=f"electrolyzer_conversion_{hour}",
        )
        model.addConstr(
            compressor_electricity_mwh[hour]
            == h2_production_kg[hour] * compressor_specific_energy,
            name=f"compressor_consumption_{hour}",
        )
        model.addConstr(
            compressor_electricity_mwh[hour]
            <= compressor_capacity_mw * time_step_hours,
            name=f"compressor_capacity_{hour}",
        )
        model.addConstr(
            pv_self_consumption_mwh[hour]
            + wind_self_consumption_mwh[hour]
            + grid_import_mwh[hour]
            == electrolyzer_electricity_mwh[hour]
            + compressor_electricity_mwh[hour],
            name=f"electricity_balance_{hour}",
        )
        model.addConstr(
            h2_storage_level_kg[hour] <= h2_storage_capacity_kg,
            name=f"storage_capacity_{hour}",
        )

        previous_hour = number_of_hours - 1 if hour == 0 else hour - 1
        model.addConstr(
            h2_storage_level_kg[hour]
            == h2_storage_level_kg[previous_hour]
            + h2_production_kg[hour] * h2_survival_fraction
            - h2_demand[hour],
            name=f"hydrogen_balance_{hour}",
        )

    if config.scenario is Scenario.RED_MONTHLY:
        for period, period_hours in _monthly_hour_groups(data):
            model.addConstr(
                gp.quicksum(
                    pv_generation_mwh[hour] + wind_generation_mwh[hour]
                    for hour in period_hours
                )
                >= gp.quicksum(
                    electrolyzer_electricity_mwh[hour]
                    + compressor_electricity_mwh[hour]
                    for hour in period_hours
                ),
                name=f"red_iii_monthly_correlation_{period.replace('-', '_')}",
            )
    elif config.scenario is Scenario.RED_HOURLY:
        for hour in hours:
            model.addConstr(
                pv_generation_mwh[hour] + wind_generation_mwh[hour]
                >= electrolyzer_electricity_mwh[hour]
                + compressor_electricity_mwh[hour],
                name=f"red_iii_hourly_correlation_{hour}",
            )

    cost_expressions = _build_annual_cost_expressions(
        config=config,
        data=data,
        annualization_factor=annualization_factor,
        pv_capacity_mw=pv_capacity_mw,
        wind_capacity_mw=wind_capacity_mw,
        electrolyzer_capacity_mw=electrolyzer_capacity_mw,
        compressor_capacity_mw=compressor_capacity_mw,
        h2_storage_capacity_kg=h2_storage_capacity_kg,
        pv_generation_mwh=pv_generation_mwh,
        wind_generation_mwh=wind_generation_mwh,
        grid_import_mwh=grid_import_mwh,
        h2_production_kg=h2_production_kg,
    )
    model.setObjective(gp.quicksum(cost_expressions.values()), GRB.MINIMIZE)
    try:
        model.optimize()
    except gp.GurobiError as exc:
        if selected_solver == "auto" and "Model too large for size-limited license" in str(exc):
            return _optimize_hydrogen_system_scipy(
                data=data,
                config=config,
                grid_import_allowed=grid_import_allowed,
                solver_output=solver_output,
            )
        raise HydrogenOptimizationError(f"Gurobi konnte das Modell nicht lösen: {exc}") from exc

    if model.Status == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.optimize()
    if model.Status == GRB.INFEASIBLE:
        raise HydrogenModelInfeasibleError(
            "Das H2-Modell ist nicht lösbar. Prüfe erneuerbare Verfügbarkeit, "
            "Netzfreigabe und H2-Nachfrage."
        )
    if model.Status == GRB.UNBOUNDED:
        raise HydrogenOptimizationError("Das H2-Modell ist unbeschränkt.")
    if model.Status != GRB.OPTIMAL:
        raise HydrogenOptimizationError(
            f"Gurobi beendete den Modelllauf mit Status {model.Status}."
        )

    optimality_gap_fraction = float(model.MIPGap) if bool(model.IsMIP) else 0.0

    annual_h2_delivered_kg = float(h2_demand.sum() * annualization_factor)
    annual_h2_produced_kg = float(
        sum(h2_production_kg[hour].X for hour in hours) * annualization_factor
    )
    annual_costs = {
        name: _clean_number(_expression_value(expression))
        for name, expression in cost_expressions.items()
    }
    objective_eur_per_year = _clean_number(model.ObjVal)
    lcoh_eur_per_kg_h2 = objective_eur_per_year / annual_h2_delivered_kg

    annual_grid_emissions = float(
        sum(
            grid_import_mwh[hour].X
            * data["grid_emission_factor"].iloc[hour]
            for hour in hours
        )
        * annualization_factor
    )

    capacities = {
        "pv_capacity_mw": _clean_number(pv_capacity_mw.X),
        "wind_capacity_mw": _clean_number(wind_capacity_mw.X),
        "electrolyzer_capacity_mw": _clean_number(electrolyzer_capacity_mw.X),
        "compressor_capacity_mw": _clean_number(compressor_capacity_mw.X),
        "h2_storage_capacity_kg": _clean_number(h2_storage_capacity_kg.X),
    }
    hourly_operation = _build_hourly_result(
        data=data,
        time_step_hours=time_step_hours,
        h2_survival_fraction=h2_survival_fraction,
        capacities=capacities,
        pv_generation_mwh=[pv_generation_mwh[hour].X for hour in hours],
        wind_generation_mwh=[wind_generation_mwh[hour].X for hour in hours],
        pv_self_consumption_mwh=[
            pv_self_consumption_mwh[hour].X for hour in hours
        ],
        wind_self_consumption_mwh=[
            wind_self_consumption_mwh[hour].X for hour in hours
        ],
        grid_import_mwh=[grid_import_mwh[hour].X for hour in hours],
        electrolyzer_electricity_mwh=[
            electrolyzer_electricity_mwh[hour].X for hour in hours
        ],
        compressor_electricity_mwh=[
            compressor_electricity_mwh[hour].X for hour in hours
        ],
        h2_production_kg=[h2_production_kg[hour].X for hour in hours],
        h2_storage_level_kg=[h2_storage_level_kg[hour].X for hour in hours],
    )
    hourly_operation = _add_regulatory_emission_columns(
        hourly_operation, config.scenario
    )
    emission_diagnostics = _calculate_regulatory_emission_diagnostics(
        hourly_operation,
        annualization_factor=annualization_factor,
        annual_h2_delivered_kg=annual_h2_delivered_kg,
    )
    diagnostics = calculate_result_diagnostics(
        hourly_operation,
        annual_costs=annual_costs,
        objective_eur_per_year=objective_eur_per_year,
    )
    _validate_result_diagnostics(
        diagnostics,
        objective_eur_per_year=objective_eur_per_year,
    )
    red_diagnostics = _calculate_red_temporal_diagnostics(
        hourly_operation,
        config.scenario,
        regulatory_intensity_kg_co2e_per_kg_h2=emission_diagnostics[
            "regulatory_emission_intensity_kg_co2e_per_kg_h2"
        ],
    )
    return HydrogenOptimizationResult(
        solver_status="optimal",
        solver_name="gurobi",
        runtime_seconds=float(model.Runtime),
        optimality_gap_fraction=optimality_gap_fraction,
        annualization_factor=float(annualization_factor),
        objective_eur_per_year=objective_eur_per_year,
        lcoh_eur_per_kg_h2=float(lcoh_eur_per_kg_h2),
        annual_h2_delivered_kg=annual_h2_delivered_kg,
        annual_h2_produced_kg=annual_h2_produced_kg,
        annual_grid_emissions_kg_co2e=_clean_number(annual_grid_emissions),
        operational_emission_intensity_kg_co2e_per_kg_h2=float(
            annual_grid_emissions / annual_h2_delivered_kg
        ),
        annual_regulatory_non_renewable_electricity_mwh=emission_diagnostics[
            "annual_regulatory_non_renewable_electricity_mwh"
        ],
        annual_regulatory_emissions_kg_co2e=emission_diagnostics[
            "annual_regulatory_emissions_kg_co2e"
        ],
        regulatory_emission_intensity_kg_co2e_per_kg_h2=emission_diagnostics[
            "regulatory_emission_intensity_kg_co2e_per_kg_h2"
        ],
        regulatory_emission_intensity_g_co2e_per_mj_h2=emission_diagnostics[
            "regulatory_emission_intensity_g_co2e_per_mj_h2"
        ],
        red_iii_ghg_savings_fraction=emission_diagnostics[
            "red_iii_ghg_savings_fraction"
        ],
        red_iii_ghg_compliant=bool(
            emission_diagnostics["red_iii_ghg_compliant"]
        ),
        red_iii_maximum_product_intensity_kg_co2e_per_kg_h2=(
            DEFAULT_RED_III_PARAMETERS.maximum_product_intensity_kg_co2e_per_kg_h2
        ),
        red_iii_maximum_product_intensity_g_co2e_per_mj=(
            DEFAULT_RED_III_PARAMETERS.maximum_product_intensity_g_co2e_per_mj
        ),
        max_electricity_balance_residual_mwh=diagnostics[
            "max_electricity_balance_residual_mwh"
        ],
        max_hydrogen_balance_residual_kg=diagnostics[
            "max_hydrogen_balance_residual_kg"
        ],
        annual_cost_balance_residual_eur_per_year=diagnostics[
            "annual_cost_balance_residual_eur_per_year"
        ],
        red_iii_temporal_mode=red_diagnostics["red_iii_temporal_mode"],
        red_iii_temporal_compliant=red_diagnostics[
            "red_iii_temporal_compliant"
        ],
        red_iii_correlation_periods=red_diagnostics[
            "red_iii_correlation_periods"
        ],
        max_red_iii_temporal_deficit_mwh=red_diagnostics[
            "max_red_iii_temporal_deficit_mwh"
        ],
        capacities=capacities,
        annual_costs=annual_costs,
        hourly_operation=hourly_operation,
    )


def _validate_scenario(scenario: Scenario) -> None:
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario muss ein Wert aus config_h2.Scenario sein.")


def _validate_solver_backend(solver_backend: str) -> str:
    if not isinstance(solver_backend, str):
        raise TypeError("solver_backend muss ein String sein.")
    normalized = solver_backend.strip().lower().replace("-", "_")
    if normalized not in {"auto", "gurobi", "scipy_highs"}:
        raise ValueError(
            "solver_backend muss 'auto', 'gurobi' oder 'scipy_highs' sein."
        )
    return normalized


def _resolve_grid_import_setting(
    scenario: Scenario,
    allow_grid_import: bool | None,
) -> bool:
    if allow_grid_import is not None and not isinstance(allow_grid_import, bool):
        raise TypeError("allow_grid_import muss bool oder None sein.")
    if scenario is Scenario.OFF_GRID:
        if allow_grid_import is True:
            raise ValueError("Im OFF_GRID-Szenario darf Netzbezug nicht aktiviert werden.")
        return False
    return True if allow_grid_import is None else allow_grid_import


def _optimize_hydrogen_system_scipy(
    *,
    data: pd.DataFrame,
    config: ModelConfig,
    grid_import_allowed: bool,
    solver_output: bool,
) -> HydrogenOptimizationResult:
    """Löse dasselbe kontinuierliche LP mit SciPy/HiGHS.

    Diese Implementierung ermöglicht den 8.760-Stunden-Lauf auch dann, wenn
    die lokal installierte Gurobi-Lizenz auf kleine Modelle begrenzt ist.
    """

    from time import perf_counter

    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    number_of_hours = len(data)
    time_step_hours = config.study.time_step_hours.value
    annualization_factor = HOURS_PER_YEAR / (number_of_hours * time_step_hours)
    technology = config.technologies
    pv = technology.pv
    wind = technology.wind_onshore
    electrolyzer = technology.electrolyzer
    compressor = technology.compressor
    storage = technology.h2_storage
    water = technology.water
    h2_survival_fraction = 1.0 - compressor.h2_loss_fraction.value
    electrolyzer_specific_energy = electrolyzer.specific_electricity_mwh_per_kg_h2
    compressor_specific_energy = compressor.specific_electricity_mwh_per_kg_h2

    pv_capacity_index = 0
    wind_capacity_index = 1
    electrolyzer_capacity_index = 2
    compressor_capacity_index = 3
    storage_capacity_index = 4
    offset = 5

    def hourly_indices() -> np.ndarray:
        nonlocal offset
        indices = np.arange(offset, offset + number_of_hours, dtype=int)
        offset += number_of_hours
        return indices

    pv_generation_indices = hourly_indices()
    wind_generation_indices = hourly_indices()
    pv_self_consumption_indices = hourly_indices()
    wind_self_consumption_indices = hourly_indices()
    grid_import_indices = hourly_indices()
    electrolyzer_electricity_indices = hourly_indices()
    compressor_electricity_indices = hourly_indices()
    h2_production_indices = hourly_indices()
    storage_level_indices = hourly_indices()
    number_of_variables = offset

    objective = np.zeros(number_of_variables, dtype=float)
    objective[pv_capacity_index] = (
        pv.capex_eur_per_mw
        * capital_recovery_factor(pv.real_wacc_fraction.value, pv.lifetime_years.value)
        + pv.fixed_opex_eur_per_mw_year
    )
    objective[wind_capacity_index] = (
        wind.capex_eur_per_mw
        * capital_recovery_factor(
            wind.real_wacc_fraction.value, wind.lifetime_years.value
        )
        + wind.fixed_opex_eur_per_mw_year
    )
    objective[electrolyzer_capacity_index] = (
        electrolyzer.capex_eur_per_mw
        * capital_recovery_factor(
            electrolyzer.real_wacc_fraction.value,
            electrolyzer.lifetime_years.value,
        )
        + electrolyzer.fixed_opex_eur_per_mw_year
    )
    objective[compressor_capacity_index] = compressor.capex_eur_per_mw * (
        capital_recovery_factor(
            compressor.real_wacc_fraction.value, compressor.lifetime_years.value
        )
        + compressor.fixed_opex_fraction_per_year.value
    )
    objective[storage_capacity_index] = storage.capex_eur_per_kg_h2.value * (
        capital_recovery_factor(
            storage.real_wacc_fraction.value, storage.lifetime_years.value
        )
        + storage.fixed_opex_fraction_per_year.value
    )
    objective[pv_generation_indices] = (
        annualization_factor * pv.variable_opex_eur_per_mwh.value
    )
    objective[wind_generation_indices] = (
        annualization_factor * wind.variable_opex_eur_per_mwh.value
    )
    objective[grid_import_indices] = (
        annualization_factor * data["electricity_price"].to_numpy(dtype=float)
    )
    objective[h2_production_indices] = annualization_factor * (
        electrolyzer.water_consumption_kg_per_kg_h2.value
        / water.density_kg_per_m3.value
        * water.price_eur_per_m3.value
    )

    inequality_rows: list[int] = []
    inequality_columns: list[int] = []
    inequality_values: list[float] = []
    inequality_rhs: list[float] = []
    equality_rows: list[int] = []
    equality_columns: list[int] = []
    equality_values: list[float] = []
    equality_rhs: list[float] = []

    def add_inequality(entries: Sequence[tuple[int, float]], rhs: float = 0.0) -> None:
        row = len(inequality_rhs)
        for column, value in entries:
            inequality_rows.append(row)
            inequality_columns.append(column)
            inequality_values.append(value)
        inequality_rhs.append(rhs)

    def add_equality(entries: Sequence[tuple[int, float]], rhs: float = 0.0) -> None:
        row = len(equality_rhs)
        for column, value in entries:
            equality_rows.append(row)
            equality_columns.append(column)
            equality_values.append(value)
        equality_rhs.append(rhs)

    pv_capacity_factors = data["pv_capacity_factor"].to_numpy(dtype=float)
    wind_capacity_factors = data["wind_capacity_factor"].to_numpy(dtype=float)
    h2_demand = data["h2_demand"].to_numpy(dtype=float)
    for hour in range(number_of_hours):
        add_inequality(
            (
                (int(pv_generation_indices[hour]), 1.0),
                (pv_capacity_index, -pv_capacity_factors[hour] * time_step_hours),
            )
        )
        add_inequality(
            (
                (int(wind_generation_indices[hour]), 1.0),
                (wind_capacity_index, -wind_capacity_factors[hour] * time_step_hours),
            )
        )
        add_inequality(
            (
                (int(pv_self_consumption_indices[hour]), 1.0),
                (int(pv_generation_indices[hour]), -1.0),
            )
        )
        add_inequality(
            (
                (int(wind_self_consumption_indices[hour]), 1.0),
                (int(wind_generation_indices[hour]), -1.0),
            )
        )
        add_inequality(
            (
                (int(electrolyzer_electricity_indices[hour]), 1.0),
                (electrolyzer_capacity_index, -time_step_hours),
            )
        )
        add_inequality(
            (
                (int(compressor_electricity_indices[hour]), 1.0),
                (compressor_capacity_index, -time_step_hours),
            )
        )
        add_inequality(
            (
                (int(storage_level_indices[hour]), 1.0),
                (storage_capacity_index, -1.0),
            )
        )

        add_equality(
            (
                (int(h2_production_indices[hour]), 1.0),
                (
                    int(electrolyzer_electricity_indices[hour]),
                    -1.0 / electrolyzer_specific_energy,
                ),
            )
        )
        add_equality(
            (
                (int(compressor_electricity_indices[hour]), 1.0),
                (int(h2_production_indices[hour]), -compressor_specific_energy),
            )
        )
        add_equality(
            (
                (int(pv_self_consumption_indices[hour]), 1.0),
                (int(wind_self_consumption_indices[hour]), 1.0),
                (int(grid_import_indices[hour]), 1.0),
                (int(electrolyzer_electricity_indices[hour]), -1.0),
                (int(compressor_electricity_indices[hour]), -1.0),
            )
        )
        previous_hour = number_of_hours - 1 if hour == 0 else hour - 1
        add_equality(
            (
                (int(storage_level_indices[hour]), 1.0),
                (int(storage_level_indices[previous_hour]), -1.0),
                (int(h2_production_indices[hour]), -h2_survival_fraction),
            ),
            -h2_demand[hour],
        )
        if config.scenario not in RED_TEMPORAL_SCENARIOS:
            add_equality(
                (
                    (int(pv_generation_indices[hour]), 1.0),
                    (int(pv_self_consumption_indices[hour]), -1.0),
                )
            )
            add_equality(
                (
                    (int(wind_generation_indices[hour]), 1.0),
                    (int(wind_self_consumption_indices[hour]), -1.0),
                )
            )

    if config.scenario is Scenario.RED_MONTHLY:
        for _period, period_hours in _monthly_hour_groups(data):
            entries: list[tuple[int, float]] = []
            for hour in period_hours:
                entries.extend(
                    (
                        (int(electrolyzer_electricity_indices[hour]), 1.0),
                        (int(compressor_electricity_indices[hour]), 1.0),
                        (int(pv_generation_indices[hour]), -1.0),
                        (int(wind_generation_indices[hour]), -1.0),
                    )
                )
            add_inequality(entries)
    elif config.scenario is Scenario.RED_HOURLY:
        for hour in range(number_of_hours):
            add_inequality(
                (
                    (int(electrolyzer_electricity_indices[hour]), 1.0),
                    (int(compressor_electricity_indices[hour]), 1.0),
                    (int(pv_generation_indices[hour]), -1.0),
                    (int(wind_generation_indices[hour]), -1.0),
                )
            )

    inequality_matrix = coo_matrix(
        (inequality_values, (inequality_rows, inequality_columns)),
        shape=(len(inequality_rhs), number_of_variables),
    ).tocsr()
    equality_matrix = coo_matrix(
        (equality_values, (equality_rows, equality_columns)),
        shape=(len(equality_rhs), number_of_variables),
    ).tocsr()
    bounds: list[tuple[float, float | None]] = [
        (0.0, None) for _ in range(number_of_variables)
    ]
    if not grid_import_allowed:
        for index in grid_import_indices:
            bounds[int(index)] = (0.0, 0.0)

    start_time = perf_counter()
    solution = linprog(
        objective,
        A_ub=inequality_matrix,
        b_ub=np.asarray(inequality_rhs),
        A_eq=equality_matrix,
        b_eq=np.asarray(equality_rhs),
        bounds=bounds,
        method="highs",
        options={"disp": solver_output},
    )
    runtime_seconds = perf_counter() - start_time
    if solution.status == 2:
        raise HydrogenModelInfeasibleError(
            "Das H2-Modell ist nicht lösbar. Prüfe erneuerbare Verfügbarkeit, Netzfreigabe und H2-Nachfrage."
        )
    if not solution.success or solution.x is None:
        raise HydrogenOptimizationError(
            f"SciPy/HiGHS beendete den Modelllauf mit Status {solution.status}: {solution.message}"
        )

    values = solution.x
    capacities = {
        "pv_capacity_mw": _clean_number(values[pv_capacity_index]),
        "wind_capacity_mw": _clean_number(values[wind_capacity_index]),
        "electrolyzer_capacity_mw": _clean_number(values[electrolyzer_capacity_index]),
        "compressor_capacity_mw": _clean_number(values[compressor_capacity_index]),
        "h2_storage_capacity_kg": _clean_number(values[storage_capacity_index]),
    }
    pv_generation = values[pv_generation_indices]
    wind_generation = values[wind_generation_indices]
    pv_self_consumption = values[pv_self_consumption_indices]
    wind_self_consumption = values[wind_self_consumption_indices]
    grid_import = values[grid_import_indices]
    electrolyzer_electricity = values[electrolyzer_electricity_indices]
    compressor_electricity = values[compressor_electricity_indices]
    h2_production = values[h2_production_indices]
    storage_level = values[storage_level_indices]
    annual_costs = _calculate_annual_costs_from_values(
        config=config,
        data=data,
        annualization_factor=annualization_factor,
        capacities=capacities,
        pv_generation_mwh=pv_generation,
        wind_generation_mwh=wind_generation,
        grid_import_mwh=grid_import,
        h2_production_kg=h2_production,
    )
    objective_eur_per_year = _clean_number(float(solution.fun))
    annual_h2_delivered_kg = float(h2_demand.sum() * annualization_factor)
    annual_h2_produced_kg = float(h2_production.sum() * annualization_factor)
    annual_grid_emissions = float(
        np.dot(grid_import, data["grid_emission_factor"].to_numpy(dtype=float))
        * annualization_factor
    )
    hourly_operation = _build_hourly_result(
        data=data,
        time_step_hours=time_step_hours,
        h2_survival_fraction=h2_survival_fraction,
        capacities=capacities,
        pv_generation_mwh=pv_generation,
        wind_generation_mwh=wind_generation,
        pv_self_consumption_mwh=pv_self_consumption,
        wind_self_consumption_mwh=wind_self_consumption,
        grid_import_mwh=grid_import,
        electrolyzer_electricity_mwh=electrolyzer_electricity,
        compressor_electricity_mwh=compressor_electricity,
        h2_production_kg=h2_production,
        h2_storage_level_kg=storage_level,
    )
    hourly_operation = _add_regulatory_emission_columns(
        hourly_operation, config.scenario
    )
    emission_diagnostics = _calculate_regulatory_emission_diagnostics(
        hourly_operation,
        annualization_factor=annualization_factor,
        annual_h2_delivered_kg=annual_h2_delivered_kg,
    )
    diagnostics = calculate_result_diagnostics(
        hourly_operation,
        annual_costs=annual_costs,
        objective_eur_per_year=objective_eur_per_year,
    )
    _validate_result_diagnostics(
        diagnostics, objective_eur_per_year=objective_eur_per_year
    )
    red_diagnostics = _calculate_red_temporal_diagnostics(
        hourly_operation,
        config.scenario,
        regulatory_intensity_kg_co2e_per_kg_h2=emission_diagnostics[
            "regulatory_emission_intensity_kg_co2e_per_kg_h2"
        ],
    )
    return HydrogenOptimizationResult(
        solver_status="optimal",
        solver_name="scipy_highs",
        runtime_seconds=float(runtime_seconds),
        optimality_gap_fraction=0.0,
        annualization_factor=float(annualization_factor),
        objective_eur_per_year=objective_eur_per_year,
        lcoh_eur_per_kg_h2=float(objective_eur_per_year / annual_h2_delivered_kg),
        annual_h2_delivered_kg=annual_h2_delivered_kg,
        annual_h2_produced_kg=annual_h2_produced_kg,
        annual_grid_emissions_kg_co2e=_clean_number(annual_grid_emissions),
        operational_emission_intensity_kg_co2e_per_kg_h2=float(
            annual_grid_emissions / annual_h2_delivered_kg
        ),
        annual_regulatory_non_renewable_electricity_mwh=emission_diagnostics[
            "annual_regulatory_non_renewable_electricity_mwh"
        ],
        annual_regulatory_emissions_kg_co2e=emission_diagnostics[
            "annual_regulatory_emissions_kg_co2e"
        ],
        regulatory_emission_intensity_kg_co2e_per_kg_h2=emission_diagnostics[
            "regulatory_emission_intensity_kg_co2e_per_kg_h2"
        ],
        regulatory_emission_intensity_g_co2e_per_mj_h2=emission_diagnostics[
            "regulatory_emission_intensity_g_co2e_per_mj_h2"
        ],
        red_iii_ghg_savings_fraction=emission_diagnostics[
            "red_iii_ghg_savings_fraction"
        ],
        red_iii_ghg_compliant=bool(
            emission_diagnostics["red_iii_ghg_compliant"]
        ),
        red_iii_maximum_product_intensity_kg_co2e_per_kg_h2=(
            DEFAULT_RED_III_PARAMETERS.maximum_product_intensity_kg_co2e_per_kg_h2
        ),
        red_iii_maximum_product_intensity_g_co2e_per_mj=(
            DEFAULT_RED_III_PARAMETERS.maximum_product_intensity_g_co2e_per_mj
        ),
        max_electricity_balance_residual_mwh=diagnostics[
            "max_electricity_balance_residual_mwh"
        ],
        max_hydrogen_balance_residual_kg=diagnostics[
            "max_hydrogen_balance_residual_kg"
        ],
        annual_cost_balance_residual_eur_per_year=diagnostics[
            "annual_cost_balance_residual_eur_per_year"
        ],
        red_iii_temporal_mode=red_diagnostics["red_iii_temporal_mode"],
        red_iii_temporal_compliant=red_diagnostics[
            "red_iii_temporal_compliant"
        ],
        red_iii_correlation_periods=red_diagnostics[
            "red_iii_correlation_periods"
        ],
        max_red_iii_temporal_deficit_mwh=red_diagnostics[
            "max_red_iii_temporal_deficit_mwh"
        ],
        capacities=capacities,
        annual_costs=annual_costs,
        hourly_operation=hourly_operation,
    )


def _calculate_annual_costs_from_values(
    *,
    config: ModelConfig,
    data: pd.DataFrame,
    annualization_factor: float,
    capacities: Mapping[str, float],
    pv_generation_mwh: np.ndarray,
    wind_generation_mwh: np.ndarray,
    grid_import_mwh: np.ndarray,
    h2_production_kg: np.ndarray,
) -> dict[str, float]:
    technology = config.technologies
    pv = technology.pv
    wind = technology.wind_onshore
    electrolyzer = technology.electrolyzer
    compressor = technology.compressor
    storage = technology.h2_storage
    water = technology.water
    costs = {
        "pv_annualized_capex_eur_per_year": capacities["pv_capacity_mw"]
        * pv.capex_eur_per_mw
        * capital_recovery_factor(pv.real_wacc_fraction.value, pv.lifetime_years.value),
        "pv_fixed_opex_eur_per_year": capacities["pv_capacity_mw"]
        * pv.fixed_opex_eur_per_mw_year,
        "pv_variable_opex_eur_per_year": annualization_factor
        * float(pv_generation_mwh.sum())
        * pv.variable_opex_eur_per_mwh.value,
        "wind_annualized_capex_eur_per_year": capacities["wind_capacity_mw"]
        * wind.capex_eur_per_mw
        * capital_recovery_factor(
            wind.real_wacc_fraction.value, wind.lifetime_years.value
        ),
        "wind_fixed_opex_eur_per_year": capacities["wind_capacity_mw"]
        * wind.fixed_opex_eur_per_mw_year,
        "wind_variable_opex_eur_per_year": annualization_factor
        * float(wind_generation_mwh.sum())
        * wind.variable_opex_eur_per_mwh.value,
        "electrolyzer_annualized_capex_eur_per_year": capacities[
            "electrolyzer_capacity_mw"
        ]
        * electrolyzer.capex_eur_per_mw
        * capital_recovery_factor(
            electrolyzer.real_wacc_fraction.value, electrolyzer.lifetime_years.value
        ),
        "electrolyzer_fixed_opex_eur_per_year": capacities[
            "electrolyzer_capacity_mw"
        ]
        * electrolyzer.fixed_opex_eur_per_mw_year,
        "compressor_annualized_capex_eur_per_year": capacities[
            "compressor_capacity_mw"
        ]
        * compressor.capex_eur_per_mw
        * capital_recovery_factor(
            compressor.real_wacc_fraction.value, compressor.lifetime_years.value
        ),
        "compressor_fixed_opex_eur_per_year": capacities[
            "compressor_capacity_mw"
        ]
        * compressor.capex_eur_per_mw
        * compressor.fixed_opex_fraction_per_year.value,
        "h2_storage_annualized_capex_eur_per_year": capacities[
            "h2_storage_capacity_kg"
        ]
        * storage.capex_eur_per_kg_h2.value
        * capital_recovery_factor(
            storage.real_wacc_fraction.value, storage.lifetime_years.value
        ),
        "h2_storage_fixed_opex_eur_per_year": capacities[
            "h2_storage_capacity_kg"
        ]
        * storage.capex_eur_per_kg_h2.value
        * storage.fixed_opex_fraction_per_year.value,
        "grid_electricity_eur_per_year": annualization_factor
        * float(
            np.dot(
                grid_import_mwh,
                data["electricity_price"].to_numpy(dtype=float),
            )
        ),
        "water_eur_per_year": annualization_factor
        * float(h2_production_kg.sum())
        * electrolyzer.water_consumption_kg_per_kg_h2.value
        / water.density_kg_per_m3.value
        * water.price_eur_per_m3.value,
    }
    return {name: _clean_number(value) for name, value in costs.items()}


def _build_annual_cost_expressions(
    *,
    config: ModelConfig,
    data: pd.DataFrame,
    annualization_factor: float,
    pv_capacity_mw: gp.Var,
    wind_capacity_mw: gp.Var,
    electrolyzer_capacity_mw: gp.Var,
    compressor_capacity_mw: gp.Var,
    h2_storage_capacity_kg: gp.Var,
    pv_generation_mwh: gp.tupledict,
    wind_generation_mwh: gp.tupledict,
    grid_import_mwh: gp.tupledict,
    h2_production_kg: gp.tupledict,
) -> dict[str, gp.LinExpr]:
    technology = config.technologies
    pv = technology.pv
    wind = technology.wind_onshore
    electrolyzer = technology.electrolyzer
    compressor = technology.compressor
    storage = technology.h2_storage
    water = technology.water
    hours = range(len(data))

    return {
        "pv_annualized_capex_eur_per_year": pv_capacity_mw
        * pv.capex_eur_per_mw
        * capital_recovery_factor(
            pv.real_wacc_fraction.value, pv.lifetime_years.value
        ),
        "pv_fixed_opex_eur_per_year": pv_capacity_mw
        * pv.fixed_opex_eur_per_mw_year,
        "pv_variable_opex_eur_per_year": annualization_factor
        * gp.quicksum(
            pv_generation_mwh[hour] * pv.variable_opex_eur_per_mwh.value
            for hour in hours
        ),
        "wind_annualized_capex_eur_per_year": wind_capacity_mw
        * wind.capex_eur_per_mw
        * capital_recovery_factor(
            wind.real_wacc_fraction.value, wind.lifetime_years.value
        ),
        "wind_fixed_opex_eur_per_year": wind_capacity_mw
        * wind.fixed_opex_eur_per_mw_year,
        "wind_variable_opex_eur_per_year": annualization_factor
        * gp.quicksum(
            wind_generation_mwh[hour]
            * wind.variable_opex_eur_per_mwh.value
            for hour in hours
        ),
        "electrolyzer_annualized_capex_eur_per_year": electrolyzer_capacity_mw
        * electrolyzer.capex_eur_per_mw
        * capital_recovery_factor(
            electrolyzer.real_wacc_fraction.value,
            electrolyzer.lifetime_years.value,
        ),
        "electrolyzer_fixed_opex_eur_per_year": electrolyzer_capacity_mw
        * electrolyzer.fixed_opex_eur_per_mw_year,
        "compressor_annualized_capex_eur_per_year": compressor_capacity_mw
        * compressor.capex_eur_per_mw
        * capital_recovery_factor(
            compressor.real_wacc_fraction.value,
            compressor.lifetime_years.value,
        ),
        "compressor_fixed_opex_eur_per_year": compressor_capacity_mw
        * compressor.capex_eur_per_mw
        * compressor.fixed_opex_fraction_per_year.value,
        "h2_storage_annualized_capex_eur_per_year": h2_storage_capacity_kg
        * storage.capex_eur_per_kg_h2.value
        * capital_recovery_factor(
            storage.real_wacc_fraction.value, storage.lifetime_years.value
        ),
        "h2_storage_fixed_opex_eur_per_year": h2_storage_capacity_kg
        * storage.capex_eur_per_kg_h2.value
        * storage.fixed_opex_fraction_per_year.value,
        "grid_electricity_eur_per_year": annualization_factor
        * gp.quicksum(
            grid_import_mwh[hour] * data["electricity_price"].iloc[hour]
            for hour in hours
        ),
        "water_eur_per_year": annualization_factor
        * gp.quicksum(
            h2_production_kg[hour]
            * electrolyzer.water_consumption_kg_per_kg_h2.value
            / water.density_kg_per_m3.value
            * water.price_eur_per_m3.value
            for hour in hours
        ),
    }


def _build_hourly_result(
    *,
    data: pd.DataFrame,
    time_step_hours: float,
    h2_survival_fraction: float,
    capacities: dict[str, float],
    pv_generation_mwh: Sequence[float],
    wind_generation_mwh: Sequence[float],
    pv_self_consumption_mwh: Sequence[float],
    wind_self_consumption_mwh: Sequence[float],
    grid_import_mwh: Sequence[float],
    electrolyzer_electricity_mwh: Sequence[float],
    compressor_electricity_mwh: Sequence[float],
    h2_production_kg: Sequence[float],
    h2_storage_level_kg: Sequence[float],
) -> pd.DataFrame:
    number_of_hours = len(data)
    pv_available = (
        capacities["pv_capacity_mw"]
        * data["pv_capacity_factor"].to_numpy(dtype=float)
        * time_step_hours
    )
    wind_available = (
        capacities["wind_capacity_mw"]
        * data["wind_capacity_factor"].to_numpy(dtype=float)
        * time_step_hours
    )
    pv_generation = np.asarray(pv_generation_mwh, dtype=float)
    wind_generation = np.asarray(wind_generation_mwh, dtype=float)
    pv_self_consumption = np.asarray(pv_self_consumption_mwh, dtype=float)
    wind_self_consumption = np.asarray(wind_self_consumption_mwh, dtype=float)
    h2_production = np.asarray(h2_production_kg, dtype=float)
    rf_nbo_electricity = np.asarray(electrolyzer_electricity_mwh, dtype=float) + np.asarray(
        compressor_electricity_mwh, dtype=float
    )

    result = pd.DataFrame(
        {
            "timestamp": data["timestamp"].to_numpy(),
            "pv_available_mwh": pv_available,
            "pv_generation_mwh": pv_generation,
            "pv_self_consumption_mwh": pv_self_consumption,
            "pv_curtailment_mwh": np.maximum(0.0, pv_available - pv_generation),
            "wind_available_mwh": wind_available,
            "wind_generation_mwh": wind_generation,
            "wind_self_consumption_mwh": wind_self_consumption,
            "wind_curtailment_mwh": np.maximum(
                0.0, wind_available - wind_generation
            ),
            "grid_import_mwh": np.asarray(grid_import_mwh, dtype=float),
            "renewable_surplus_mwh": np.maximum(
                0.0,
                pv_generation
                + wind_generation
                - pv_self_consumption
                - wind_self_consumption,
            ),
            "eligible_renewable_electricity_mwh": (
                pv_generation + wind_generation
            ),
            "rf_nbo_electricity_mwh": rf_nbo_electricity,
            "electrolyzer_electricity_mwh": np.asarray(
                electrolyzer_electricity_mwh, dtype=float
            ),
            "compressor_electricity_mwh": np.asarray(
                compressor_electricity_mwh, dtype=float
            ),
            "h2_production_kg": h2_production,
            "h2_after_compression_kg": h2_production * h2_survival_fraction,
            "h2_loss_kg": h2_production * (1.0 - h2_survival_fraction),
            "h2_demand_kg": data["h2_demand"].to_numpy(dtype=float),
            "h2_storage_level_kg": np.asarray(h2_storage_level_kg, dtype=float),
            "electricity_price_eur_per_mwh": data[
                "electricity_price"
            ].to_numpy(dtype=float),
            "grid_emission_factor_kg_co2e_per_mwh": data[
                "grid_emission_factor"
            ].to_numpy(dtype=float),
        }
    )
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].map(_clean_number)
    result.attrs["time_zone"] = "UTC"
    result.attrs["frequency"] = "1h"
    return result


def _add_regulatory_emission_columns(
    hourly_operation: pd.DataFrame,
    scenario: Scenario,
) -> pd.DataFrame:
    """Trenne physische Netz- von regulatorisch angerechneten Emissionen.

    Im allgemeinen Art.-4(4)-Pfad erhält zeitlich zugeordneter erneuerbarer
    Strom gemäß Delegierter Verordnung (EU) 2023/1185 einen Emissionsfaktor
    von null. Nur eine innerhalb des Korrelationsfensters ungedeckte
    RFNBO-Strommenge wird mit dem stündlichen Netzfaktor bewertet.
    """

    result = hourly_operation.copy()
    grid_import = result["grid_import_mwh"].to_numpy(dtype=float)
    emission_factor = result[
        "grid_emission_factor_kg_co2e_per_mwh"
    ].to_numpy(dtype=float)
    rf_nbo_electricity = result["rf_nbo_electricity_mwh"].to_numpy(dtype=float)
    regulatory_non_renewable = np.zeros(len(result), dtype=float)

    if scenario is Scenario.REFERENCE:
        regulatory_non_renewable = grid_import.copy()
    elif scenario in RED_TEMPORAL_SCENARIOS:
        temporal_mode = (
            TemporalCorrelation.MONTHLY
            if scenario is Scenario.RED_MONTHLY
            else TemporalCorrelation.HOURLY
        )
        periods = build_correlation_periods(result["timestamp"], temporal_mode)
        eligible = result[
            "eligible_renewable_electricity_mwh"
        ].to_numpy(dtype=float)
        period_values = periods.to_numpy()
        for period in periods.drop_duplicates().tolist():
            indices = np.flatnonzero(period_values == period)
            deficit = max(
                0.0,
                float(rf_nbo_electricity[indices].sum() - eligible[indices].sum()),
            )
            if deficit <= BALANCE_ABSOLUTE_TOLERANCE:
                continue
            period_grid_import = float(grid_import[indices].sum())
            if period_grid_import <= BALANCE_ABSOLUTE_TOLERANCE:
                raise HydrogenOptimizationError(
                    "Die regulatorische Strombilanz weist eine ungedeckte "
                    "Menge ohne zuordenbaren Netzbezug auf."
                )
            if deficit > period_grid_import + BALANCE_ABSOLUTE_TOLERANCE:
                raise HydrogenOptimizationError(
                    "Die regulatorisch nicht erneuerbare Strommenge ist größer "
                    "als der physische Netzbezug."
                )
            regulatory_non_renewable[indices] = (
                grid_import[indices] * deficit / period_grid_import
            )

    result["operational_grid_emissions_kg_co2e"] = (
        grid_import * emission_factor
    )
    result["regulatory_non_renewable_electricity_mwh"] = (
        regulatory_non_renewable
    )
    result["regulatory_renewable_electricity_mwh"] = np.maximum(
        0.0, rf_nbo_electricity - regulatory_non_renewable
    )
    result["regulatory_emissions_kg_co2e"] = (
        regulatory_non_renewable * emission_factor
    )
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].map(_clean_number)
    return result


def _calculate_regulatory_emission_diagnostics(
    hourly_operation: pd.DataFrame,
    *,
    annualization_factor: float,
    annual_h2_delivered_kg: float,
) -> dict[str, float | bool]:
    """Berechne die RED-III-Produktintensität auf H2-LHV-Basis."""

    annual_non_renewable_electricity = _clean_number(
        float(
            hourly_operation["regulatory_non_renewable_electricity_mwh"].sum()
            * annualization_factor
        )
    )
    annual_emissions = _clean_number(
        float(
            hourly_operation["regulatory_emissions_kg_co2e"].sum()
            * annualization_factor
        )
    )
    intensity_kg_per_kg = float(annual_emissions / annual_h2_delivered_kg)
    intensity_g_per_mj = float(
        intensity_kg_per_kg
        * 1_000.0
        / DEFAULT_RED_III_PARAMETERS.h2_lhv_mj_per_kg
    )
    savings_fraction = calculate_ghg_savings_fraction(intensity_g_per_mj)
    compliant = bool(
        intensity_kg_per_kg
        <= DEFAULT_RED_III_PARAMETERS.maximum_product_intensity_kg_co2e_per_kg_h2
        + NUMERICAL_ZERO_TOLERANCE
    )
    return {
        "annual_regulatory_non_renewable_electricity_mwh": (
            annual_non_renewable_electricity
        ),
        "annual_regulatory_emissions_kg_co2e": annual_emissions,
        "regulatory_emission_intensity_kg_co2e_per_kg_h2": intensity_kg_per_kg,
        "regulatory_emission_intensity_g_co2e_per_mj_h2": intensity_g_per_mj,
        "red_iii_ghg_savings_fraction": savings_fraction,
        "red_iii_ghg_compliant": compliant,
    }


def _monthly_hour_groups(data: pd.DataFrame) -> list[tuple[str, list[int]]]:
    """Ordne jede Modellstunde genau einem UTC-Kalendermonat zu."""

    periods = build_correlation_periods(
        data["timestamp"], TemporalCorrelation.MONTHLY
    )
    groups: list[tuple[str, list[int]]] = []
    for period in periods.drop_duplicates().tolist():
        indices = np.flatnonzero(periods.to_numpy() == period).astype(int).tolist()
        groups.append((str(period), indices))
    return groups


def _calculate_red_temporal_diagnostics(
    hourly_operation: pd.DataFrame,
    scenario: Scenario,
    *,
    regulatory_intensity_kg_co2e_per_kg_h2: float,
) -> dict[str, str | bool | int | float | None]:
    """Prüfe eine implementierte Zeitkorrelation nochmals ex post."""

    temporal_mode_by_scenario = {
        Scenario.RED_MONTHLY: TemporalCorrelation.MONTHLY,
        Scenario.RED_HOURLY: TemporalCorrelation.HOURLY,
    }
    temporal_mode = temporal_mode_by_scenario.get(scenario)
    if temporal_mode is None:
        return {
            "red_iii_temporal_mode": None,
            "red_iii_temporal_compliant": None,
            "red_iii_correlation_periods": 0,
            "max_red_iii_temporal_deficit_mwh": 0.0,
        }

    check = assess_red_iii_compliance(
        operation_from_optimizer_output(hourly_operation),
        temporal_mode=temporal_mode,
        evidence=EligibilityEvidence(
            additionality_verified=True,
            geographical_correlation_verified=True,
            exclusive_allocation_verified=True,
        ),
        product_intensity_kg_co2e_per_kg_h2=(
            regulatory_intensity_kg_co2e_per_kg_h2
        ),
        tolerance_mwh=BALANCE_ABSOLUTE_TOLERANCE,
    )
    minimum_balance = float(check.period_results["balance_mwh"].min())
    maximum_deficit = _clean_number(max(0.0, -minimum_balance))
    if not check.temporal_compliant:
        mode_label = (
            "monatliche"
            if temporal_mode is TemporalCorrelation.MONTHLY
            else "stündliche"
        )
        raise HydrogenOptimizationError(
            f"Die unabhängige Prüfung bestätigt die {mode_label} "
            "RED-III-Zeitkorrelation des Solverergebnisses nicht."
        )
    if not check.ghg_compliant:
        raise HydrogenOptimizationError(
            "Die regulatorische H2-Emissionsintensität überschreitet den "
            "RED-III-Grenzwert von "
            f"{DEFAULT_RED_III_PARAMETERS.maximum_product_intensity_kg_co2e_per_kg_h2:.3f} "
            "kg CO2e/kg H2."
        )
    return {
        "red_iii_temporal_mode": temporal_mode.value,
        "red_iii_temporal_compliant": True,
        "red_iii_correlation_periods": len(check.period_results),
        "max_red_iii_temporal_deficit_mwh": maximum_deficit,
    }


def calculate_result_diagnostics(
    hourly_operation: pd.DataFrame,
    *,
    annual_costs: Mapping[str, float],
    objective_eur_per_year: float,
) -> dict[str, float]:
    """Berechne Strom-, H2- und Kostenbilanzfehler aus exportierbaren Ergebnissen."""

    if not isinstance(hourly_operation, pd.DataFrame) or hourly_operation.empty:
        raise ValueError("hourly_operation muss ein nichtleerer DataFrame sein.")
    required_columns = {
        "pv_generation_mwh",
        "wind_generation_mwh",
        "grid_import_mwh",
        "electrolyzer_electricity_mwh",
        "compressor_electricity_mwh",
        "h2_after_compression_kg",
        "h2_demand_kg",
        "h2_storage_level_kg",
    }
    missing = sorted(required_columns - set(hourly_operation.columns))
    if missing:
        raise ValueError("Für die Ergebnisprüfung fehlen Spalten: " + ", ".join(missing) + ".")

    pv_supply_column = (
        "pv_self_consumption_mwh"
        if "pv_self_consumption_mwh" in hourly_operation.columns
        else "pv_generation_mwh"
    )
    wind_supply_column = (
        "wind_self_consumption_mwh"
        if "wind_self_consumption_mwh" in hourly_operation.columns
        else "wind_generation_mwh"
    )
    electricity_supply = (
        hourly_operation[pv_supply_column]
        + hourly_operation[wind_supply_column]
        + hourly_operation["grid_import_mwh"]
    )
    electricity_use = (
        hourly_operation["electrolyzer_electricity_mwh"]
        + hourly_operation["compressor_electricity_mwh"]
    )
    electricity_residual = electricity_supply - electricity_use

    previous_storage = hourly_operation["h2_storage_level_kg"].shift(1)
    previous_storage.iloc[0] = hourly_operation["h2_storage_level_kg"].iloc[-1]
    hydrogen_residual = (
        hourly_operation["h2_storage_level_kg"]
        - previous_storage
        - hourly_operation["h2_after_compression_kg"]
        + hourly_operation["h2_demand_kg"]
    )
    cost_residual = float(sum(float(value) for value in annual_costs.values())) - float(
        objective_eur_per_year
    )
    return {
        "max_electricity_balance_residual_mwh": _clean_number(
            float(electricity_residual.abs().max())
        ),
        "max_hydrogen_balance_residual_kg": _clean_number(
            float(hydrogen_residual.abs().max())
        ),
        "annual_cost_balance_residual_eur_per_year": _clean_number(
            abs(cost_residual)
        ),
    }


def _validate_result_diagnostics(
    diagnostics: Mapping[str, float],
    *,
    objective_eur_per_year: float,
) -> None:
    if diagnostics["max_electricity_balance_residual_mwh"] > BALANCE_ABSOLUTE_TOLERANCE:
        raise HydrogenOptimizationError("Die Strombilanz des Solverergebnisses schließt nicht.")
    if diagnostics["max_hydrogen_balance_residual_kg"] > BALANCE_ABSOLUTE_TOLERANCE:
        raise HydrogenOptimizationError("Die H2-Bilanz des Solverergebnisses schließt nicht.")
    cost_tolerance = COST_BALANCE_RELATIVE_TOLERANCE * max(
        1.0, abs(float(objective_eur_per_year))
    )
    if diagnostics["annual_cost_balance_residual_eur_per_year"] > cost_tolerance:
        raise HydrogenOptimizationError("Die Summe der Kostenkomponenten entspricht nicht der Zielfunktion.")


def _expression_value(expression: gp.LinExpr | float) -> float:
    if isinstance(expression, (int, float)):
        return float(expression)
    return float(expression.getValue())


def _clean_number(value: float) -> float:
    numeric = float(value)
    return 0.0 if abs(numeric) < NUMERICAL_ZERO_TOLERANCE else numeric


__all__ = [
    "BALANCE_ABSOLUTE_TOLERANCE",
    "HydrogenModelInfeasibleError",
    "HydrogenOptimizationError",
    "HydrogenOptimizationResult",
    "capital_recovery_factor",
    "calculate_result_diagnostics",
    "optimize_hydrogen_system",
]
