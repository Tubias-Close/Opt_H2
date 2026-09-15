"""Physikalische und ökonomische Tests des linearen H2-Basiskerns."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from config_h2 import DEFAULT_CONFIG, ScalarParameter, Scenario
from opt_hydrogen_functions import (
    HydrogenModelInfeasibleError,
    capital_recovery_factor,
    calculate_result_diagnostics,
    optimize_hydrogen_system,
)
from red_iii_data import DEFAULT_RED_III_PARAMETERS


def hourly_case(
    *,
    pv_capacity_factor: float | np.ndarray,
    wind_capacity_factor: float | np.ndarray = 0.0,
    electricity_price: float = 100.0,
    grid_emission_factor: float = 300.0,
    h2_demand: float | np.ndarray = 100.0,
) -> pd.DataFrame:
    def values(value: float | np.ndarray) -> np.ndarray:
        if np.isscalar(value):
            return np.full(24, float(value))
        array = np.asarray(value, dtype=float)
        if array.shape != (24,):
            raise ValueError("Testprofile müssen genau 24 Werte enthalten.")
        return array

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=24, freq="h"),
            "pv_capacity_factor": values(pv_capacity_factor),
            "wind_capacity_factor": values(wind_capacity_factor),
            "electricity_price": values(electricity_price),
            "grid_emission_factor": values(grid_emission_factor),
            "h2_demand": values(h2_demand),
        }
    )


def off_grid_config():
    return replace(DEFAULT_CONFIG, scenario=Scenario.OFF_GRID)


def red_monthly_config():
    return replace(DEFAULT_CONFIG, scenario=Scenario.RED_MONTHLY)


def red_hourly_config():
    return replace(DEFAULT_CONFIG, scenario=Scenario.RED_HOURLY)


def test_capital_recovery_factor_matches_formula() -> None:
    expected = 0.05 * 1.05**25 / (1.05**25 - 1.0)
    assert capital_recovery_factor(0.05, 25) == pytest.approx(expected)


def test_capital_recovery_factor_handles_zero_wacc() -> None:
    assert capital_recovery_factor(0.0, 20) == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("wacc", "lifetime"),
    [(-0.01, 20), (1.0, 20), (0.05, 0), (0.05, -1)],
)
def test_capital_recovery_factor_rejects_invalid_values(
    wacc: float, lifetime: float
) -> None:
    with pytest.raises(ValueError):
        capital_recovery_factor(wacc, lifetime)


def test_constant_pv_can_supply_off_grid_demand() -> None:
    result = optimize_hydrogen_system(
        hourly_case(pv_capacity_factor=1.0),
        config=off_grid_config(),
    )

    assert result.solver_status == "optimal"
    assert result.capacities["pv_capacity_mw"] > 0.0
    assert result.capacities["wind_capacity_mw"] == pytest.approx(0.0, abs=1e-7)
    assert result.capacities["h2_storage_capacity_kg"] == pytest.approx(
        0.0, abs=1e-7
    )
    assert result.hourly_operation["grid_import_mwh"].sum() == pytest.approx(
        0.0, abs=1e-7
    )
    _assert_hourly_balances(result.hourly_operation)


def test_grid_supplies_demand_when_renewables_are_unavailable() -> None:
    result = optimize_hydrogen_system(
        hourly_case(pv_capacity_factor=0.0, wind_capacity_factor=0.0)
    )

    assert result.capacities["pv_capacity_mw"] == pytest.approx(0.0, abs=1e-7)
    assert result.capacities["wind_capacity_mw"] == pytest.approx(0.0, abs=1e-7)
    assert result.hourly_operation["grid_import_mwh"].sum() > 0.0
    assert result.annual_grid_emissions_kg_co2e > 0.0
    assert result.annual_regulatory_emissions_kg_co2e == pytest.approx(
        result.annual_grid_emissions_kg_co2e
    )
    assert result.regulatory_emission_intensity_kg_co2e_per_kg_h2 == pytest.approx(
        result.operational_emission_intensity_kg_co2e_per_kg_h2
    )
    assert result.optimality_gap_fraction == pytest.approx(0.0)
    assert result.max_electricity_balance_residual_mwh <= 1e-6
    assert result.max_hydrogen_balance_residual_kg <= 1e-6
    assert result.annual_cost_balance_residual_eur_per_year <= 1e-6
    _assert_hourly_balances(result.hourly_operation)


def test_scipy_highs_matches_gurobi_for_same_linear_problem() -> None:
    data = hourly_case(
        pv_capacity_factor=0.0,
        wind_capacity_factor=0.0,
        electricity_price=50.0,
    )
    gurobi_result = optimize_hydrogen_system(data, solver_backend="gurobi")
    scipy_result = optimize_hydrogen_system(data, solver_backend="scipy_highs")

    assert gurobi_result.solver_name == "gurobi"
    assert scipy_result.solver_name == "scipy_highs"
    assert scipy_result.objective_eur_per_year == pytest.approx(
        gurobi_result.objective_eur_per_year, rel=1e-8
    )
    assert scipy_result.lcoh_eur_per_kg_h2 == pytest.approx(
        gurobi_result.lcoh_eur_per_kg_h2, rel=1e-8
    )
    for capacity_name, gurobi_value in gurobi_result.capacities.items():
        assert scipy_result.capacities[capacity_name] == pytest.approx(
            gurobi_value, abs=1e-7
        )


@pytest.mark.parametrize("solver_backend", ["gurobi", "scipy_highs"])
@pytest.mark.parametrize(
    ("threshold_multiplier", "expected_compliant"),
    [(1.0 - 1e-6, True), (1.0 + 1e-6, False)],
)
def test_regulatory_ghg_threshold_is_stable_on_both_solver_paths(
    solver_backend: str,
    threshold_multiplier: float,
    expected_compliant: bool,
) -> None:
    zero_emission_result = optimize_hydrogen_system(
        hourly_case(
            pv_capacity_factor=0.0,
            wind_capacity_factor=0.0,
            grid_emission_factor=0.0,
        ),
        solver_backend=solver_backend,
    )
    electricity_per_delivered_h2 = (
        zero_emission_result.hourly_operation["grid_import_mwh"].sum()
        * zero_emission_result.annualization_factor
        / zero_emission_result.annual_h2_delivered_kg
    )
    factor_at_threshold = (
        DEFAULT_RED_III_PARAMETERS.maximum_product_intensity_kg_co2e_per_kg_h2
        / electricity_per_delivered_h2
    )
    result = optimize_hydrogen_system(
        hourly_case(
            pv_capacity_factor=0.0,
            wind_capacity_factor=0.0,
            grid_emission_factor=factor_at_threshold * threshold_multiplier,
        ),
        solver_backend=solver_backend,
    )

    assert result.red_iii_ghg_compliant is expected_compliant
    assert result.regulatory_emission_intensity_kg_co2e_per_kg_h2 == pytest.approx(
        DEFAULT_RED_III_PARAMETERS.maximum_product_intensity_kg_co2e_per_kg_h2
        * threshold_multiplier,
        rel=1e-9,
    )


def test_scipy_highs_matches_gurobi_for_monthly_red_problem() -> None:
    pv_profile = np.r_[np.ones(12), np.zeros(12)]
    data = hourly_case(
        pv_capacity_factor=pv_profile,
        wind_capacity_factor=0.0,
        electricity_price=0.0,
    )
    gurobi_result = optimize_hydrogen_system(
        data, config=red_monthly_config(), solver_backend="gurobi"
    )
    scipy_result = optimize_hydrogen_system(
        data, config=red_monthly_config(), solver_backend="scipy_highs"
    )

    assert scipy_result.objective_eur_per_year == pytest.approx(
        gurobi_result.objective_eur_per_year, rel=1e-8
    )
    for capacity_name, gurobi_value in gurobi_result.capacities.items():
        assert scipy_result.capacities[capacity_name] == pytest.approx(
            gurobi_value, abs=1e-6
        )
    assert gurobi_result.red_iii_temporal_compliant is True
    assert scipy_result.red_iii_temporal_compliant is True


def test_scipy_highs_matches_gurobi_for_hourly_red_problem() -> None:
    pv_profile = np.tile([1.0, 0.0], 12)
    data = hourly_case(
        pv_capacity_factor=pv_profile,
        wind_capacity_factor=0.0,
        electricity_price=0.0,
    )
    gurobi_result = optimize_hydrogen_system(
        data, config=red_hourly_config(), solver_backend="gurobi"
    )
    scipy_result = optimize_hydrogen_system(
        data, config=red_hourly_config(), solver_backend="scipy_highs"
    )

    assert scipy_result.objective_eur_per_year == pytest.approx(
        gurobi_result.objective_eur_per_year, rel=1e-8
    )
    for capacity_name, gurobi_value in gurobi_result.capacities.items():
        assert scipy_result.capacities[capacity_name] == pytest.approx(
            gurobi_value, abs=1e-6
        )
    assert gurobi_result.red_iii_temporal_mode == "hourly"
    assert scipy_result.red_iii_temporal_mode == "hourly"
    assert gurobi_result.red_iii_temporal_compliant is True
    assert scipy_result.red_iii_temporal_compliant is True


def test_unknown_solver_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="solver_backend"):
        optimize_hydrogen_system(
            hourly_case(pv_capacity_factor=0.0), solver_backend="unknown"
        )


def test_high_and_low_pv_hours_create_curtailment_when_storage_is_expensive() -> None:
    pv_profile = np.tile([0.5, 1.0], 12)
    storage = DEFAULT_CONFIG.technologies.h2_storage
    expensive_storage = replace(
        storage,
        capex_eur_per_kg_h2=ScalarParameter(
            1e9,
            "EUR_2023/kg_H2",
            "Künstlicher Grenzfalltest",
            2023,
        ),
    )
    technologies = replace(
        DEFAULT_CONFIG.technologies,
        h2_storage=expensive_storage,
    )
    config = replace(
        DEFAULT_CONFIG,
        scenario=Scenario.OFF_GRID,
        technologies=technologies,
    )
    result = optimize_hydrogen_system(
        hourly_case(pv_capacity_factor=pv_profile),
        config=config,
    )

    assert result.capacities["h2_storage_capacity_kg"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert result.hourly_operation["pv_curtailment_mwh"].sum() > 0.0
    _assert_hourly_balances(result.hourly_operation)


def test_daytime_pv_uses_hydrogen_storage_for_night_demand() -> None:
    pv_profile = np.r_[np.ones(12), np.zeros(12)]
    result = optimize_hydrogen_system(
        hourly_case(pv_capacity_factor=pv_profile),
        config=off_grid_config(),
    )

    operation = result.hourly_operation
    assert result.capacities["h2_storage_capacity_kg"] > 0.0
    assert operation.loc[12:, "h2_production_kg"].sum() == pytest.approx(
        0.0, abs=1e-7
    )
    assert operation.loc[12:, "h2_storage_level_kg"].max() > 0.0
    _assert_hourly_balances(operation)


def test_off_grid_without_renewable_generation_is_infeasible() -> None:
    with pytest.raises(HydrogenModelInfeasibleError, match="nicht lösbar"):
        optimize_hydrogen_system(
            hourly_case(pv_capacity_factor=0.0, wind_capacity_factor=0.0),
            config=off_grid_config(),
        )


def test_grid_only_cost_and_lcoh_are_reproducible_by_hand() -> None:
    demand_kg_per_hour = 100.0
    electricity_price = 50.0
    result = optimize_hydrogen_system(
        hourly_case(
            pv_capacity_factor=0.0,
            wind_capacity_factor=0.0,
            electricity_price=electricity_price,
            h2_demand=demand_kg_per_hour,
        )
    )

    technology = DEFAULT_CONFIG.technologies
    survival = 1.0 - technology.compressor.h2_loss_fraction.value
    production_kg_per_hour = demand_kg_per_hour / survival
    electrolyzer_mw = (
        production_kg_per_hour
        * technology.electrolyzer.specific_electricity_mwh_per_kg_h2
    )
    compressor_mw = (
        production_kg_per_hour
        * technology.compressor.specific_electricity_mwh_per_kg_h2
    )
    annual_h2 = demand_kg_per_hour * 8_760.0
    annual_production = production_kg_per_hour * 8_760.0

    expected_cost = (
        electrolyzer_mw
        * technology.electrolyzer.capex_eur_per_mw
        * capital_recovery_factor(
            technology.electrolyzer.real_wacc_fraction.value,
            technology.electrolyzer.lifetime_years.value,
        )
        + electrolyzer_mw
        * technology.electrolyzer.fixed_opex_eur_per_mw_year
        + compressor_mw
        * technology.compressor.capex_eur_per_mw
        * capital_recovery_factor(
            technology.compressor.real_wacc_fraction.value,
            technology.compressor.lifetime_years.value,
        )
        + compressor_mw
        * technology.compressor.capex_eur_per_mw
        * technology.compressor.fixed_opex_fraction_per_year.value
        + (electrolyzer_mw + compressor_mw)
        * 8_760.0
        * electricity_price
        + annual_production
        * technology.electrolyzer.water_consumption_kg_per_kg_h2.value
        / technology.water.density_kg_per_m3.value
        * technology.water.price_eur_per_m3.value
    )

    assert result.annualization_factor == pytest.approx(365.0)
    assert result.annual_h2_delivered_kg == pytest.approx(annual_h2)
    assert result.annual_h2_produced_kg == pytest.approx(annual_production)
    assert sum(result.annual_costs.values()) == pytest.approx(
        result.objective_eur_per_year
    )
    assert result.objective_eur_per_year == pytest.approx(expected_cost)
    assert result.lcoh_eur_per_kg_h2 == pytest.approx(expected_cost / annual_h2)


def test_monthly_red_allows_renewable_generation_and_use_in_different_hours() -> None:
    pv_profile = np.r_[np.ones(12), np.zeros(12)]
    result = optimize_hydrogen_system(
        hourly_case(
            pv_capacity_factor=pv_profile,
            wind_capacity_factor=0.0,
            electricity_price=0.0,
        ),
        config=red_monthly_config(),
    )

    operation = result.hourly_operation
    assert operation.loc[12:, "pv_generation_mwh"].sum() == pytest.approx(0.0)
    assert operation.loc[12:, "grid_import_mwh"].sum() > 0.0
    assert result.annual_grid_emissions_kg_co2e > 0.0
    assert result.annual_regulatory_non_renewable_electricity_mwh == pytest.approx(
        0.0, abs=1e-7
    )
    assert result.annual_regulatory_emissions_kg_co2e == pytest.approx(
        0.0, abs=1e-7
    )
    assert result.regulatory_emission_intensity_kg_co2e_per_kg_h2 == pytest.approx(
        0.0, abs=1e-10
    )
    assert result.red_iii_ghg_savings_fraction == pytest.approx(1.0)
    assert result.red_iii_ghg_compliant is True
    assert operation["regulatory_emissions_kg_co2e"].sum() == pytest.approx(
        0.0, abs=1e-7
    )
    assert operation["renewable_surplus_mwh"].sum() > 0.0
    assert (
        operation["eligible_renewable_electricity_mwh"].sum() + 1e-6
        >= operation["rf_nbo_electricity_mwh"].sum()
    )
    assert result.red_iii_temporal_mode == "monthly"
    assert result.red_iii_temporal_compliant is True
    assert result.red_iii_correlation_periods == 1
    assert result.max_red_iii_temporal_deficit_mwh <= 1e-6
    _assert_hourly_balances(operation)


def test_monthly_red_is_infeasible_without_renewable_availability() -> None:
    with pytest.raises(HydrogenModelInfeasibleError, match="nicht lösbar"):
        optimize_hydrogen_system(
            hourly_case(pv_capacity_factor=0.0, wind_capacity_factor=0.0),
            config=red_monthly_config(),
        )


def test_hourly_red_requires_matching_in_every_hour() -> None:
    pv_profile = np.r_[np.ones(12), np.zeros(12)]
    data = hourly_case(
        pv_capacity_factor=pv_profile,
        wind_capacity_factor=0.0,
        electricity_price=0.0,
    )
    monthly_result = optimize_hydrogen_system(
        data,
        config=red_monthly_config(),
        solver_backend="scipy_highs",
    )
    hourly_result = optimize_hydrogen_system(
        data,
        config=red_hourly_config(),
        solver_backend="scipy_highs",
    )

    operation = hourly_result.hourly_operation
    assert np.all(
        operation["eligible_renewable_electricity_mwh"] + 1e-6
        >= operation["rf_nbo_electricity_mwh"]
    )
    assert operation.loc[12:, "rf_nbo_electricity_mwh"].sum() == pytest.approx(
        0.0, abs=1e-6
    )
    assert hourly_result.capacities["h2_storage_capacity_kg"] > 0.0
    assert hourly_result.objective_eur_per_year > monthly_result.objective_eur_per_year
    assert hourly_result.red_iii_temporal_mode == "hourly"
    assert hourly_result.red_iii_temporal_compliant is True
    assert hourly_result.red_iii_correlation_periods == 24
    assert hourly_result.max_red_iii_temporal_deficit_mwh <= 1e-6
    _assert_hourly_balances(operation)


def test_hourly_red_is_infeasible_without_renewable_availability() -> None:
    with pytest.raises(HydrogenModelInfeasibleError, match="nicht lösbar"):
        optimize_hydrogen_system(
            hourly_case(pv_capacity_factor=0.0, wind_capacity_factor=0.0),
            config=red_hourly_config(),
            solver_backend="scipy_highs",
        )


def test_result_diagnostics_detect_a_modified_electricity_balance() -> None:
    result = optimize_hydrogen_system(
        hourly_case(pv_capacity_factor=0.0, wind_capacity_factor=0.0)
    )
    modified = result.hourly_operation.copy()
    modified.loc[3, "grid_import_mwh"] += 0.25

    diagnostics = calculate_result_diagnostics(
        modified,
        annual_costs=result.annual_costs,
        objective_eur_per_year=result.objective_eur_per_year,
    )

    assert diagnostics["max_electricity_balance_residual_mwh"] == pytest.approx(0.25)
    assert diagnostics["max_hydrogen_balance_residual_kg"] <= 1e-6


def _assert_hourly_balances(operation: pd.DataFrame) -> None:
    electricity_supply = (
        operation["pv_self_consumption_mwh"]
        + operation["wind_self_consumption_mwh"]
        + operation["grid_import_mwh"]
    )
    electricity_use = (
        operation["electrolyzer_electricity_mwh"]
        + operation["compressor_electricity_mwh"]
    )
    assert np.allclose(electricity_supply, electricity_use, atol=1e-7)

    previous_storage = operation["h2_storage_level_kg"].shift(1)
    previous_storage.iloc[0] = operation["h2_storage_level_kg"].iloc[-1]
    expected_storage = (
        previous_storage
        + operation["h2_after_compression_kg"]
        - operation["h2_demand_kg"]
    )
    assert np.allclose(
        operation["h2_storage_level_kg"], expected_storage, atol=1e-6
    )
