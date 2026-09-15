"""Unabhängige Abschlussprüfung gespeicherter H2-Szenarienergebnisse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd


CHECKS_FILENAME: Final[str] = "validation_checks.csv"
SAMPLES_FILENAME: Final[str] = "validation_samples.csv"
REPORT_FILENAME: Final[str] = "validation_report.json"
COMPARISON_FILENAME: Final[str] = "scenario_comparison.csv"
ELECTRICITY_TOLERANCE_MWH: Final[float] = 1e-6
HYDROGEN_TOLERANCE_KG: Final[float] = 1e-4
COST_TOLERANCE_EUR_PER_YEAR: Final[float] = 0.05
RELATIVE_TOLERANCE: Final[float] = 1e-8
COST_COLUMNS: Final[tuple[str, ...]] = (
    "pv_annualized_capex_eur_per_year",
    "pv_fixed_opex_eur_per_year",
    "pv_variable_opex_eur_per_year",
    "wind_annualized_capex_eur_per_year",
    "wind_fixed_opex_eur_per_year",
    "wind_variable_opex_eur_per_year",
    "electrolyzer_annualized_capex_eur_per_year",
    "electrolyzer_fixed_opex_eur_per_year",
    "compressor_annualized_capex_eur_per_year",
    "compressor_fixed_opex_eur_per_year",
    "h2_storage_annualized_capex_eur_per_year",
    "h2_storage_fixed_opex_eur_per_year",
    "grid_electricity_eur_per_year",
    "water_eur_per_year",
)


@dataclass(frozen=True, slots=True)
class ValidationArtifacts:
    output_directory: Path
    checks_path: Path
    samples_path: Path
    report_path: Path
    all_checks_passed: bool


def validate_h2_results(
    results_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
    expected_hours: int = 8_760,
    overwrite: bool = False,
) -> ValidationArtifacts:
    """Rechne zentrale Bilanzen unabhängig aus den Exportdateien nach."""

    if isinstance(expected_hours, bool) or not isinstance(expected_hours, int):
        raise TypeError("expected_hours muss eine ganze Zahl sein.")
    if expected_hours <= 0:
        raise ValueError("expected_hours muss positiv sein.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite muss ein boolescher Wert sein.")

    results_root = Path(results_directory).expanduser().resolve()
    comparison_path = results_root / COMPARISON_FILENAME
    if not comparison_path.is_file():
        raise FileNotFoundError(f"Szenarienvergleich nicht gefunden: {comparison_path}")
    output_root = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else results_root / "validation"
    )
    paths = {
        "checks": output_root / CHECKS_FILENAME,
        "samples": output_root / SAMPLES_FILENAME,
        "report": output_root / REPORT_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Validierungsergebnisse existieren bereits: "
            + ", ".join(path.name for path in existing)
            + ". Nutze --overwrite zum Ersetzen."
        )

    comparison = pd.read_csv(comparison_path)
    required_comparison = {
        "scenario_id",
        "scenario",
        "result_directory",
        "input_sha256",
        "annual_h2_delivered_kg",
    }
    missing = sorted(required_comparison.difference(comparison.columns))
    if missing:
        raise ValueError(
            "Szenarienvergleich enthält nicht alle Pflichtspalten: " + ", ".join(missing)
        )
    if comparison.empty:
        raise ValueError("Szenarienvergleich darf nicht leer sein.")

    checks: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    input_hashes = comparison["input_sha256"].astype(str)
    _add_check(
        checks,
        scenario="all",
        check_id="same_source_input_hash",
        value=float(input_hashes.nunique()),
        limit="= 1",
        passed=input_hashes.nunique() == 1,
        detail="Alle Szenarien müssen auf derselben Standort-Eingabedatei beruhen.",
    )
    expected_core = {"reference", "red_monthly", "red_hourly"}
    actual_scenarios = set(comparison["scenario"].astype(str))
    _add_check(
        checks,
        scenario="all",
        check_id="core_scenarios_present",
        value=float(len(actual_scenarios.intersection(expected_core))),
        limit="= 3",
        passed=expected_core.issubset(actual_scenarios),
        detail="S0, S1 und S2 müssen im Vergleich enthalten sein.",
    )

    for comparison_row in comparison.itertuples(index=False):
        scenario = str(comparison_row.scenario)
        scenario_directory = results_root / str(comparison_row.result_directory)
        summary_path = scenario_directory / "summary.csv"
        hourly_path = scenario_directory / "hourly_operation.csv"
        metadata_path = scenario_directory / "run_metadata.json"
        validated_input_path = scenario_directory / "validated_input.csv"
        for path in (summary_path, hourly_path, metadata_path, validated_input_path):
            if not path.is_file():
                raise FileNotFoundError(f"Ergebnisdatei fehlt: {path}")

        summary = pd.read_csv(summary_path)
        hourly = pd.read_csv(hourly_path)
        validated_input = pd.read_csv(validated_input_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if len(summary) != 1:
            raise ValueError(f"{summary_path} muss genau eine Zeile enthalten.")
        summary_row = summary.iloc[0]
        parameters = metadata.get("model_parameters", {})
        annualization_factor = float(summary_row["annualization_factor"])

        _check_time_axis(
            checks,
            scenario=scenario,
            hourly=hourly,
            validated_input=validated_input,
            expected_hours=expected_hours,
        )
        _add_check(
            checks,
            scenario=scenario,
            check_id="summary_input_hash_matches_comparison",
            value=float(
                str(summary_row["input_sha256"]) == str(comparison_row.input_sha256)
            ),
            limit="= 1",
            passed=str(summary_row["input_sha256"]) == str(comparison_row.input_sha256),
            detail="Hash aus Einzellauf und Szenarienvergleich.",
        )

        electric_residual = (
            hourly["pv_self_consumption_mwh"]
            + hourly["wind_self_consumption_mwh"]
            + hourly["grid_import_mwh"]
            - hourly["electrolyzer_electricity_mwh"]
            - hourly["compressor_electricity_mwh"]
        )
        storage_previous = hourly["h2_storage_level_kg"].shift(1)
        storage_previous.iloc[0] = hourly["h2_storage_level_kg"].iloc[-1]
        hydrogen_residual = (
            hourly["h2_storage_level_kg"]
            - storage_previous
            - hourly["h2_after_compression_kg"]
            + hourly["h2_demand_kg"]
        )
        _max_abs_check(
            checks,
            scenario,
            "hourly_electricity_balance",
            electric_residual,
            ELECTRICITY_TOLERANCE_MWH,
            "PV-Eigenverbrauch + Wind-Eigenverbrauch + Netz = Elektrolyseur + Kompressor.",
        )
        _max_abs_check(
            checks,
            scenario,
            "cyclic_hydrogen_storage_balance",
            hydrogen_residual,
            HYDROGEN_TOLERANCE_KG,
            "Speicherstand einschließlich zyklischem Übergang von Stunde 8760 zu Stunde 1.",
        )

        specific_electricity = _parameter_value(
            parameters,
            "technologies.electrolyzer.specific_electricity_kwh_per_kg_h2",
        ) / 1_000.0
        compressor_specific = (
            _parameter_value(
                parameters,
                "technologies.compressor.isentropic_energy_kwh_per_kg_h2",
            )
            / _parameter_value(
                parameters,
                "technologies.compressor.isentropic_efficiency_fraction",
            )
            / _parameter_value(
                parameters,
                "technologies.compressor.mechanical_efficiency_fraction",
            )
            / 1_000.0
        )
        h2_loss_fraction = _parameter_value(
            parameters, "technologies.compressor.h2_loss_fraction"
        )
        _max_abs_check(
            checks,
            scenario,
            "electrolyzer_conversion",
            hourly["h2_production_kg"]
            - hourly["electrolyzer_electricity_mwh"] / specific_electricity,
            HYDROGEN_TOLERANCE_KG,
            "H2-Produktion aus Elektrolyseur-Strom und spezifischem Strombedarf.",
        )
        _max_abs_check(
            checks,
            scenario,
            "compressor_electricity",
            hourly["compressor_electricity_mwh"]
            - hourly["h2_production_kg"] * compressor_specific,
            ELECTRICITY_TOLERANCE_MWH,
            "Verdichtungsstrom aus H2-Produktion und Wirkungsgraden.",
        )
        _max_abs_check(
            checks,
            scenario,
            "compressor_hydrogen_loss",
            hourly["h2_after_compression_kg"]
            - hourly["h2_production_kg"] * (1.0 - h2_loss_fraction),
            HYDROGEN_TOLERANCE_KG,
            "H2-Menge nach dem modellierten Verdichtungsverlust.",
        )
        _max_upper_check(
            checks,
            scenario,
            "pv_availability",
            hourly["pv_generation_mwh"] - hourly["pv_available_mwh"],
            ELECTRICITY_TOLERANCE_MWH,
            "PV-Erzeugung darf die verfügbare Energiemenge nicht überschreiten.",
        )
        _max_upper_check(
            checks,
            scenario,
            "wind_availability",
            hourly["wind_generation_mwh"] - hourly["wind_available_mwh"],
            ELECTRICITY_TOLERANCE_MWH,
            "Winderzeugung darf die verfügbare Energiemenge nicht überschreiten.",
        )
        _max_upper_check(
            checks,
            scenario,
            "storage_capacity",
            hourly["h2_storage_level_kg"] - float(summary_row["h2_storage_capacity_kg"]),
            HYDROGEN_TOLERANCE_KG,
            "Speicherstand darf die optimierte Kapazität nicht überschreiten.",
        )

        cost_sum = float(summary_row[list(COST_COLUMNS)].sum())
        objective = float(summary_row["objective_eur_per_year"])
        cost_residual = cost_sum - objective
        _add_check(
            checks,
            scenario=scenario,
            check_id="annual_cost_sum",
            value=abs(cost_residual),
            limit=f"<= {COST_TOLERANCE_EUR_PER_YEAR} EUR/a",
            passed=abs(cost_residual) <= COST_TOLERANCE_EUR_PER_YEAR,
            detail=f"Kostenkomponenten minus Zielfunktion: {cost_residual:.9g} EUR/a.",
        )
        delivered = float(summary_row["annual_h2_delivered_kg"])
        lcoh_residual = float(summary_row["lcoh_eur_per_kg_h2"]) - objective / delivered
        _add_check(
            checks,
            scenario=scenario,
            check_id="lcoh_definition",
            value=abs(lcoh_residual),
            limit=f"<= {RELATIVE_TOLERANCE} EUR/kg",
            passed=abs(lcoh_residual) <= RELATIVE_TOLERANCE,
            detail="LCOH = Jahreskosten / jährlich ausgelieferte H2-Menge.",
        )
        delivered_from_hours = float(hourly["h2_demand_kg"].sum()) * annualization_factor
        delivery_residual = delivered_from_hours - delivered
        _add_check(
            checks,
            scenario=scenario,
            check_id="annual_hydrogen_delivery",
            value=abs(delivery_residual),
            limit=f"<= {HYDROGEN_TOLERANCE_KG} kg/a",
            passed=abs(delivery_residual) <= HYDROGEN_TOLERANCE_KG,
            detail="Jahresnachfrage aus Stundenwerten und Annualisierungsfaktor.",
        )

        operational_emission_residual = (
            hourly["operational_grid_emissions_kg_co2e"]
            - hourly["grid_import_mwh"] * hourly["grid_emission_factor_kg_co2e_per_mwh"]
        )
        regulatory_emission_residual = (
            hourly["regulatory_emissions_kg_co2e"]
            - hourly["regulatory_non_renewable_electricity_mwh"]
            * hourly["grid_emission_factor_kg_co2e_per_mwh"]
        )
        _max_abs_check(
            checks,
            scenario,
            "hourly_operational_emissions",
            operational_emission_residual,
            1e-5,
            "Physischer Netzbezug multipliziert mit dem stündlichen Netzfaktor.",
        )
        _max_abs_check(
            checks,
            scenario,
            "hourly_regulatory_emissions",
            regulatory_emission_residual,
            1e-5,
            "Regulatorisch nicht erneuerbare Strommenge multipliziert mit dem Netzfaktor.",
        )
        annual_operational = (
            float(hourly["operational_grid_emissions_kg_co2e"].sum())
            * annualization_factor
        )
        annual_regulatory = (
            float(hourly["regulatory_emissions_kg_co2e"].sum())
            * annualization_factor
        )
        for check_id, calculated, exported in (
            (
                "annual_operational_emissions",
                annual_operational,
                float(summary_row["annual_grid_emissions_kg_co2e"]),
            ),
            (
                "annual_regulatory_emissions",
                annual_regulatory,
                float(summary_row["annual_regulatory_emissions_kg_co2e"]),
            ),
        ):
            residual = calculated - exported
            tolerance = max(1e-4, RELATIVE_TOLERANCE * max(abs(exported), 1.0))
            _add_check(
                checks,
                scenario=scenario,
                check_id=check_id,
                value=abs(residual),
                limit=f"<= {tolerance:.6g} kg CO2e/a",
                passed=abs(residual) <= tolerance,
                detail="Unabhängige Summe der stündlichen Emissionsspalte.",
            )

        renewable = hourly["pv_generation_mwh"] + hourly["wind_generation_mwh"]
        rfnbo_use = hourly["rf_nbo_electricity_mwh"]
        if scenario == "red_monthly":
            timestamps = pd.to_datetime(hourly["timestamp"], utc=True)
            monthly_margin = (renewable - rfnbo_use).groupby(
                timestamps.dt.strftime("%Y-%m")
            ).sum()
            minimum_margin = float(monthly_margin.min())
            _add_check(
                checks,
                scenario=scenario,
                check_id="red_monthly_energy_correlation",
                value=minimum_margin,
                limit=f">= {-ELECTRICITY_TOLERANCE_MWH} MWh",
                passed=minimum_margin >= -ELECTRICITY_TOLERANCE_MWH,
                detail="Kleinste monatliche Differenz Erzeugung minus RFNBO-Strombedarf.",
            )
        elif scenario == "red_hourly":
            minimum_margin = float((renewable - rfnbo_use).min())
            _add_check(
                checks,
                scenario=scenario,
                check_id="red_hourly_energy_correlation",
                value=minimum_margin,
                limit=f">= {-ELECTRICITY_TOLERANCE_MWH} MWh",
                passed=minimum_margin >= -ELECTRICITY_TOLERANCE_MWH,
                detail="Kleinste stündliche Differenz Erzeugung minus RFNBO-Strombedarf.",
            )

        _check_parameter_documentation(checks, scenario, parameters)
        samples.extend(
            _sample_hourly_balances(
                scenario,
                hourly,
                electric_residual,
                hydrogen_residual,
                storage_previous,
            )
        )

    checks_frame = pd.DataFrame(checks)
    samples_frame = pd.DataFrame(samples)
    all_passed = bool(checks_frame["passed"].all())
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(checks_frame, paths["checks"])
    _write_csv_atomic(samples_frame, paths["samples"])
    _write_json_atomic(
        {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Unabhängige Ex-post-Nachrechnung aus exportierten CSV- und JSON-Dateien",
            "source_comparison": {
                "path": str(comparison_path),
                "sha256": _sha256(comparison_path),
            },
            "expected_hours_per_scenario": expected_hours,
            "number_of_scenarios": int(len(comparison)),
            "number_of_checks": int(len(checks_frame)),
            "passed_checks": int(checks_frame["passed"].sum()),
            "failed_checks": int((~checks_frame["passed"]).sum()),
            "all_checks_passed": all_passed,
            "tolerances": {
                "electricity_mwh": ELECTRICITY_TOLERANCE_MWH,
                "hydrogen_kg": HYDROGEN_TOLERANCE_KG,
                "annual_cost_eur": COST_TOLERANCE_EUR_PER_YEAR,
                "relative": RELATIVE_TOLERANCE,
            },
            "scope_note": (
                "Die Prüfung validiert numerische Modellbilanzen, Ergebnisexport, "
                "monatliche beziehungsweise stündliche Strommengen-Korrelation und "
                "den implementierten THG-Teil. Externe Nachweise zu Zusätzlichkeit, "
                "geografischer Korrelation, Gebotszone, Vertrag und Allokation bleiben "
                "außerhalb dieser technischen Validierung."
            ),
            "output_files": {
                "checks": CHECKS_FILENAME,
                "samples": SAMPLES_FILENAME,
                "report": REPORT_FILENAME,
            },
        },
        paths["report"],
    )
    return ValidationArtifacts(
        output_root,
        paths["checks"],
        paths["samples"],
        paths["report"],
        all_passed,
    )


def _check_time_axis(
    checks: list[dict[str, object]],
    *,
    scenario: str,
    hourly: pd.DataFrame,
    validated_input: pd.DataFrame,
    expected_hours: int,
) -> None:
    row_count_ok = len(hourly) == expected_hours == len(validated_input)
    _add_check(
        checks,
        scenario=scenario,
        check_id="hour_count",
        value=float(len(hourly)),
        limit=f"= {expected_hours}",
        passed=row_count_ok,
        detail="Stundenbetrieb und validierte Eingabe müssen gleich lang sein.",
    )
    timestamps = pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce")
    valid_timestamps = not timestamps.isna().any()
    hourly_steps = (
        valid_timestamps
        and timestamps.is_monotonic_increasing
        and not timestamps.duplicated().any()
        and (timestamps.diff().dropna() == pd.Timedelta(hours=1)).all()
    )
    _add_check(
        checks,
        scenario=scenario,
        check_id="continuous_utc_hourly_axis",
        value=float(bool(hourly_steps)),
        limit="= 1",
        passed=bool(hourly_steps),
        detail="Eindeutige, sortierte und lückenlos stündliche UTC-Zeitachse.",
    )


def _check_parameter_documentation(
    checks: list[dict[str, object]],
    scenario: str,
    parameters: object,
) -> None:
    if not isinstance(parameters, dict) or not parameters:
        _add_check(
            checks,
            scenario=scenario,
            check_id="parameter_documentation",
            value=0.0,
            limit="= 1",
            passed=False,
            detail="model_parameters fehlt oder ist leer.",
        )
        return
    complete = all(
        isinstance(entry, dict)
        and str(entry.get("unit", "")).strip()
        and str(entry.get("source", "")).strip()
        for entry in parameters.values()
    )
    _add_check(
        checks,
        scenario=scenario,
        check_id="parameter_documentation",
        value=float(bool(complete)),
        limit="= 1",
        passed=bool(complete),
        detail=f"Einheit und Quelle für {len(parameters)} exportierte Modellparameter.",
    )


def _parameter_value(parameters: object, name: str) -> float:
    if not isinstance(parameters, dict) or name not in parameters:
        raise ValueError(f"Modellparameter fehlt in run_metadata.json: {name}")
    entry = parameters[name]
    if not isinstance(entry, dict) or "value" not in entry:
        raise ValueError(f"Modellparameter ist unvollständig: {name}")
    return float(entry["value"])


def _max_abs_check(
    checks: list[dict[str, object]],
    scenario: str,
    check_id: str,
    residual: pd.Series,
    tolerance: float,
    detail: str,
) -> None:
    value = float(np.max(np.abs(pd.to_numeric(residual).to_numpy(dtype=float))))
    _add_check(
        checks,
        scenario=scenario,
        check_id=check_id,
        value=value,
        limit=f"<= {tolerance}",
        passed=value <= tolerance,
        detail=detail,
    )


def _max_upper_check(
    checks: list[dict[str, object]],
    scenario: str,
    check_id: str,
    excess: pd.Series,
    tolerance: float,
    detail: str,
) -> None:
    value = float(pd.to_numeric(excess).max())
    _add_check(
        checks,
        scenario=scenario,
        check_id=check_id,
        value=value,
        limit=f"<= {tolerance}",
        passed=value <= tolerance,
        detail=detail,
    )


def _sample_hourly_balances(
    scenario: str,
    hourly: pd.DataFrame,
    electric_residual: pd.Series,
    hydrogen_residual: pd.Series,
    storage_previous: pd.Series,
) -> list[dict[str, object]]:
    renewable_self = hourly["pv_self_consumption_mwh"] + hourly["wind_self_consumption_mwh"]
    candidate_indices = [
        0,
        int(hourly["grid_import_mwh"].astype(float).idxmax()),
        int(renewable_self.astype(float).idxmax()),
        int(hourly["h2_storage_level_kg"].astype(float).idxmax()),
    ]
    selected_indices: list[int] = []
    for index in candidate_indices:
        if index not in selected_indices:
            selected_indices.append(index)
        if len(selected_indices) == 3:
            break
    for index in np.linspace(0, len(hourly) - 1, 3, dtype=int):
        if len(selected_indices) == 3:
            break
        if int(index) not in selected_indices:
            selected_indices.append(int(index))

    rows: list[dict[str, object]] = []
    for index in selected_indices:
        row = hourly.iloc[index]
        rows.append(
            {
                "scenario": scenario,
                "timestamp": row["timestamp"],
                "pv_self_consumption_mwh": float(row["pv_self_consumption_mwh"]),
                "wind_self_consumption_mwh": float(row["wind_self_consumption_mwh"]),
                "grid_import_mwh": float(row["grid_import_mwh"]),
                "electrolyzer_electricity_mwh": float(row["electrolyzer_electricity_mwh"]),
                "compressor_electricity_mwh": float(row["compressor_electricity_mwh"]),
                "electricity_balance_residual_mwh": float(electric_residual.iloc[index]),
                "previous_storage_kg": float(storage_previous.iloc[index]),
                "h2_after_compression_kg": float(row["h2_after_compression_kg"]),
                "h2_demand_kg": float(row["h2_demand_kg"]),
                "storage_level_kg": float(row["h2_storage_level_kg"]),
                "hydrogen_balance_residual_kg": float(hydrogen_residual.iloc[index]),
            }
        )
    return rows


def _add_check(
    checks: list[dict[str, object]],
    *,
    scenario: str,
    check_id: str,
    value: float,
    limit: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append(
        {
            "scenario": scenario,
            "check_id": check_id,
            "value": value,
            "limit": limit,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv_atomic(data: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        data.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(data: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prüft gespeicherte S0-, S1- und S2-Ergebnisse unabhängig nach."
    )
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-hours", type=int, default=8_760)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        print("\nEs wurde noch keine Ergebnisvalidierung gestartet.")
        return 0
    args = parser.parse_args(arguments)
    try:
        artifacts = validate_h2_results(
            args.results_dir,
            output_directory=args.output_dir,
            expected_hours=args.expected_hours,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    checks = pd.read_csv(artifacts.checks_path)
    print(
        f"Validierung: {int(checks['passed'].sum())} von {len(checks)} Prüfungen bestanden."
    )
    print(f"Prüftabelle: {artifacts.checks_path}")
    print(f"Stichproben: {artifacts.samples_path}")
    print(f"Bericht: {artifacts.report_path}")
    return 0 if artifacts.all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ValidationArtifacts",
    "build_argument_parser",
    "main",
    "validate_h2_results",
]
