r"""Kontrollierter Einstiegspunkt für einen einzelnen H2-Modelllauf.

Beispiel im Windows-Terminal::

    python run_single_site_h2.py --input input.csv --scenario reference --output-dir results\testlauf

Der Runner liest ausschließlich die ausdrücklich angegebene CSV-Datei. Er
erzeugt eine validierte Eingabekopie, eine einzeilige Zusammenfassung, den
stündlichen Betrieb und JSON-Metadaten für die Nachvollziehbarkeit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import gurobipy as gp
import pandas as pd
import scipy

from config_h2 import DEFAULT_CONFIG, ModelConfig, Scenario, iter_scalar_parameters
from h2_input_data import HourlyInputError, validate_hourly_input
from opt_hydrogen_functions import (
    HydrogenOptimizationError,
    HydrogenOptimizationResult,
    optimize_hydrogen_system,
)


SUMMARY_FILENAME: Final[str] = "summary.csv"
HOURLY_OPERATION_FILENAME: Final[str] = "hourly_operation.csv"
VALIDATED_INPUT_FILENAME: Final[str] = "validated_input.csv"
RUN_METADATA_FILENAME: Final[str] = "run_metadata.json"
OUTPUT_FILENAMES: Final[tuple[str, ...]] = (
    SUMMARY_FILENAME,
    HOURLY_OPERATION_FILENAME,
    VALIDATED_INPUT_FILENAME,
    RUN_METADATA_FILENAME,
)
SUPPORTED_CLI_SCENARIOS: Final[tuple[str, ...]] = (
    Scenario.REFERENCE.value,
    Scenario.RED_MONTHLY.value,
    Scenario.RED_HOURLY.value,
    Scenario.OFF_GRID.value,
)


class H2RunError(RuntimeError):
    """Fehler beim Laden oder Schreiben eines vollständigen Modelllaufs."""


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Pfade und Ergebnis des erfolgreich abgeschlossenen Modelllaufs."""

    output_directory: Path
    summary_path: Path
    hourly_operation_path: Path
    validated_input_path: Path
    metadata_path: Path
    result: HydrogenOptimizationResult


def run_single_site_h2(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    scenario: Scenario | str = Scenario.REFERENCE,
    config: ModelConfig = DEFAULT_CONFIG,
    overwrite: bool = False,
    solver_output: bool = False,
    solver_backend: str = "auto",
) -> RunArtifacts:
    """Führe einen Modelllauf aus und schreibe seine vier Ergebnisdateien."""

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite muss ein boolescher Wert sein.")
    selected_scenario = _coerce_scenario(scenario)

    source_path = Path(input_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(
            f"Die Eingabedatei wurde nicht gefunden: {source_path}"
        )
    if not source_path.is_file():
        raise H2RunError(f"Der Eingabepfad ist keine Datei: {source_path}")

    try:
        raw_input = pd.read_csv(source_path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise H2RunError(
            f"Die CSV-Eingabedatei konnte nicht gelesen werden: {source_path}"
        ) from exc
    validated_input = validate_hourly_input(raw_input)
    input_sha256 = _sha256(source_path)

    run_config = replace(config, scenario=selected_scenario)
    run_config.validate()
    result = optimize_hydrogen_system(
        validated_input,
        config=run_config,
        solver_output=solver_output,
        solver_backend=solver_backend,
    )

    output_paths = _prepare_output_paths(
        Path(output_directory).expanduser(),
        source_path=source_path,
        overwrite=overwrite,
    )
    summary = _build_summary(
        result=result,
        config=run_config,
        validated_input=validated_input,
        source_path=source_path,
        input_sha256=input_sha256,
    )
    metadata = _build_metadata(
        result=result,
        config=run_config,
        validated_input=validated_input,
        source_path=source_path,
        input_sha256=input_sha256,
    )

    _write_csv_atomic(validated_input, output_paths[VALIDATED_INPUT_FILENAME])
    _write_csv_atomic(summary, output_paths[SUMMARY_FILENAME])
    _write_csv_atomic(
        result.hourly_operation,
        output_paths[HOURLY_OPERATION_FILENAME],
    )
    _write_json_atomic(metadata, output_paths[RUN_METADATA_FILENAME])

    return RunArtifacts(
        output_directory=output_paths[SUMMARY_FILENAME].parent,
        summary_path=output_paths[SUMMARY_FILENAME],
        hourly_operation_path=output_paths[HOURLY_OPERATION_FILENAME],
        validated_input_path=output_paths[VALIDATED_INPUT_FILENAME],
        metadata_path=output_paths[RUN_METADATA_FILENAME],
        result=result,
    )


def _coerce_scenario(scenario: Scenario | str) -> Scenario:
    if isinstance(scenario, Scenario):
        return scenario
    if not isinstance(scenario, str):
        raise TypeError("scenario muss ein String oder ein Scenario-Wert sein.")
    try:
        return Scenario(scenario)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in Scenario)
        raise ValueError(
            f"Unbekanntes Szenario '{scenario}'. Erlaubte Werte: {allowed}."
        ) from exc


def _prepare_output_paths(
    output_directory: Path,
    *,
    source_path: Path,
    overwrite: bool,
) -> dict[str, Path]:
    resolved_output_directory = output_directory.resolve()
    if resolved_output_directory.exists() and not resolved_output_directory.is_dir():
        raise H2RunError(
            f"Der Ausgabepfad ist kein Verzeichnis: {resolved_output_directory}"
        )

    output_paths = {
        filename: resolved_output_directory / filename
        for filename in OUTPUT_FILENAMES
    }
    colliding_outputs = [
        path.name for path in output_paths.values() if path.resolve() == source_path
    ]
    if colliding_outputs:
        raise H2RunError(
            "Die Eingabedatei darf nicht denselben Pfad wie eine Ergebnisdatei haben: "
            + ", ".join(colliding_outputs)
            + "."
        )
    existing_outputs = [path for path in output_paths.values() if path.exists()]
    if existing_outputs and not overwrite:
        names = ", ".join(path.name for path in existing_outputs)
        raise FileExistsError(
            f"Ergebnisdateien existieren bereits ({names}). Nutze --overwrite zum Ersetzen."
        )

    resolved_output_directory.mkdir(parents=True, exist_ok=True)
    return output_paths


def _build_summary(
    *,
    result: HydrogenOptimizationResult,
    config: ModelConfig,
    validated_input: pd.DataFrame,
    source_path: Path,
    input_sha256: str,
) -> pd.DataFrame:
    summary: dict[str, object] = {
        "solver_status": result.solver_status,
        "solver_name": result.solver_name,
        "scenario": config.scenario.value,
        "input_file": str(source_path),
        "input_sha256": input_sha256,
        "number_of_hours": len(validated_input),
        "start_timestamp_utc": validated_input["timestamp"].iloc[0].isoformat(),
        "end_timestamp_utc": validated_input["timestamp"].iloc[-1].isoformat(),
        "annualization_factor": result.annualization_factor,
        "objective_eur_per_year": result.objective_eur_per_year,
        "lcoh_eur_per_kg_h2": result.lcoh_eur_per_kg_h2,
        "annual_h2_delivered_kg": result.annual_h2_delivered_kg,
        "annual_h2_produced_kg": result.annual_h2_produced_kg,
        "annual_grid_emissions_kg_co2e": result.annual_grid_emissions_kg_co2e,
        "operational_emission_intensity_kg_co2e_per_kg_h2": (
            result.operational_emission_intensity_kg_co2e_per_kg_h2
        ),
        "annual_regulatory_non_renewable_electricity_mwh": (
            result.annual_regulatory_non_renewable_electricity_mwh
        ),
        "annual_regulatory_emissions_kg_co2e": (
            result.annual_regulatory_emissions_kg_co2e
        ),
        "regulatory_emission_intensity_kg_co2e_per_kg_h2": (
            result.regulatory_emission_intensity_kg_co2e_per_kg_h2
        ),
        "regulatory_emission_intensity_g_co2e_per_mj_h2": (
            result.regulatory_emission_intensity_g_co2e_per_mj_h2
        ),
        "red_iii_ghg_savings_fraction": result.red_iii_ghg_savings_fraction,
        "red_iii_ghg_compliant": result.red_iii_ghg_compliant,
        "red_iii_maximum_product_intensity_kg_co2e_per_kg_h2": (
            result.red_iii_maximum_product_intensity_kg_co2e_per_kg_h2
        ),
        "red_iii_maximum_product_intensity_g_co2e_per_mj": (
            result.red_iii_maximum_product_intensity_g_co2e_per_mj
        ),
        "solver_runtime_seconds": result.runtime_seconds,
        "optimality_gap_fraction": result.optimality_gap_fraction,
        "max_electricity_balance_residual_mwh": (
            result.max_electricity_balance_residual_mwh
        ),
        "max_hydrogen_balance_residual_kg": (
            result.max_hydrogen_balance_residual_kg
        ),
        "annual_cost_balance_residual_eur_per_year": (
            result.annual_cost_balance_residual_eur_per_year
        ),
        "red_iii_temporal_mode": result.red_iii_temporal_mode,
        "red_iii_temporal_compliant": result.red_iii_temporal_compliant,
        "red_iii_correlation_periods": result.red_iii_correlation_periods,
        "max_red_iii_temporal_deficit_mwh": (
            result.max_red_iii_temporal_deficit_mwh
        ),
    }
    summary.update(result.capacities)
    summary.update(result.annual_costs)
    return pd.DataFrame([summary])


def _build_metadata(
    *,
    result: HydrogenOptimizationResult,
    config: ModelConfig,
    validated_input: pd.DataFrame,
    source_path: Path,
    input_sha256: str,
) -> dict[str, object]:
    parameters = {
        name: {
            "value": float(parameter.value),
            "unit": parameter.unit,
            "source": parameter.source,
            "reference_year": parameter.reference_year,
            "note": parameter.note,
        }
        for name, parameter in iter_scalar_parameters(config)
    }
    gurobi_version = ".".join(str(part) for part in gp.gurobi.version())
    return {
        "schema_version": "1.3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": config.scenario.value,
        "input": {
            "source_path": str(source_path),
            "sha256": input_sha256,
            "number_of_hours": len(validated_input),
            "start_timestamp_utc": validated_input["timestamp"].iloc[0].isoformat(),
            "end_timestamp_utc": validated_input["timestamp"].iloc[-1].isoformat(),
            "time_zone": "UTC",
            "frequency": "1h",
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "gurobi": gurobi_version,
        },
        "result": {
            "solver_status": result.solver_status,
            "solver_name": result.solver_name,
            "solver_runtime_seconds": result.runtime_seconds,
            "optimality_gap_fraction": result.optimality_gap_fraction,
            "objective_eur_per_year": result.objective_eur_per_year,
            "lcoh_eur_per_kg_h2": result.lcoh_eur_per_kg_h2,
            "emissions": {
                "operational_grid_emissions_kg_co2e_per_year": (
                    result.annual_grid_emissions_kg_co2e
                ),
                "operational_intensity_kg_co2e_per_kg_h2": (
                    result.operational_emission_intensity_kg_co2e_per_kg_h2
                ),
                "regulatory_non_renewable_electricity_mwh_per_year": (
                    result.annual_regulatory_non_renewable_electricity_mwh
                ),
                "regulatory_emissions_kg_co2e_per_year": (
                    result.annual_regulatory_emissions_kg_co2e
                ),
                "regulatory_intensity_kg_co2e_per_kg_h2": (
                    result.regulatory_emission_intensity_kg_co2e_per_kg_h2
                ),
                "regulatory_intensity_g_co2e_per_mj_h2_lhv": (
                    result.regulatory_emission_intensity_g_co2e_per_mj_h2
                ),
                "red_iii_ghg_savings_fraction": (
                    result.red_iii_ghg_savings_fraction
                ),
                "red_iii_ghg_compliant": result.red_iii_ghg_compliant,
                "maximum_intensity_kg_co2e_per_kg_h2": (
                    result.red_iii_maximum_product_intensity_kg_co2e_per_kg_h2
                ),
                "maximum_intensity_g_co2e_per_mj_h2_lhv": (
                    result.red_iii_maximum_product_intensity_g_co2e_per_mj
                ),
                "scope_note": (
                    "Die operative Bilanz bewertet jeden physischen Netzbezug "
                    "mit dem stündlichen Netzfaktor. Die regulatorische Bilanz "
                    "bewertet zeitlich zugeordneten erneuerbaren Strom im "
                    "allgemeinen Art.-4(4)-Pfad mit null. Berücksichtigt werden "
                    "nur Stromemissionen bis zur H2-Bereitstellung; Herstellung "
                    "der Anlagen und weitere Vorketten liegen außerhalb der "
                    "gewählten Systemgrenze. Bezugsbasis ist der H2-Heizwert "
                    "von 120 MJ/kg."
                ),
            },
            "balance_diagnostics": {
                "max_electricity_balance_residual_mwh": (
                    result.max_electricity_balance_residual_mwh
                ),
                "max_hydrogen_balance_residual_kg": (
                    result.max_hydrogen_balance_residual_kg
                ),
                "annual_cost_balance_residual_eur_per_year": (
                    result.annual_cost_balance_residual_eur_per_year
                ),
            },
            "red_iii_temporal_check": {
                "mode": result.red_iii_temporal_mode,
                "compliant": result.red_iii_temporal_compliant,
                "number_of_correlation_periods": (
                    result.red_iii_correlation_periods
                ),
                "max_deficit_mwh": result.max_red_iii_temporal_deficit_mwh,
                "scope_note": (
                    "Die zeitliche Strommengen-Korrelation ist in den Szenarien "
                    "red_monthly und red_hourly optimiert und ex post geprüft. "
                    "Die Niedrigpreis-Ausnahme ist mangels Day-Ahead- und ETS-Daten "
                    "nicht aktiviert. Die regulatorische THG-Intensität wird "
                    "berechnet und für RED-Szenarien gegen den Grenzwert geprüft. "
                    "Zusätzlichkeit, geografische Korrelation und eindeutige "
                    "Allokation bleiben extern zu belegende Eingangsnachweise."
                ),
            },
        },
        "model_parameters": parameters,
        "output_files": {
            "validated_input": VALIDATED_INPUT_FILENAME,
            "summary": SUMMARY_FILENAME,
            "hourly_operation": HOURLY_OPERATION_FILENAME,
            "metadata": RUN_METADATA_FILENAME,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv_atomic(data: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        data.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(data: dict[str, object], destination: Path) -> None:
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
        description=(
            "Führt den geprüften H2-Basiskern für eine stündliche CSV-Datei aus."
        ),
        epilog=(
            "Beispiel: python run_single_site_h2.py --input test_input.csv "
            "--scenario reference --output-dir results\\testlauf"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV-Datei mit den sechs Pflichtspalten aus h2_input_data.py.",
    )
    parser.add_argument(
        "--scenario",
        choices=SUPPORTED_CLI_SCENARIOS,
        default=Scenario.REFERENCE.value,
        help="Stromversorgungsszenario; Standard: reference.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Zielordner für Eingabekopie, Zusammenfassung, Stundenwerte und Metadaten.",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "gurobi", "scipy-highs"),
        default="auto",
        help=(
            "LP-Löser; auto nutzt Gurobi und wechselt bei einer größenbeschränkten "
            "Lizenz für große Modelle zu SciPy/HiGHS."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ersetzt vorhandene Ergebnisdateien mit denselben vier Namen.",
    )
    parser.add_argument(
        "--solver-output",
        action="store_true",
        help="Zeigt das vollständige Gurobi-Protokoll.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        print(
            "\nEs wurde noch kein Modelllauf gestartet. Gib --input und --output-dir an."
        )
        return 0

    args = parser.parse_args(arguments)
    try:
        artifacts = run_single_site_h2(
            args.input,
            args.output_dir,
            scenario=args.scenario,
            overwrite=args.overwrite,
            solver_output=args.solver_output,
            solver_backend=args.solver,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        H2RunError,
        HourlyInputError,
        HydrogenOptimizationError,
        NotImplementedError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    print("Modelllauf erfolgreich abgeschlossen.")
    print(f"Szenario: {args.scenario}")
    print(f"Solver: {artifacts.result.solver_name}")
    print(f"LCOH: {artifacts.result.lcoh_eur_per_kg_h2:.4f} EUR/kg H2")
    print(
        "Regulatorische THG-Intensität: "
        f"{artifacts.result.regulatory_emission_intensity_kg_co2e_per_kg_h2:.4f} "
        "kg CO2e/kg H2, "
        f"RED-III-Grenzwert eingehalten={artifacts.result.red_iii_ghg_compliant}"
    )
    print(
        "Maximale Bilanzfehler: "
        f"Strom {artifacts.result.max_electricity_balance_residual_mwh:.3e} MWh, "
        f"H2 {artifacts.result.max_hydrogen_balance_residual_kg:.3e} kg"
    )
    if artifacts.result.red_iii_temporal_compliant is not None:
        print(
            "RED-III-Zeitkorrelation: "
            f"{artifacts.result.red_iii_temporal_mode}, "
            f"konform={artifacts.result.red_iii_temporal_compliant}"
        )
    print(f"Zusammenfassung: {artifacts.summary_path}")
    print(f"Stundenwerte: {artifacts.hourly_operation_path}")
    print(f"Validierte Eingabe: {artifacts.validated_input_path}")
    print(f"Metadaten: {artifacts.metadata_path}")
    return 0


if __name__ == "__main__":
    return_code = main()
    if return_code:
        raise SystemExit(return_code)


__all__ = [
    "H2RunError",
    "RunArtifacts",
    "build_argument_parser",
    "main",
    "run_single_site_h2",
]
