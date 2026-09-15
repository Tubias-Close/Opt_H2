r"""Einheitlicher Szenarienlauf für das standortunabhängige H2-Modell.

Ein Aufruf rechnet S0, S1 und S2 mit derselben validierten Eingabedatei::

    python run_h2_scenarios.py --input input.csv --output-dir outputs_h2\vergleich

Optional wird mit ``--include-off-grid`` zusätzlich S3 gerechnet. Jeder Lauf
erhält einen eigenen Ergebnisordner. Anschließend werden die einzeiligen
Zusammenfassungen in einer gemeinsamen Vergleichstabelle zusammengeführt.
"""

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

from config_h2 import DEFAULT_CONFIG, ModelConfig, Scenario, with_uniform_real_wacc
from h2_input_data import HourlyInputError
from opt_hydrogen_functions import HydrogenOptimizationError
from run_single_site_h2 import (
    H2RunError,
    OUTPUT_FILENAMES,
    RunArtifacts,
    run_single_site_h2,
)


COMPARISON_FILENAME: Final[str] = "scenario_comparison.csv"
BATCH_METADATA_FILENAME: Final[str] = "scenario_comparison_metadata.json"


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Feste Bezeichnung und Ausgabezuordnung eines Kernszenarios."""

    scenario_id: str
    scenario: Scenario
    label: str
    directory_name: str


CORE_SCENARIOS: Final[tuple[ScenarioSpec, ...]] = (
    ScenarioSpec("S0", Scenario.REFERENCE, "Referenz", "S0_reference"),
    ScenarioSpec("S1", Scenario.RED_MONTHLY, "RED III monatlich", "S1_red_monthly"),
    ScenarioSpec("S2", Scenario.RED_HOURLY, "RED III stündlich", "S2_red_hourly"),
)
OFF_GRID_SCENARIO: Final[ScenarioSpec] = ScenarioSpec(
    "S3", Scenario.OFF_GRID, "Off-grid", "S3_off_grid"
)


@dataclass(frozen=True, slots=True)
class ScenarioBatchArtifacts:
    """Pfade und Einzelergebnisse eines vollständigen Szenarienvergleichs."""

    output_directory: Path
    comparison_path: Path
    metadata_path: Path
    scenario_runs: dict[str, RunArtifacts]


def run_h2_scenarios(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    include_off_grid: bool = False,
    config: ModelConfig = DEFAULT_CONFIG,
    overwrite: bool = False,
    solver_output: bool = False,
    solver_backend: str = "auto",
) -> ScenarioBatchArtifacts:
    """Rechne alle ausgewählten Szenarien mit exakt derselben Eingabedatei."""

    for name, value in (
        ("include_off_grid", include_off_grid),
        ("overwrite", overwrite),
        ("solver_output", solver_output),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} muss ein boolescher Wert sein.")

    source_path = Path(input_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(
            f"Die Eingabedatei wurde nicht gefunden: {source_path}"
        )
    if not source_path.is_file():
        raise H2RunError(f"Der Eingabepfad ist keine Datei: {source_path}")

    output_root = Path(output_directory).expanduser().resolve()
    scenario_specs = list(CORE_SCENARIOS)
    if include_off_grid:
        scenario_specs.append(OFF_GRID_SCENARIO)
    _preflight_outputs(
        source_path=source_path,
        output_root=output_root,
        scenario_specs=scenario_specs,
        overwrite=overwrite,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    input_sha256 = _sha256(source_path)
    scenario_runs: dict[str, RunArtifacts] = {}
    summary_rows: list[pd.DataFrame] = []
    for spec in scenario_specs:
        artifacts = run_single_site_h2(
            source_path,
            output_root / spec.directory_name,
            scenario=spec.scenario,
            config=config,
            overwrite=overwrite,
            solver_output=solver_output,
            solver_backend=solver_backend,
        )
        summary = pd.read_csv(artifacts.summary_path)
        if len(summary) != 1:
            raise H2RunError(
                f"{artifacts.summary_path} enthält nicht genau eine Ergebniszeile."
            )
        if str(summary.loc[0, "input_sha256"]) != input_sha256:
            raise H2RunError(
                f"Der Eingabe-Hash des Szenarios {spec.scenario_id} stimmt nicht überein."
            )
        summary.insert(0, "scenario_id", spec.scenario_id)
        summary.insert(1, "scenario_label", spec.label)
        summary.insert(2, "result_directory", spec.directory_name)
        summary_rows.append(summary)
        scenario_runs[spec.scenario_id] = artifacts

    comparison = _build_comparison(summary_rows)
    comparison_path = output_root / COMPARISON_FILENAME
    metadata_path = output_root / BATCH_METADATA_FILENAME
    _write_csv_atomic(comparison, comparison_path)
    _write_json_atomic(
        _build_batch_metadata(
            source_path=source_path,
            input_sha256=input_sha256,
            solver_backend=solver_backend,
            scenario_specs=scenario_specs,
            scenario_runs=scenario_runs,
        ),
        metadata_path,
    )
    return ScenarioBatchArtifacts(
        output_directory=output_root,
        comparison_path=comparison_path,
        metadata_path=metadata_path,
        scenario_runs=scenario_runs,
    )


def _preflight_outputs(
    *,
    source_path: Path,
    output_root: Path,
    scenario_specs: Sequence[ScenarioSpec],
    overwrite: bool,
) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise H2RunError(f"Der Ausgabepfad ist kein Verzeichnis: {output_root}")
    expected_paths = [
        output_root / COMPARISON_FILENAME,
        output_root / BATCH_METADATA_FILENAME,
    ]
    expected_paths.extend(
        output_root / spec.directory_name / filename
        for spec in scenario_specs
        for filename in OUTPUT_FILENAMES
    )
    if source_path in expected_paths:
        raise H2RunError(
            "Die Eingabedatei darf nicht denselben Pfad wie eine Ergebnisdatei haben."
        )
    existing = [path for path in expected_paths if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path.relative_to(output_root)) for path in existing[:5])
        if len(existing) > 5:
            preview += f", … und {len(existing) - 5} weitere"
        raise FileExistsError(
            f"Ergebnisdateien existieren bereits ({preview}). "
            "Nutze --overwrite zum vollständigen Neuberechnen."
        )


def _build_comparison(summary_rows: Sequence[pd.DataFrame]) -> pd.DataFrame:
    comparison = pd.concat(summary_rows, ignore_index=True)
    for boolean_column in (
        "red_iii_temporal_compliant",
        "red_iii_ghg_compliant",
    ):
        if boolean_column in comparison.columns:
            comparison[boolean_column] = comparison[boolean_column].apply(
                _coerce_optional_bool
            ).astype("boolean")
    reference = comparison.loc[comparison["scenario_id"] == "S0"]
    if len(reference) != 1:
        raise H2RunError("Der Szenarienvergleich benötigt genau einen Referenzfall S0.")
    reference_row = reference.iloc[0]
    delta_metrics = (
        "lcoh_eur_per_kg_h2",
        "pv_capacity_mw",
        "wind_capacity_mw",
        "electrolyzer_capacity_mw",
        "compressor_capacity_mw",
        "h2_storage_capacity_kg",
        "operational_emission_intensity_kg_co2e_per_kg_h2",
        "regulatory_emission_intensity_kg_co2e_per_kg_h2",
    )
    for metric in delta_metrics:
        if metric in comparison.columns:
            delta_column = f"{metric}_delta_vs_s0"
            comparison[delta_column] = (
                pd.to_numeric(comparison[metric]) - float(reference_row[metric])
            )
            comparison.loc[
                comparison[delta_column].abs() < 1e-9, delta_column
            ] = 0.0
    reference_lcoh = float(reference_row["lcoh_eur_per_kg_h2"])
    comparison["lcoh_change_percent_vs_s0"] = (
        100.0
        * comparison["lcoh_eur_per_kg_h2_delta_vs_s0"]
        / reference_lcoh
    )
    comparison.loc[
        comparison["lcoh_change_percent_vs_s0"].abs() < 1e-9,
        "lcoh_change_percent_vs_s0",
    ] = 0.0
    return comparison


def _coerce_optional_bool(value: object) -> object:
    """Normalisiere CSV-Boolesche Werte, ohne leere S0-Felder zu erfinden."""

    if pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return True
    if normalized in {"false", "0", "0.0"}:
        return False
    raise H2RunError(f"Ungültiger boolescher Ergebniswert: {value!r}")


def _build_batch_metadata(
    *,
    source_path: Path,
    input_sha256: str,
    solver_backend: str,
    scenario_specs: Sequence[ScenarioSpec],
    scenario_runs: dict[str, RunArtifacts],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "source_path": str(source_path),
            "sha256": input_sha256,
        },
        "requested_solver_backend": solver_backend,
        "scenarios": [
            {
                "scenario_id": spec.scenario_id,
                "scenario": spec.scenario.value,
                "label": spec.label,
                "result_directory": spec.directory_name,
                "solver_used": scenario_runs[spec.scenario_id].result.solver_name,
                "summary": str(
                    scenario_runs[spec.scenario_id].summary_path.relative_to(
                        scenario_runs[spec.scenario_id].output_directory.parent
                    )
                ),
            }
            for spec in scenario_specs
        ],
        "comparison_file": COMPARISON_FILENAME,
        "reproducibility_note": (
            "Alle Szenarien wurden im selben Aufruf, mit derselben Eingabedatei "
            "und demselben ModelConfig-Ausgangsobjekt gerechnet. Nur das "
            "Szenariofeld wurde je Lauf durch run_single_site_h2 gesetzt."
        ),
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
            "Rechnet S0 Referenz, S1 monatlich und S2 stündlich mit derselben "
            "H2-Eingabedatei und erstellt eine Vergleichstabelle."
        ),
        epilog=(
            "Beispiel: python run_h2_scenarios.py --input input.csv "
            "--solver scipy-highs --output-dir outputs_h2\\vergleich"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Gemeinsame validierbare Stunden-CSV für alle Szenarien.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Oberordner für S0, S1, S2 und die Vergleichstabelle.",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "gurobi", "scipy-highs"),
        default="auto",
        help="Solver für alle Szenarien; Standard: auto.",
    )
    parser.add_argument(
        "--uniform-real-wacc",
        type=float,
        help=(
            "Optionaler einheitlicher realer WACC als Dezimalzahl, zum Beispiel "
            "0.11 für den regionalen Namibia-Proxy der alten Modelllogik."
        ),
    )
    parser.add_argument(
        "--wacc-source",
        help="Quellenbeschreibung für --uniform-real-wacc.",
    )
    parser.add_argument(
        "--include-off-grid",
        action="store_true",
        help="Rechnet zusätzlich S3 ohne Netzstrombezug.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Berechnet vorhandene Szenarioergebnisse vollständig neu.",
    )
    parser.add_argument(
        "--solver-output",
        action="store_true",
        help="Zeigt das ausführliche Solverprotokoll für alle Läufe.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        print(
            "\nEs wurde noch kein Szenarienlauf gestartet. "
            "Gib --input und --output-dir an."
        )
        return 0
    args = parser.parse_args(arguments)
    run_config = DEFAULT_CONFIG
    if args.uniform_real_wacc is not None:
        if not args.wacc_source:
            parser.error("--wacc-source ist zusammen mit --uniform-real-wacc erforderlich.")
        run_config = with_uniform_real_wacc(
            DEFAULT_CONFIG,
            args.uniform_real_wacc,
            source=args.wacc_source,
            note=(
                "Einheitlicher Standort-WACC analog zur Länder-/Regionalzuordnung "
                "des ursprünglichen Ammoniakmodells."
            ),
        )
    try:
        artifacts = run_h2_scenarios(
            args.input,
            args.output_dir,
            include_off_grid=args.include_off_grid,
            overwrite=args.overwrite,
            solver_output=args.solver_output,
            solver_backend=args.solver,
            config=run_config,
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

    comparison = pd.read_csv(artifacts.comparison_path)
    print("Szenarienlauf erfolgreich abgeschlossen.")
    print(
        comparison[
            [
                "scenario_id",
                "scenario",
                "lcoh_eur_per_kg_h2",
                "lcoh_change_percent_vs_s0",
            ]
        ].to_string(index=False)
    )
    print(f"Vergleichstabelle: {artifacts.comparison_path}")
    print(f"Metadaten: {artifacts.metadata_path}")
    return 0


if __name__ == "__main__":
    return_code = main()
    if return_code:
        raise SystemExit(return_code)


__all__ = [
    "BATCH_METADATA_FILENAME",
    "COMPARISON_FILENAME",
    "CORE_SCENARIOS",
    "OFF_GRID_SCENARIO",
    "ScenarioBatchArtifacts",
    "ScenarioSpec",
    "build_argument_parser",
    "main",
    "run_h2_scenarios",
]
