r"""Reproduzierbare Einzelfaktor-Sensitivitätsanalyse des H2-Modells.

Die Fälle werden aus einer CSV-Tabelle gelesen. Jeder Lauf verändert gegenüber
dem Basisfall genau einen Parameter und schreibt die üblichen vier
Modellergebnisse in einen eigenen Ordner. Beispiel im Windows-Terminal::

    python run_h2_sensitivity.py --input input.csv --scenarios red_hourly --solver scipy-highs --base-uniform-real-wacc 0.11 --base-wacc-source "Regionalproxy" --output-dir outputs_h2\sensitivitaet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from config_h2 import (
    CONFIG_SENSITIVITY_PARAMETERS,
    DEFAULT_CONFIG,
    ModelConfig,
    Scenario,
    with_sensitivity_parameter,
    with_uniform_real_wacc,
)
from h2_input_data import HourlyInputError, validate_hourly_input
from opt_hydrogen_functions import HydrogenOptimizationError
from run_single_site_h2 import H2RunError, OUTPUT_FILENAMES, RunArtifacts, run_single_site_h2


DEFAULT_CASE_TABLE: Final[Path] = (
    Path(__file__).resolve().parent / "input_data" / "h2_sensitivity_cases.csv"
)
COMPARISON_FILENAME: Final[str] = "sensitivity_comparison.csv"
METADATA_FILENAME: Final[str] = "sensitivity_metadata.json"
CASE_INPUT_DIRECTORY: Final[str] = "case_inputs"
RUN_DIRECTORY: Final[str] = "runs"
BASELINE_PARAMETER: Final[str] = "baseline"
INPUT_SENSITIVITY_PARAMETERS: Final[tuple[str, ...]] = (
    "electricity_price_eur_per_mwh",
)
SUPPORTED_PARAMETERS: Final[tuple[str, ...]] = (
    BASELINE_PARAMETER,
    *INPUT_SENSITIVITY_PARAMETERS,
    *CONFIG_SENSITIVITY_PARAMETERS,
)
PARAMETER_UNITS: Final[dict[str, str]] = {
    BASELINE_PARAMETER: "-",
    "electricity_price_eur_per_mwh": "EUR_2023/MWh",
    "electrolyzer_capex_eur_per_kw": "EUR_2023/kW",
    "electrolyzer_specific_electricity_kwh_per_kg_h2": "kWh/kg_H2",
    "pv_capex_eur_per_kw": "EUR_2023/kW",
    "wind_capex_eur_per_kw": "EUR_2023/kW",
    "h2_storage_capex_eur_per_kg_h2": "EUR_2023/kg_H2",
    "uniform_real_wacc_fraction": "fraction",
}
CASE_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
CORE_SENSITIVITY_SCENARIOS: Final[tuple[Scenario, ...]] = (
    Scenario.REFERENCE,
    Scenario.RED_MONTHLY,
    Scenario.RED_HOURLY,
)
DELTA_METRICS: Final[tuple[str, ...]] = (
    "lcoh_eur_per_kg_h2",
    "pv_capacity_mw",
    "wind_capacity_mw",
    "electrolyzer_capacity_mw",
    "compressor_capacity_mw",
    "h2_storage_capacity_kg",
    "annual_grid_import_mwh",
    "operational_emission_intensity_kg_co2e_per_kg_h2",
    "regulatory_emission_intensity_kg_co2e_per_kg_h2",
)


@dataclass(frozen=True, slots=True)
class SensitivityCase:
    """Eine Tabellenzeile und damit genau eine Änderung zum Basisfall."""

    case_id: str
    parameter: str
    level: str
    label: str
    value: float | None
    unit: str
    source: str
    note: str


@dataclass(frozen=True, slots=True)
class SensitivityArtifacts:
    """Ergebnisdateien eines vollständigen Sensitivitätsaufrufs."""

    output_directory: Path
    comparison_path: Path
    metadata_path: Path
    runs: dict[tuple[str, str], RunArtifacts]


def load_sensitivity_cases(path: str | Path) -> list[SensitivityCase]:
    """Lade und validiere eine Sensitivitätstabelle in stabiler Zeilenfolge."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Sensitivitätstabelle nicht gefunden: {source_path}")
    try:
        table = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Sensitivitätstabelle konnte nicht gelesen werden: {source_path}"
        ) from exc

    required_columns = (
        "case_id",
        "parameter",
        "level",
        "label",
        "value",
        "unit",
        "source",
        "note",
    )
    missing = [column for column in required_columns if column not in table.columns]
    if missing:
        raise ValueError(
            "Sensitivitätstabelle enthält nicht alle Pflichtspalten: "
            + ", ".join(missing)
        )
    if table.empty:
        raise ValueError("Sensitivitätstabelle darf nicht leer sein.")

    cases: list[SensitivityCase] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(table.itertuples(index=False), start=2):
        values = {column: str(getattr(row, column)).strip() for column in required_columns}
        case_id = values["case_id"]
        parameter = values["parameter"]
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError(
                f"Zeile {row_number}: case_id '{case_id}' darf nur Kleinbuchstaben, "
                "Ziffern, '-' und '_' enthalten."
            )
        if case_id in seen_ids:
            raise ValueError(f"Zeile {row_number}: case_id '{case_id}' ist doppelt.")
        seen_ids.add(case_id)
        if parameter not in SUPPORTED_PARAMETERS:
            allowed = ", ".join(SUPPORTED_PARAMETERS)
            raise ValueError(
                f"Zeile {row_number}: unbekannter Parameter '{parameter}'. "
                f"Erlaubt: {allowed}."
            )
        if not values["level"] or not values["label"] or not values["source"]:
            raise ValueError(
                f"Zeile {row_number}: level, label und source dürfen nicht leer sein."
            )
        expected_unit = PARAMETER_UNITS[parameter]
        if values["unit"] != expected_unit:
            raise ValueError(
                f"Zeile {row_number}: Einheit für '{parameter}' muss "
                f"'{expected_unit}' sein."
            )

        if parameter == BASELINE_PARAMETER:
            if values["value"]:
                raise ValueError(
                    f"Zeile {row_number}: Der Basisfall darf keinen Einzelwert ändern."
                )
            numeric_value = None
        else:
            try:
                numeric_value = float(values["value"])
            except ValueError as exc:
                raise ValueError(
                    f"Zeile {row_number}: value muss eine Zahl mit Dezimalpunkt sein."
                ) from exc
            if not np.isfinite(numeric_value) or numeric_value <= 0.0:
                raise ValueError(
                    f"Zeile {row_number}: Sensitivitätswerte müssen positiv und endlich sein."
                )
            if parameter == "uniform_real_wacc_fraction" and numeric_value >= 1.0:
                raise ValueError(
                    f"Zeile {row_number}: WACC muss als Dezimalzahl unter 1 angegeben werden."
                )

        cases.append(
            SensitivityCase(
                case_id=case_id,
                parameter=parameter,
                level=values["level"],
                label=values["label"],
                value=numeric_value,
                unit=values["unit"],
                source=values["source"],
                note=values["note"],
            )
        )

    baseline_cases = [case for case in cases if case.parameter == BASELINE_PARAMETER]
    if len(baseline_cases) != 1:
        raise ValueError("Sensitivitätstabelle benötigt genau einen Basisfall.")
    if cases[0].parameter != BASELINE_PARAMETER:
        raise ValueError("Der Basisfall muss die erste Tabellenzeile sein.")
    return cases


def run_h2_sensitivity(
    input_path: str | Path,
    cases_path: str | Path,
    output_directory: str | Path,
    *,
    scenarios: Sequence[Scenario | str] = (Scenario.RED_HOURLY,),
    config: ModelConfig = DEFAULT_CONFIG,
    overwrite: bool = False,
    solver_output: bool = False,
    solver_backend: str = "auto",
) -> SensitivityArtifacts:
    """Rechne alle Tabellenfälle mit je genau einer Parameteränderung."""

    for name, value in (("overwrite", overwrite), ("solver_output", solver_output)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} muss ein boolescher Wert sein.")
    selected_scenarios = _coerce_scenarios(scenarios)
    cases = load_sensitivity_cases(cases_path)

    source_path = Path(input_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {source_path}")
    try:
        original_input = validate_hourly_input(pd.read_csv(source_path))
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise H2RunError(f"Eingabedatei konnte nicht gelesen werden: {source_path}") from exc

    config.validate()
    output_root = Path(output_directory).expanduser().resolve()
    _preflight_outputs(
        output_root=output_root,
        cases=cases,
        scenarios=selected_scenarios,
        overwrite=overwrite,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / CASE_INPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    base_values = _base_parameter_values(config, original_input)
    runs: dict[tuple[str, str], RunArtifacts] = {}
    summary_rows: list[pd.DataFrame] = []
    for case_order, case in enumerate(cases):
        case_config, case_input = _apply_case(case, config, original_input)
        case_input_path = output_root / CASE_INPUT_DIRECTORY / f"{case.case_id}.csv"
        _write_csv_atomic(case_input, case_input_path)
        for scenario_order, selected_scenario in enumerate(selected_scenarios):
            artifacts = run_single_site_h2(
                case_input_path,
                output_root / RUN_DIRECTORY / case.case_id / selected_scenario.value,
                scenario=selected_scenario,
                config=case_config,
                overwrite=overwrite,
                solver_output=solver_output,
                solver_backend=solver_backend,
            )
            summary = pd.read_csv(artifacts.summary_path)
            if len(summary) != 1:
                raise H2RunError(
                    f"{artifacts.summary_path} enthält nicht genau eine Ergebniszeile."
                )
            summary.insert(0, "case_order", case_order)
            summary.insert(1, "case_id", case.case_id)
            summary.insert(2, "case_label", case.label)
            summary.insert(3, "sensitivity_parameter", case.parameter)
            summary.insert(4, "sensitivity_level", case.level)
            summary.insert(5, "sensitivity_value", case.value)
            summary.insert(6, "sensitivity_unit", case.unit)
            summary.insert(7, "sensitivity_source", case.source)
            summary.insert(8, "sensitivity_note", case.note)
            baseline_value = base_values.get(case.parameter)
            summary.insert(9, "baseline_parameter_value", baseline_value)
            if case.value is None or baseline_value is None:
                parameter_change = np.nan
            else:
                parameter_change = 100.0 * (case.value - baseline_value) / baseline_value
            summary.insert(10, "parameter_change_percent", parameter_change)
            summary.insert(
                11,
                "result_directory",
                str(
                    Path(RUN_DIRECTORY)
                    / case.case_id
                    / selected_scenario.value
                ),
            )
            summary["annual_grid_import_mwh"] = (
                float(artifacts.result.hourly_operation["grid_import_mwh"].sum())
                * artifacts.result.annualization_factor
            )
            summary["scenario_order"] = scenario_order
            summary_rows.append(summary)
            runs[(case.case_id, selected_scenario.value)] = artifacts

    comparison = _add_baseline_deltas(pd.concat(summary_rows, ignore_index=True))
    comparison_path = output_root / COMPARISON_FILENAME
    metadata_path = output_root / METADATA_FILENAME
    _write_csv_atomic(comparison, comparison_path)
    _write_json_atomic(
        {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Einzelfaktor-Sensitivitätsanalyse; pro Fall genau eine Änderung zum Basisfall",
            "source_input": {
                "path": str(source_path),
                "sha256": _sha256(source_path),
            },
            "case_table": {
                "path": str(Path(cases_path).expanduser().resolve()),
                "sha256": _sha256(Path(cases_path).expanduser().resolve()),
            },
            "scenarios": [scenario.value for scenario in selected_scenarios],
            "requested_solver_backend": solver_backend,
            "number_of_cases": len(cases),
            "number_of_optimization_runs": len(runs),
            "comparison_file": COMPARISON_FILENAME,
            "case_input_directory": CASE_INPUT_DIRECTORY,
            "run_directory": RUN_DIRECTORY,
            "baseline_parameter_values": base_values,
            "cases": [
                {
                    "case_id": case.case_id,
                    "parameter": case.parameter,
                    "level": case.level,
                    "label": case.label,
                    "value": case.value,
                    "unit": case.unit,
                    "source": case.source,
                    "note": case.note,
                }
                for case in cases
            ],
        },
        metadata_path,
    )
    return SensitivityArtifacts(output_root, comparison_path, metadata_path, runs)


def _coerce_scenarios(values: Sequence[Scenario | str]) -> tuple[Scenario, ...]:
    if isinstance(values, (str, bytes)):
        values = (values,)
    selected: list[Scenario] = []
    for value in values:
        try:
            scenario = value if isinstance(value, Scenario) else Scenario(value)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(scenario.value for scenario in Scenario)
            raise ValueError(f"Unbekanntes Szenario '{value}'. Erlaubt: {allowed}.") from exc
        if scenario not in selected:
            selected.append(scenario)
    if not selected:
        raise ValueError("Mindestens ein Szenario muss ausgewählt werden.")
    return tuple(selected)


def _apply_case(
    case: SensitivityCase,
    base_config: ModelConfig,
    base_input: pd.DataFrame,
) -> tuple[ModelConfig, pd.DataFrame]:
    case_input = base_input.copy(deep=True)
    if case.parameter == BASELINE_PARAMETER:
        return base_config, case_input
    assert case.value is not None
    if case.parameter == "electricity_price_eur_per_mwh":
        case_input["electricity_price"] = case.value
        return base_config, validate_hourly_input(case_input)
    return (
        with_sensitivity_parameter(
            base_config,
            case.parameter,
            case.value,
            source=case.source,
            note=case.note,
        ),
        case_input,
    )


def _base_parameter_values(
    config: ModelConfig,
    hourly_input: pd.DataFrame,
) -> dict[str, float]:
    technology = config.technologies
    wacc_values = np.asarray(
        [
            technology.pv.real_wacc_fraction.value,
            technology.wind_onshore.real_wacc_fraction.value,
            technology.electrolyzer.real_wacc_fraction.value,
            technology.compressor.real_wacc_fraction.value,
            technology.h2_storage.real_wacc_fraction.value,
        ],
        dtype=float,
    )
    if not np.allclose(wacc_values, wacc_values[0], rtol=0.0, atol=1e-12):
        uniform_wacc = float(np.mean(wacc_values))
    else:
        uniform_wacc = float(wacc_values[0])
    return {
        "electricity_price_eur_per_mwh": float(hourly_input["electricity_price"].mean()),
        "electrolyzer_capex_eur_per_kw": technology.electrolyzer.capex_eur_per_kw.value,
        "electrolyzer_specific_electricity_kwh_per_kg_h2": (
            technology.electrolyzer.specific_electricity_kwh_per_kg_h2.value
        ),
        "pv_capex_eur_per_kw": technology.pv.capex_eur_per_kw.value,
        "wind_capex_eur_per_kw": technology.wind_onshore.capex_eur_per_kw.value,
        "h2_storage_capex_eur_per_kg_h2": technology.h2_storage.capex_eur_per_kg_h2.value,
        "uniform_real_wacc_fraction": uniform_wacc,
    }


def _add_baseline_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    baseline = comparison.loc[
        comparison["sensitivity_parameter"] == BASELINE_PARAMETER
    ]
    expected_scenarios = set(comparison["scenario"])
    if set(baseline["scenario"]) != expected_scenarios or baseline["scenario"].duplicated().any():
        raise H2RunError("Für jedes Szenario wird genau ein Basislauf benötigt.")
    baseline_by_scenario = baseline.set_index("scenario")
    for metric in DELTA_METRICS:
        if metric not in comparison.columns:
            continue
        base_values = comparison["scenario"].map(baseline_by_scenario[metric]).astype(float)
        comparison[f"{metric}_delta_vs_baseline"] = (
            pd.to_numeric(comparison[metric]) - base_values
        )
    lcoh_base = comparison["scenario"].map(
        baseline_by_scenario["lcoh_eur_per_kg_h2"]
    ).astype(float)
    comparison["lcoh_change_percent_vs_baseline"] = (
        100.0
        * comparison["lcoh_eur_per_kg_h2_delta_vs_baseline"]
        / lcoh_base
    )
    numeric_columns = comparison.select_dtypes(include=["number"]).columns
    comparison.loc[:, numeric_columns] = comparison[numeric_columns].mask(
        comparison[numeric_columns].abs() < 1e-10,
        0.0,
    )
    return comparison.sort_values(
        ["case_order", "scenario_order"], kind="stable"
    ).reset_index(drop=True)


def _preflight_outputs(
    *,
    output_root: Path,
    cases: Sequence[SensitivityCase],
    scenarios: Sequence[Scenario],
    overwrite: bool,
) -> None:
    if output_root.exists() and not output_root.is_dir():
        raise H2RunError(f"Ausgabepfad ist kein Verzeichnis: {output_root}")
    expected = [output_root / COMPARISON_FILENAME, output_root / METADATA_FILENAME]
    expected.extend(
        output_root / CASE_INPUT_DIRECTORY / f"{case.case_id}.csv" for case in cases
    )
    expected.extend(
        output_root / RUN_DIRECTORY / case.case_id / scenario.value / filename
        for case in cases
        for scenario in scenarios
        for filename in OUTPUT_FILENAMES
    )
    existing = [path for path in expected if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path.relative_to(output_root)) for path in existing[:5])
        if len(existing) > 5:
            preview += f", … und {len(existing) - 5} weitere"
        raise FileExistsError(
            f"Sensitivitätsergebnisse existieren bereits ({preview}). "
            "Nutze --overwrite zum Neuberechnen."
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
        description=(
            "Rechnet eine tabellengesteuerte Einzelfaktor-Sensitivitätsanalyse "
            "für das H2-Modell."
        ),
        epilog=(
            "Beispiel: python run_h2_sensitivity.py --input input.csv "
            "--scenarios red_hourly --solver scipy-highs "
            "--base-uniform-real-wacc 0.11 --base-wacc-source Regionalproxy "
            "--output-dir outputs_h2\\sensitivitaet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="Validierte Stunden-CSV.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASE_TABLE,
        help=f"Falltabelle; Standard: {DEFAULT_CASE_TABLE}",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Ergebnisordner.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("all", *(scenario.value for scenario in Scenario)),
        default=[Scenario.RED_HOURLY.value],
        help="Zu rechnende Szenarien; 'all' bedeutet S0, S1 und S2.",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "gurobi", "scipy-highs"),
        default="auto",
        help="Solver für alle Läufe; Standard: auto.",
    )
    parser.add_argument(
        "--base-uniform-real-wacc",
        type=float,
        help="Optionaler einheitlicher WACC des Basisfalls, z. B. 0.11.",
    )
    parser.add_argument(
        "--base-wacc-source",
        help="Quellenbeschreibung für --base-uniform-real-wacc.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Berechnet Ergebnisse neu.")
    parser.add_argument(
        "--solver-output", action="store_true", help="Zeigt das Solverprotokoll."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        print("\nEs wurde noch keine Sensitivitätsanalyse gestartet.")
        return 0
    args = parser.parse_args(arguments)
    if "all" in args.scenarios:
        if len(args.scenarios) != 1:
            parser.error("'all' darf nicht mit einzelnen Szenarien kombiniert werden.")
        scenarios: Sequence[Scenario] = CORE_SENSITIVITY_SCENARIOS
    else:
        scenarios = tuple(Scenario(value) for value in args.scenarios)

    run_config = DEFAULT_CONFIG
    if args.base_uniform_real_wacc is not None:
        if not args.base_wacc_source:
            parser.error(
                "--base-wacc-source ist zusammen mit --base-uniform-real-wacc erforderlich."
            )
        run_config = with_uniform_real_wacc(
            run_config,
            args.base_uniform_real_wacc,
            source=args.base_wacc_source,
            note="Standortbezogener Basis-WACC der Sensitivitätsanalyse.",
        )
    try:
        artifacts = run_h2_sensitivity(
            args.input,
            args.cases,
            args.output_dir,
            scenarios=scenarios,
            config=run_config,
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

    comparison = pd.read_csv(artifacts.comparison_path)
    columns = [
        "case_id",
        "scenario",
        "lcoh_eur_per_kg_h2",
        "lcoh_change_percent_vs_baseline",
        "wind_capacity_mw",
        "h2_storage_capacity_kg",
    ]
    print("Sensitivitätsanalyse erfolgreich abgeschlossen.")
    print(comparison[columns].to_string(index=False))
    print(f"Vergleichstabelle: {artifacts.comparison_path}")
    print(f"Metadaten: {artifacts.metadata_path}")
    return 0


if __name__ == "__main__":
    return_code = main()
    if return_code:
        raise SystemExit(return_code)


__all__ = [
    "COMPARISON_FILENAME",
    "DEFAULT_CASE_TABLE",
    "METADATA_FILENAME",
    "SensitivityArtifacts",
    "SensitivityCase",
    "build_argument_parser",
    "load_sensitivity_cases",
    "main",
    "run_h2_sensitivity",
]
