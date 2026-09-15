"""Erzeuge reproduzierbare Abbildungen aus der Sensitivitätsvergleichstabelle."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_h2 import Scenario


LCOH_FIGURE: Final[str] = "01_lcoh_sensitivity.png"
DESIGN_FIGURE: Final[str] = "02_design_sensitivity.png"
WIND_FIGURE: Final[str] = "03_wind_break_even.png"
MANIFEST_FILENAME: Final[str] = "sensitivity_figure_manifest.json"
SCENARIO_LABELS: Final[dict[Scenario, str]] = {
    Scenario.REFERENCE: "S0 – Referenz",
    Scenario.RED_MONTHLY: "S1 – RED III monatlich",
    Scenario.RED_HOURLY: "S2 – RED III stündlich",
    Scenario.OFF_GRID: "S3 – Off-grid",
}


@dataclass(frozen=True, slots=True)
class SensitivityFigureArtifacts:
    output_directory: Path
    figure_paths: tuple[Path, ...]
    manifest_path: Path


def plot_h2_sensitivity_results(
    comparison_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    scenario: Scenario | str = Scenario.RED_HOURLY,
    dpi: int = 180,
    overwrite: bool = False,
) -> SensitivityFigureArtifacts:
    """Visualisiere LCOH- und Auslegungsänderungen eines Szenarios."""

    selected_scenario = scenario if isinstance(scenario, Scenario) else Scenario(scenario)
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 72:
        raise ValueError("dpi muss eine ganze Zahl von mindestens 72 sein.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite muss ein boolescher Wert sein.")

    source_path = Path(comparison_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Sensitivitätsvergleich nicht gefunden: {source_path}")
    try:
        comparison = pd.read_csv(source_path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Sensitivitätsvergleich konnte nicht gelesen werden: {source_path}") from exc
    required = {
        "case_order",
        "case_id",
        "case_label",
        "scenario",
        "sensitivity_parameter",
        "sensitivity_value",
        "baseline_parameter_value",
        "lcoh_eur_per_kg_h2",
        "lcoh_change_percent_vs_baseline",
        "pv_capacity_mw",
        "wind_capacity_mw",
        "electrolyzer_capacity_mw",
        "h2_storage_capacity_kg",
    }
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise ValueError("Sensitivitätsvergleich enthält nicht alle Pflichtspalten: " + ", ".join(missing))

    selected = comparison.loc[
        comparison["scenario"] == selected_scenario.value
    ].sort_values("case_order", kind="stable")
    if selected.empty:
        raise ValueError(
            f"Sensitivitätsvergleich enthält kein Szenario '{selected_scenario.value}'."
        )
    baseline = selected.loc[selected["sensitivity_parameter"] == "baseline"]
    if len(baseline) != 1:
        raise ValueError("Das ausgewählte Szenario benötigt genau einen Basisfall.")
    variants = selected.loc[selected["sensitivity_parameter"] != "baseline"].copy()
    if variants.empty:
        raise ValueError("Für die Abbildung wird mindestens ein Sensitivitätsfall benötigt.")

    output_root = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else source_path.parent / "figures"
    )
    has_wind_sweep = (
        variants["sensitivity_parameter"] == "wind_capex_eur_per_kw"
    ).sum() >= 2
    figure_names = [LCOH_FIGURE, DESIGN_FIGURE]
    if has_wind_sweep:
        figure_names.append(WIND_FIGURE)
    expected = [output_root / name for name in (*figure_names, MANIFEST_FILENAME)]
    existing = [path for path in expected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Sensitivitätsabbildungen existieren bereits: "
            + ", ".join(path.name for path in existing)
            + ". Nutze --overwrite zum Ersetzen."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.30,
            "font.size": 10,
            "figure.facecolor": "white",
        }
    )
    paths: list[Path] = []
    paths.append(_plot_lcoh(variants, output_root / LCOH_FIGURE, selected_scenario, dpi))
    paths.append(
        _plot_design(
            variants,
            baseline.iloc[0],
            output_root / DESIGN_FIGURE,
            selected_scenario,
            dpi,
        )
    )
    if has_wind_sweep:
        paths.append(
            _plot_wind_break_even(
                selected,
                output_root / WIND_FIGURE,
                selected_scenario,
                dpi,
            )
        )

    manifest_path = output_root / MANIFEST_FILENAME
    _write_json_atomic(
        {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {"path": str(source_path), "sha256": _sha256(source_path)},
            "scenario": selected_scenario.value,
            "number_of_variants": len(variants),
            "dpi": dpi,
            "figures": [path.name for path in paths],
            "interpretation_note": (
                "Negative LCOH-Änderungen bedeuten geringere Kosten als im Basisfall. "
                "Die Wind-CAPEX-Punkte 25 und 50 Prozent sind explorative "
                "Break-even-Fälle und keine Preisprognosen."
            ),
        },
        manifest_path,
    )
    return SensitivityFigureArtifacts(output_root, tuple(paths), manifest_path)


def _plot_lcoh(
    variants: pd.DataFrame,
    destination: Path,
    scenario: Scenario,
    dpi: int,
) -> Path:
    plot_data = variants.iloc[::-1]
    values = pd.to_numeric(plot_data["lcoh_change_percent_vs_baseline"]).to_numpy()
    colors = np.where(values < 0.0, "#16a34a", "#dc2626")
    figure, axis = plt.subplots(figsize=(11.5, max(6.5, 0.48 * len(plot_data))))
    bars = axis.barh(plot_data["case_label"], values, color=colors)
    axis.axvline(0.0, color="#334155", linewidth=1.0)
    axis.set_xlabel("Änderung des LCOH gegenüber dem Basisfall [%]")
    axis.set_title(f"LCOH-Sensitivität – {SCENARIO_LABELS[scenario]}", fontweight="bold")
    axis.grid(axis="y", visible=False)
    span = max(float(np.max(np.abs(values))), 1.0)
    axis.set_xlim(min(-1.15 * span, float(values.min()) - 1.0), max(1.15 * span, float(values.max()) + 1.0))
    for bar, value in zip(bars, values, strict=True):
        offset = 0.015 * span
        axis.text(
            value + (offset if value >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}%",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def _plot_design(
    variants: pd.DataFrame,
    baseline: pd.Series,
    destination: Path,
    scenario: Scenario,
    dpi: int,
) -> Path:
    plot_data = variants.iloc[::-1]
    labels = plot_data["case_label"]
    metric_specs = (
        ("pv_capacity_mw", "PV-Kapazität", "%"),
        ("wind_capacity_mw", "Windkapazität", "MW"),
        ("electrolyzer_capacity_mw", "Elektrolyseur", "%"),
        ("h2_storage_capacity_kg", "H₂-Speicher", "%"),
    )
    figure, axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=(17, max(7.0, 0.48 * len(plot_data))),
        sharey=True,
    )
    for index, (axis, (column, title, unit)) in enumerate(zip(axes, metric_specs, strict=True)):
        raw = pd.to_numeric(plot_data[column]).to_numpy(dtype=float)
        if unit == "%":
            baseline_value = float(baseline[column])
            values = 100.0 * (raw - baseline_value) / baseline_value
            xlabel = "Änderung [%]"
        else:
            values = raw
            xlabel = "Kapazität [MW]"
        axis.barh(labels, values, color="#2563eb")
        axis.axvline(0.0, color="#334155", linewidth=0.9)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", visible=False)
        if index > 0:
            axis.tick_params(axis="y", labelleft=False)
    figure.suptitle(
        f"Auslegungsänderungen gegenüber dem Basisfall – {SCENARIO_LABELS[scenario]}",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def _plot_wind_break_even(
    selected: pd.DataFrame,
    destination: Path,
    scenario: Scenario,
    dpi: int,
) -> Path:
    wind_rows = selected.loc[
        selected["sensitivity_parameter"] == "wind_capex_eur_per_kw"
    ].copy()
    baseline = selected.loc[selected["sensitivity_parameter"] == "baseline"].iloc[0].copy()
    baseline["sensitivity_value"] = float(wind_rows["baseline_parameter_value"].iloc[0])
    wind_rows = pd.concat([wind_rows, baseline.to_frame().T], ignore_index=True)
    wind_rows["sensitivity_value"] = pd.to_numeric(wind_rows["sensitivity_value"])
    wind_rows = wind_rows.sort_values("sensitivity_value")
    x = wind_rows["sensitivity_value"].to_numpy(dtype=float)
    wind_capacity = pd.to_numeric(wind_rows["wind_capacity_mw"]).to_numpy(dtype=float)
    lcoh = pd.to_numeric(wind_rows["lcoh_eur_per_kg_h2"]).to_numpy(dtype=float)

    figure, left = plt.subplots(figsize=(9.5, 5.8))
    right = left.twinx()
    line_capacity = left.plot(x, wind_capacity, marker="o", color="#0891b2", linewidth=2.2, label="Windkapazität")
    line_lcoh = right.plot(x, lcoh, marker="s", color="#7c3aed", linewidth=2.0, label="LCOH")
    left.set_xlabel("Wind-CAPEX [EUR₍₂₀₂₃₎/kW]")
    left.set_ylabel("Optimale Windkapazität [MW]", color="#0891b2")
    right.set_ylabel("LCOH [EUR₍₂₀₂₃₎/kg H₂]", color="#7c3aed")
    left.tick_params(axis="y", labelcolor="#0891b2")
    right.tick_params(axis="y", labelcolor="#7c3aed")
    left.set_title(
        f"Explorative Wind-CAPEX-Break-even-Analyse – {SCENARIO_LABELS[scenario]}",
        fontweight="bold",
    )
    left.legend(line_capacity + line_lcoh, [line.get_label() for line in line_capacity + line_lcoh], loc="best")
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        description="Erzeugt LCOH-, Auslegungs- und Wind-Break-even-Abbildungen aus Schritt 14."
    )
    parser.add_argument("--comparison", required=True, type=Path, help="sensitivity_comparison.csv")
    parser.add_argument("--output-dir", type=Path, help="Standard: figures neben der Vergleichstabelle")
    parser.add_argument(
        "--scenario",
        choices=tuple(scenario.value for scenario in Scenario),
        default=Scenario.RED_HOURLY.value,
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        print("\nEs wurden noch keine Sensitivitätsabbildungen erzeugt.")
        return 0
    args = parser.parse_args(arguments)
    try:
        artifacts = plot_h2_sensitivity_results(
            args.comparison,
            output_directory=args.output_dir,
            scenario=args.scenario,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, OSError, TypeError, ValueError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    print("Sensitivitätsabbildungen erfolgreich erzeugt.")
    for path in artifacts.figure_paths:
        print(path)
    print(f"Manifest: {artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    return_code = main()
    if return_code:
        raise SystemExit(return_code)


__all__ = [
    "SensitivityFigureArtifacts",
    "build_argument_parser",
    "main",
    "plot_h2_sensitivity_results",
]
