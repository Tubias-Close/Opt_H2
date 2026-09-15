r"""Erzeuge die Ergebnisabbildungen des H2-Szenarienvergleichs.

Beispiel::

    python plot_h2_results.py --results-dir outputs_h2\namibia_szenarienvergleich

Das Skript liest ausschließlich die von ``run_h2_scenarios.py`` erzeugte
Vergleichstabelle und die zugehörigen Stundenwerte. Es rechnet kein Szenario
neu und verändert keine Modelldaten.
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
from typing import Final, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPARISON_FILENAME: Final[str] = "scenario_comparison.csv"
MANIFEST_FILENAME: Final[str] = "figure_manifest.json"
FIGURE_FILENAMES: Final[tuple[str, ...]] = (
    "01_lcoh_comparison.png",
    "02_installed_capacities.png",
    "03_lcoh_cost_components.png",
    "04_operational_indicators.png",
    "05_emissions_and_red_iii.png",
    "06_hourly_operation.png",
)

SCENARIO_COLORS: Final[Mapping[str, str]] = {
    "S0": "#64748b",
    "S1": "#2563eb",
    "S2": "#16a34a",
    "S3": "#f59e0b",
}

COST_COMPONENTS: Final[tuple[tuple[str, str, tuple[str, ...], str], ...]] = (
    (
        "renewables",
        "PV und Wind",
        (
            "pv_annualized_capex_eur_per_year",
            "pv_fixed_opex_eur_per_year",
            "pv_variable_opex_eur_per_year",
            "wind_annualized_capex_eur_per_year",
            "wind_fixed_opex_eur_per_year",
            "wind_variable_opex_eur_per_year",
        ),
        "#eab308",
    ),
    (
        "electrolyzer",
        "Elektrolyseur",
        (
            "electrolyzer_annualized_capex_eur_per_year",
            "electrolyzer_fixed_opex_eur_per_year",
        ),
        "#0ea5e9",
    ),
    (
        "compressor",
        "Kompressor",
        (
            "compressor_annualized_capex_eur_per_year",
            "compressor_fixed_opex_eur_per_year",
        ),
        "#8b5cf6",
    ),
    (
        "storage",
        "H₂-Speicher",
        (
            "h2_storage_annualized_capex_eur_per_year",
            "h2_storage_fixed_opex_eur_per_year",
        ),
        "#ec4899",
    ),
    (
        "grid",
        "Netzstrom",
        ("grid_electricity_eur_per_year",),
        "#ef4444",
    ),
    (
        "water",
        "Wasser",
        ("water_eur_per_year",),
        "#14b8a6",
    ),
)


class H2PlotError(RuntimeError):
    """Verständlicher Fehler beim Lesen oder Darstellen der Ergebnisse."""


@dataclass(frozen=True, slots=True)
class PlotArtifacts:
    """Pfade aller erfolgreich erzeugten Abbildungen."""

    output_directory: Path
    figure_paths: tuple[Path, ...]
    manifest_path: Path


def plot_h2_results(
    results_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    hours: int = 168,
    dpi: int = 180,
    overwrite: bool = False,
) -> PlotArtifacts:
    """Erzeuge sechs Abbildungen aus einem gespeicherten Szenarienlauf."""

    if isinstance(hours, bool) or not isinstance(hours, int) or hours <= 0:
        raise ValueError("hours muss eine positive ganze Zahl sein.")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 72:
        raise ValueError("dpi muss eine ganze Zahl von mindestens 72 sein.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite muss ein boolescher Wert sein.")

    results_root = Path(results_directory).expanduser().resolve()
    comparison_path = results_root / COMPARISON_FILENAME
    if not comparison_path.is_file():
        raise FileNotFoundError(
            f"Die Vergleichstabelle wurde nicht gefunden: {comparison_path}"
        )
    output_root = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else results_root / "figures"
    )
    if output_root.exists() and not output_root.is_dir():
        raise H2PlotError(f"Der Ausgabepfad ist kein Verzeichnis: {output_root}")
    figure_paths = tuple(output_root / name for name in FIGURE_FILENAMES)
    manifest_path = output_root / MANIFEST_FILENAME
    existing = [path for path in (*figure_paths, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Abbildungen existieren bereits. Nutze --overwrite zum Ersetzen: "
            + ", ".join(path.name for path in existing)
        )

    comparison = pd.read_csv(comparison_path)
    _validate_comparison(comparison)
    hourly_by_scenario, hourly_paths = _load_hourly_results(
        results_root, comparison
    )
    selected_hourly, selected_start, selected_end = _select_time_window(
        hourly_by_scenario,
        start=start,
        hours=hours,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _apply_plot_style()
    source_note = _source_note(comparison_path, comparison)

    plotters = (
        lambda path: _plot_lcoh(comparison, path, dpi, source_note),
        lambda path: _plot_capacities(comparison, path, dpi, source_note),
        lambda path: _plot_cost_components(comparison, path, dpi, source_note),
        lambda path: _plot_operational_indicators(
            comparison, hourly_by_scenario, path, dpi, source_note
        ),
        lambda path: _plot_emissions(comparison, path, dpi, source_note),
        lambda path: _plot_hourly_operation(
            comparison, selected_hourly, path, dpi, source_note
        ),
    )
    for path, plotter in zip(figure_paths, plotters, strict=True):
        plotter(path)

    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_directory": str(results_root),
        "comparison_file": {
            "path": str(comparison_path),
            "sha256": _sha256(comparison_path),
        },
        "input_sha256": str(comparison["input_sha256"].iloc[0]),
        "scenarios": comparison["scenario_id"].astype(str).tolist(),
        "hourly_sources": {
            scenario_id: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for scenario_id, path in hourly_paths.items()
        },
        "time_window": {
            "start_utc": selected_start.isoformat(),
            "end_utc": selected_end.isoformat(),
            "number_of_hours": len(next(iter(selected_hourly.values()))),
        },
        "dpi": dpi,
        "figures": list(FIGURE_FILENAMES),
    }
    _write_json_atomic(manifest, manifest_path)
    return PlotArtifacts(output_root, figure_paths, manifest_path)


def _validate_comparison(comparison: pd.DataFrame) -> None:
    required = {
        "scenario_id",
        "scenario_label",
        "result_directory",
        "input_sha256",
        "annualization_factor",
        "annual_h2_delivered_kg",
        "lcoh_eur_per_kg_h2",
        "pv_capacity_mw",
        "wind_capacity_mw",
        "electrolyzer_capacity_mw",
        "compressor_capacity_mw",
        "h2_storage_capacity_kg",
        "operational_emission_intensity_kg_co2e_per_kg_h2",
        "regulatory_emission_intensity_kg_co2e_per_kg_h2",
        "red_iii_ghg_savings_fraction",
        "red_iii_ghg_compliant",
        "red_iii_maximum_product_intensity_kg_co2e_per_kg_h2",
    }
    required.update(
        column
        for _, _, columns, _ in COST_COMPONENTS
        for column in columns
    )
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise H2PlotError(
            "In scenario_comparison.csv fehlen Spalten: " + ", ".join(missing)
        )
    if comparison.empty:
        raise H2PlotError("scenario_comparison.csv enthält keine Szenarien.")
    if comparison["scenario_id"].duplicated().any():
        raise H2PlotError("scenario_id muss in der Vergleichstabelle eindeutig sein.")
    if comparison["input_sha256"].nunique(dropna=False) != 1:
        raise H2PlotError("Die Szenarien verwenden nicht denselben Eingabe-Hash.")


def _load_hourly_results(
    results_root: Path,
    comparison: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    hourly_by_scenario: dict[str, pd.DataFrame] = {}
    hourly_paths: dict[str, Path] = {}
    required_columns = {
        "timestamp",
        "pv_generation_mwh",
        "wind_generation_mwh",
        "grid_import_mwh",
        "rf_nbo_electricity_mwh",
        "electrolyzer_electricity_mwh",
        "h2_storage_level_kg",
        "pv_curtailment_mwh",
        "wind_curtailment_mwh",
    }
    common_timestamps: pd.Series | None = None
    for row in comparison.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        path = results_root / str(row.result_directory) / "hourly_operation.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Stundenwerte für {scenario_id} wurden nicht gefunden: {path}"
            )
        hourly = pd.read_csv(path)
        missing = sorted(required_columns.difference(hourly.columns))
        if missing:
            raise H2PlotError(
                f"In {path} fehlen Spalten: " + ", ".join(missing)
            )
        timestamps = pd.to_datetime(hourly["timestamp"], utc=True, errors="coerce")
        if timestamps.isna().any() or timestamps.duplicated().any():
            raise H2PlotError(f"{path} enthält ungültige oder doppelte Zeitstempel.")
        if not timestamps.is_monotonic_increasing:
            raise H2PlotError(f"{path} ist zeitlich nicht aufsteigend sortiert.")
        hourly = hourly.copy()
        hourly["timestamp"] = timestamps
        if common_timestamps is None:
            common_timestamps = timestamps
        elif not timestamps.reset_index(drop=True).equals(
            common_timestamps.reset_index(drop=True)
        ):
            raise H2PlotError("Die Szenarien besitzen unterschiedliche Zeitachsen.")
        hourly_by_scenario[scenario_id] = hourly
        hourly_paths[scenario_id] = path
    return hourly_by_scenario, hourly_paths


def _select_time_window(
    hourly_by_scenario: Mapping[str, pd.DataFrame],
    *,
    start: str | pd.Timestamp | None,
    hours: int,
) -> tuple[dict[str, pd.DataFrame], pd.Timestamp, pd.Timestamp]:
    first = next(iter(hourly_by_scenario.values()))
    if start is None:
        start_timestamp = pd.Timestamp(first["timestamp"].iloc[0])
    else:
        start_timestamp = pd.Timestamp(start)
        if start_timestamp.tzinfo is None:
            start_timestamp = start_timestamp.tz_localize("UTC")
        else:
            start_timestamp = start_timestamp.tz_convert("UTC")
    selected: dict[str, pd.DataFrame] = {}
    for scenario_id, hourly in hourly_by_scenario.items():
        subset = hourly.loc[hourly["timestamp"] >= start_timestamp].head(hours).copy()
        if subset.empty:
            raise H2PlotError(
                f"Ab {start_timestamp.isoformat()} liegen keine Stundenwerte vor."
            )
        selected[scenario_id] = subset
    lengths = {len(frame) for frame in selected.values()}
    if len(lengths) != 1:
        raise H2PlotError("Das ausgewählte Zeitfenster ist nicht für alle Szenarien gleich.")
    selected_start = pd.Timestamp(next(iter(selected.values()))["timestamp"].iloc[0])
    selected_end = pd.Timestamp(next(iter(selected.values()))["timestamp"].iloc[-1])
    return selected, selected_start, selected_end


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
        }
    )


def _labels(comparison: pd.DataFrame) -> list[str]:
    return [
        f"{row.scenario_id}\n{row.scenario_label}"
        for row in comparison.itertuples(index=False)
    ]


def _colors(comparison: pd.DataFrame) -> list[str]:
    return [SCENARIO_COLORS.get(str(value), "#475569") for value in comparison["scenario_id"]]


def _plot_lcoh(
    comparison: pd.DataFrame, path: Path, dpi: int, source_note: str
) -> None:
    values = comparison["lcoh_eur_per_kg_h2"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(_labels(comparison), values, color=_colors(comparison), width=0.62)
    ax.set_title("Wasserstoffgestehungskosten nach Szenario")
    ax.set_ylabel("LCOH [EUR/kg H₂]")
    ax.set_ylim(0.0, max(values) * 1.22)
    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        change = 100.0 * (value / values[0] - 1.0)
        label = f"{value:.3f}"
        if index:
            label += f"\n({change:+.2f} % vs. S0)"
        ax.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom")
    _finish_figure(fig, path, dpi, source_note)


def _plot_capacities(
    comparison: pd.DataFrame, path: Path, dpi: int, source_note: str
) -> None:
    fig, (ax_mw, ax_storage) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [2.2, 1.0]})
    x = np.arange(len(comparison))
    technologies = (
        ("pv_capacity_mw", "PV", "#eab308"),
        ("wind_capacity_mw", "Wind", "#0891b2"),
        ("electrolyzer_capacity_mw", "Elektrolyseur", "#2563eb"),
        ("compressor_capacity_mw", "Kompressor", "#8b5cf6"),
    )
    width = 0.19
    for offset, (column, label, color) in enumerate(technologies):
        values = comparison[column].to_numpy(dtype=float)
        ax_mw.bar(x + (offset - 1.5) * width, values, width, label=label, color=color)
    ax_mw.set_xticks(x, _labels(comparison))
    ax_mw.set_ylabel("Installierte Leistung [MW]")
    ax_mw.set_title("Leistungskapazitäten")
    ax_mw.legend(ncols=2, frameon=False)

    storage = comparison["h2_storage_capacity_kg"].to_numpy(dtype=float)
    bars = ax_storage.bar(_labels(comparison), storage / 1_000.0, color=_colors(comparison), width=0.62)
    ax_storage.set_ylabel("H₂-Speicherkapazität [t H₂]")
    ax_storage.set_title("Druckspeicher")
    for bar, value in zip(bars, storage / 1_000.0, strict=True):
        ax_storage.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom")
    fig.suptitle("Kostenoptimale Anlagenkonfiguration", fontsize=15, fontweight="bold")
    _finish_figure(fig, path, dpi, source_note)


def _plot_cost_components(
    comparison: pd.DataFrame, path: Path, dpi: int, source_note: str
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(comparison))
    annual_h2 = comparison["annual_h2_delivered_kg"].to_numpy(dtype=float)
    bottom = np.zeros(len(comparison))
    for _, label, columns, color in COST_COMPONENTS:
        annual_cost = comparison[list(columns)].sum(axis=1).to_numpy(dtype=float)
        contribution = annual_cost / annual_h2
        ax.bar(x, contribution, bottom=bottom, label=label, color=color, width=0.64)
        bottom += contribution
    total_lcoh = comparison["lcoh_eur_per_kg_h2"].to_numpy(dtype=float)
    for position, value in zip(x, total_lcoh, strict=True):
        ax.text(position, value, f"{value:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x, _labels(comparison))
    ax.set_ylabel("Kostenbeitrag [EUR/kg H₂]")
    ax.set_title("Zusammensetzung der Wasserstoffgestehungskosten")
    ax.legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _finish_figure(fig, path, dpi, source_note, bottom=0.22)


def _plot_operational_indicators(
    comparison: pd.DataFrame,
    hourly_by_scenario: Mapping[str, pd.DataFrame],
    path: Path,
    dpi: int,
    source_note: str,
) -> None:
    full_load_hours: list[float] = []
    grid_gwh: list[float] = []
    curtailment_gwh: list[float] = []
    for row in comparison.itertuples(index=False):
        hourly = hourly_by_scenario[str(row.scenario_id)]
        factor = float(row.annualization_factor)
        capacity = float(row.electrolyzer_capacity_mw)
        full_load_hours.append(
            float(hourly["electrolyzer_electricity_mwh"].sum()) * factor / capacity
        )
        grid_gwh.append(float(hourly["grid_import_mwh"].sum()) * factor / 1_000.0)
        curtailment_gwh.append(
            float(
                hourly[["pv_curtailment_mwh", "wind_curtailment_mwh"]]
                .sum(axis=1)
                .sum()
            )
            * factor
            / 1_000.0
        )
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    values_and_titles = (
        (full_load_hours, "Elektrolyseur-Volllaststunden", "Stunden pro Jahr [h/a]"),
        (grid_gwh, "Physischer Netzbezug", "Strommenge [GWh/a]"),
        (curtailment_gwh, "Abregelung von PV und Wind", "Strommenge [GWh/a]"),
    )
    labels = _labels(comparison)
    colors = _colors(comparison)
    for ax, (values, title, unit) in zip(axes, values_and_titles, strict=True):
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(unit)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.1f}".replace(",", " "), ha="center", va="bottom", fontsize=9)
    fig.suptitle("Betriebskennzahlen des optimalen H₂-Systems", fontsize=15, fontweight="bold")
    _finish_figure(fig, path, dpi, source_note)


def _plot_emissions(
    comparison: pd.DataFrame, path: Path, dpi: int, source_note: str
) -> None:
    fig, (ax_intensity, ax_savings) = plt.subplots(1, 2, figsize=(14, 6.5))
    x = np.arange(len(comparison))
    width = 0.34
    operational = comparison[
        "operational_emission_intensity_kg_co2e_per_kg_h2"
    ].to_numpy(dtype=float)
    regulatory = comparison[
        "regulatory_emission_intensity_kg_co2e_per_kg_h2"
    ].to_numpy(dtype=float)
    threshold = float(
        comparison["red_iii_maximum_product_intensity_kg_co2e_per_kg_h2"].iloc[0]
    )
    operational_bars = ax_intensity.bar(
        x - width / 2,
        operational,
        width,
        label="Betrieblich",
        color="#ef4444",
    )
    regulatory_bars = ax_intensity.bar(
        x + width / 2,
        regulatory,
        width,
        label="Regulatorisch",
        color="#16a34a",
    )
    ax_intensity.axhline(threshold, color="#111827", linestyle="--", label=f"Grenzwert {threshold:.3f}")
    ax_intensity.set_xticks(x, _labels(comparison))
    ax_intensity.set_ylabel("THG-Intensität [kg CO₂e/kg H₂]")
    ax_intensity.set_title("Betriebliche und regulatorische Intensität")
    ax_intensity.legend(frameon=False)
    for bars, values in (
        (operational_bars, operational),
        (regulatory_bars, regulatory),
    ):
        for bar, value in zip(bars, values, strict=True):
            ax_intensity.text(
                bar.get_x() + bar.get_width() / 2,
                max(value, 0.025),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    savings = 100.0 * comparison["red_iii_ghg_savings_fraction"].to_numpy(dtype=float)
    bars = ax_savings.bar(_labels(comparison), savings, color=_colors(comparison), width=0.62)
    ax_savings.axhline(70.0, color="#111827", linestyle="--", label="Mindestwert 70 %")
    ax_savings.set_ylim(0.0, 108.0)
    ax_savings.set_ylabel("THG-Einsparung [%]")
    ax_savings.set_title("Einsparung gegenüber 94 g CO₂e/MJ")
    ax_savings.legend(frameon=False, loc="lower right")
    compliance_values = comparison["red_iii_ghg_compliant"].tolist()
    for bar, value, compliant in zip(
        bars, savings, compliance_values, strict=True
    ):
        status = "ja" if _coerce_bool(compliant) else "nein"
        ax_savings.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f} %\nTHG-Grenzwert: {status}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.suptitle("RED-III-Treibhausgasbewertung", fontsize=15, fontweight="bold")
    _finish_figure(fig, path, dpi, source_note)


def _plot_hourly_operation(
    comparison: pd.DataFrame,
    selected_hourly: Mapping[str, pd.DataFrame],
    path: Path,
    dpi: int,
    source_note: str,
) -> None:
    number_of_scenarios = len(comparison)
    fig, axes = plt.subplots(number_of_scenarios + 1, 1, figsize=(16, 3.0 * number_of_scenarios + 3.5), sharex=True)
    for ax, row in zip(axes[:-1], comparison.itertuples(index=False), strict=True):
        scenario_id = str(row.scenario_id)
        hourly = selected_hourly[scenario_id]
        renewable = hourly["pv_generation_mwh"] + hourly["wind_generation_mwh"]
        ax.fill_between(hourly["timestamp"], 0.0, renewable, color="#facc15", alpha=0.45, label="PV + Wind")
        ax.plot(hourly["timestamp"], hourly["rf_nbo_electricity_mwh"], color="#2563eb", linewidth=1.4, label="Elektrolyse + Verdichtung")
        ax.plot(hourly["timestamp"], hourly["grid_import_mwh"], color="#ef4444", linewidth=1.2, label="Netzbezug")
        ax.set_ylabel("MWh/h")
        ax.set_title(f"{scenario_id} – {row.scenario_label}", loc="left", fontsize=11)
        ax.legend(ncols=3, frameon=False, loc="upper right")
    storage_ax = axes[-1]
    for row in comparison.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        hourly = selected_hourly[scenario_id]
        storage_ax.plot(
            hourly["timestamp"],
            hourly["h2_storage_level_kg"] / 1_000.0,
            color=SCENARIO_COLORS.get(scenario_id, "#475569"),
            linewidth=1.5,
            label=scenario_id,
        )
    storage_ax.set_ylabel("Speicherfüllstand [t H₂]")
    storage_ax.set_title("H₂-Speicher im ausgewählten Zeitraum", loc="left", fontsize=11)
    storage_ax.legend(ncols=len(comparison), frameon=False)
    storage_ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(next(iter(selected_hourly.values()))) // 168)))
    storage_ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))
    storage_ax.set_xlabel("UTC-Zeit")
    fig.suptitle("Stündlicher Betrieb im ausgewählten Zeitraum", fontsize=15, fontweight="bold")
    _finish_figure(fig, path, dpi, source_note)


def _source_note(comparison_path: Path, comparison: pd.DataFrame) -> str:
    input_hash = str(comparison["input_sha256"].iloc[0])
    return (
        f"Quelle: {comparison_path.name}; gemeinsamer Eingabe-Hash: "
        f"{input_hash[:12]}…; eigene Modellrechnung."
    )


def _coerce_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "ja"}:
            return True
        if normalized in {"false", "0", "no", "nein"}:
            return False
    if isinstance(value, (int, float)) and not pd.isna(value):
        if float(value) in {0.0, 1.0}:
            return bool(value)
    raise H2PlotError(f"Ungültiger boolescher Konformitätswert: {value!r}")


def _finish_figure(
    fig: plt.Figure,
    destination: Path,
    dpi: int,
    source_note: str,
    *,
    bottom: float = 0.12,
) -> None:
    fig.text(0.01, 0.012, source_note, ha="left", va="bottom", fontsize=8, color="#64748b")
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.96))
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        fig.savefig(temporary, dpi=dpi, format="png", bbox_inches="tight")
        os.replace(temporary, destination)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()


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
        description="Erzeugt sechs Abbildungen aus run_h2_scenarios.py-Ergebnissen.",
        epilog=(
            "Beispiel: python plot_h2_results.py --results-dir "
            "outputs_h2\\namibia_szenarienvergleich --start 2025-01-01 --hours 168"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Ordner mit scenario_comparison.csv und den Szenariounterordnern.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Ausgabeordner; Standard: Unterordner figures im Ergebnisordner.",
    )
    parser.add_argument(
        "--start",
        help="Erste UTC-Stunde des Zeitreihenausschnitts; Standard: erste Stunde.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=168,
        help="Länge des Zeitreihenausschnitts in Stunden; Standard: 168.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Auflösung der PNG-Dateien; Standard: 180 dpi.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ersetzt vorhandene Abbildungen mit denselben Namen.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        print("\nEs wurden noch keine Abbildungen erzeugt. Gib --results-dir an.")
        return 0
    args = parser.parse_args(arguments)
    try:
        artifacts = plot_h2_results(
            args.results_dir,
            args.output_dir,
            start=args.start,
            hours=args.hours,
            dpi=args.dpi,
            overwrite=args.overwrite,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        H2PlotError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1
    print("Abbildungen erfolgreich erzeugt:")
    for path in artifacts.figure_paths:
        print(f"- {path}")
    print(f"Manifest: {artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    return_code = main()
    if return_code:
        raise SystemExit(return_code)


__all__ = [
    "FIGURE_FILENAMES",
    "H2PlotError",
    "MANIFEST_FILENAME",
    "PlotArtifacts",
    "build_argument_parser",
    "main",
    "plot_h2_results",
]
