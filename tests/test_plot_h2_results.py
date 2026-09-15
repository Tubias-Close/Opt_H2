"""Tests für die reproduzierbaren Ergebnisabbildungen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plot_h2_results import FIGURE_FILENAMES, H2PlotError, main, plot_h2_results


def write_mock_batch(root: Path, *, mismatched_hash: bool = False) -> None:
    root.mkdir()
    scenarios = (
        ("S0", "Referenz", "S0_reference", 5.0, 100.0, 60.0, 6_000.0),
        ("S1", "RED III monatlich", "S1_red_monthly", 5.1, 102.0, 61.0, 7_000.0),
        ("S2", "RED III stündlich", "S2_red_hourly", 5.4, 110.0, 65.0, 18_000.0),
    )
    rows: list[dict[str, object]] = []
    timestamps = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    pv_profile = np.maximum(0.0, np.sin(np.linspace(-np.pi / 2, 7 * np.pi / 2, 48)))
    for index, (scenario_id, label, directory, lcoh, pv_mw, electrolyzer_mw, storage_kg) in enumerate(scenarios):
        input_hash = "b" * 64 if mismatched_hash and index == 2 else "a" * 64
        annual_h2 = 3_650_000.0
        row: dict[str, object] = {
            "scenario_id": scenario_id,
            "scenario_label": label,
            "result_directory": directory,
            "input_sha256": input_hash,
            "annualization_factor": 182.5,
            "annual_h2_delivered_kg": annual_h2,
            "lcoh_eur_per_kg_h2": lcoh,
            "pv_capacity_mw": pv_mw,
            "wind_capacity_mw": 5.0,
            "electrolyzer_capacity_mw": electrolyzer_mw,
            "compressor_capacity_mw": 2.0,
            "h2_storage_capacity_kg": storage_kg,
            "operational_emission_intensity_kg_co2e_per_kg_h2": 0.5 - 0.2 * index,
            "regulatory_emission_intensity_kg_co2e_per_kg_h2": 0.5 if index == 0 else 0.0,
            "red_iii_ghg_savings_fraction": 0.95 + 0.025 * index,
            "red_iii_ghg_compliant": True,
            "red_iii_maximum_product_intensity_kg_co2e_per_kg_h2": 3.384,
        }
        annual_cost = lcoh * annual_h2
        cost_shares = {
            "pv_annualized_capex_eur_per_year": 0.25,
            "pv_fixed_opex_eur_per_year": 0.05,
            "pv_variable_opex_eur_per_year": 0.00,
            "wind_annualized_capex_eur_per_year": 0.03,
            "wind_fixed_opex_eur_per_year": 0.01,
            "wind_variable_opex_eur_per_year": 0.00,
            "electrolyzer_annualized_capex_eur_per_year": 0.35,
            "electrolyzer_fixed_opex_eur_per_year": 0.07,
            "compressor_annualized_capex_eur_per_year": 0.05,
            "compressor_fixed_opex_eur_per_year": 0.02,
            "h2_storage_annualized_capex_eur_per_year": 0.05,
            "h2_storage_fixed_opex_eur_per_year": 0.01,
            "grid_electricity_eur_per_year": 0.10,
            "water_eur_per_year": 0.02,
        }
        row.update({column: share * annual_cost for column, share in cost_shares.items()})
        rows.append(row)

        scenario_directory = root / directory
        scenario_directory.mkdir()
        renewable = pv_profile * pv_mw
        grid = np.full(48, 4.0 if index < 2 else 0.0)
        hourly = pd.DataFrame(
            {
                "timestamp": timestamps,
                "pv_generation_mwh": renewable,
                "wind_generation_mwh": np.full(48, 1.0),
                "grid_import_mwh": grid,
                "rf_nbo_electricity_mwh": np.full(48, 30.0),
                "electrolyzer_electricity_mwh": np.full(48, 28.0),
                "h2_storage_level_kg": np.linspace(0.0, storage_kg, 48),
                "pv_curtailment_mwh": renewable * 0.1,
                "wind_curtailment_mwh": np.zeros(48),
            }
        )
        hourly.to_csv(scenario_directory / "hourly_operation.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "scenario_comparison.csv", index=False)


def test_plotter_creates_six_pngs_and_manifest(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_mock_batch(results)

    artifacts = plot_h2_results(
        results,
        start="2025-01-01T06:00:00Z",
        hours=24,
        dpi=100,
    )

    assert tuple(path.name for path in artifacts.figure_paths) == FIGURE_FILENAMES
    for figure_path in artifacts.figure_paths:
        assert figure_path.is_file()
        assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert figure_path.stat().st_size > 10_000
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["time_window"]["number_of_hours"] == 24
    assert manifest["time_window"]["start_utc"] == "2025-01-01T06:00:00+00:00"
    assert manifest["scenarios"] == ["S0", "S1", "S2"]


def test_existing_figures_require_overwrite(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_mock_batch(results)
    first = plot_h2_results(results, hours=12, dpi=72)

    with pytest.raises(FileExistsError, match="--overwrite"):
        plot_h2_results(results, hours=12, dpi=72)
    second = plot_h2_results(results, hours=12, dpi=72, overwrite=True)
    assert second.figure_paths == first.figure_paths


def test_mismatched_input_hash_is_rejected(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_mock_batch(results, mismatched_hash=True)

    with pytest.raises(H2PlotError, match="Eingabe-Hash"):
        plot_h2_results(results, dpi=72)


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "--results-dir" in capsys.readouterr().out
