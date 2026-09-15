"""Tests für die Ergebnisdarstellung der Sensitivitätsanalyse."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from plot_h2_sensitivity import main, plot_h2_sensitivity_results


def write_comparison(path: Path) -> None:
    rows = []
    specifications = (
        (0, "baseline", "Basisfall", "baseline", None, 7.4, 100.0, 0.0, 64.0, 18_000.0),
        (1, "pv_low", "PV-CAPEX niedrig", "pv_capex_eur_per_kw", 700.0, 6.8, 110.0, 0.0, 66.0, 17_000.0),
        (2, "wind_quarter", "Wind-CAPEX 25 Prozent", "wind_capex_eur_per_kw", 445.0, 7.2, 90.0, 20.0, 63.0, 19_000.0),
        (3, "wind_half", "Wind-CAPEX 50 Prozent", "wind_capex_eur_per_kw", 890.0, 7.4, 100.0, 0.0, 64.0, 18_000.0),
    )
    for order, case_id, label, parameter, value, lcoh, pv, wind, electrolyzer, storage in specifications:
        rows.append(
            {
                "case_order": order,
                "case_id": case_id,
                "case_label": label,
                "scenario": "red_hourly",
                "sensitivity_parameter": parameter,
                "sensitivity_value": value,
                "baseline_parameter_value": 1779.5 if parameter == "wind_capex_eur_per_kw" else 921.0,
                "lcoh_eur_per_kg_h2": lcoh,
                "lcoh_change_percent_vs_baseline": 100.0 * (lcoh - 7.4) / 7.4,
                "pv_capacity_mw": pv,
                "wind_capacity_mw": wind,
                "electrolyzer_capacity_mw": electrolyzer,
                "h2_storage_capacity_kg": storage,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_three_figures_and_manifest_are_created(tmp_path: Path) -> None:
    comparison = tmp_path / "sensitivity_comparison.csv"
    write_comparison(comparison)
    artifacts = plot_h2_sensitivity_results(
        comparison,
        output_directory=tmp_path / "figures",
        dpi=72,
    )
    assert [path.name for path in artifacts.figure_paths] == [
        "01_lcoh_sensitivity.png",
        "02_design_sensitivity.png",
        "03_wind_break_even.png",
    ]
    assert all(path.stat().st_size > 1_000 for path in artifacts.figure_paths)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["scenario"] == "red_hourly"
    assert manifest["number_of_variants"] == 3


def test_existing_figures_require_overwrite(tmp_path: Path) -> None:
    comparison = tmp_path / "sensitivity_comparison.csv"
    write_comparison(comparison)
    output = tmp_path / "figures"
    plot_h2_sensitivity_results(comparison, output_directory=output, dpi=72)
    with pytest.raises(FileExistsError, match="--overwrite"):
        plot_h2_sensitivity_results(comparison, output_directory=output, dpi=72)


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "keine Sensitivitätsabbildungen" in capsys.readouterr().out
