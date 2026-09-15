"""Tests des tabellengesteuerten H2-Sensitivitätslaufs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config_h2 import DEFAULT_CONFIG, Scenario, with_uniform_real_wacc
from run_h2_sensitivity import load_sensitivity_cases, main, run_h2_sensitivity


def write_feasible_input(path: Path) -> None:
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=24, freq="h"),
            "pv_capacity_factor": np.ones(24),
            "wind_capacity_factor": np.full(24, 0.5),
            "electricity_price": np.full(24, 100.0),
            "grid_emission_factor": np.full(24, 300.0),
            "h2_demand": np.full(24, 100.0),
        }
    ).to_csv(path, index=False)


def write_cases(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "case_id": "baseline",
                "parameter": "baseline",
                "level": "base",
                "label": "Basisfall",
                "value": "",
                "unit": "-",
                "source": "Testbasis",
                "note": "",
            },
            {
                "case_id": "price_low",
                "parameter": "electricity_price_eur_per_mwh",
                "level": "low",
                "label": "Niedriger Strompreis",
                "value": "50",
                "unit": "EUR_2023/MWh",
                "source": "Methodischer Testfall",
                "note": "",
            },
            {
                "case_id": "electrolyzer_low",
                "parameter": "electrolyzer_capex_eur_per_kw",
                "level": "low",
                "label": "Niedrige Elektrolyseur-CAPEX",
                "value": "900",
                "unit": "EUR_2023/kW",
                "source": "Methodischer Testfall",
                "note": "",
            },
        ]
    ).to_csv(path, index=False)


def test_case_table_requires_first_unique_baseline(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.csv"
    write_cases(cases_path)
    cases = load_sensitivity_cases(cases_path)
    assert [case.case_id for case in cases] == [
        "baseline",
        "price_low",
        "electrolyzer_low",
    ]
    assert cases[0].value is None


def test_one_factor_runs_keep_base_config_and_input_separate(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    cases_path = tmp_path / "cases.csv"
    write_feasible_input(input_path)
    write_cases(cases_path)
    base_config = with_uniform_real_wacc(
        DEFAULT_CONFIG,
        0.11,
        source="Regionaler Testwert",
    )

    artifacts = run_h2_sensitivity(
        input_path,
        cases_path,
        tmp_path / "results",
        scenarios=(Scenario.REFERENCE, Scenario.RED_HOURLY),
        config=base_config,
        solver_backend="scipy-highs",
    )

    comparison = pd.read_csv(artifacts.comparison_path)
    assert len(comparison) == 6
    assert set(comparison["case_id"]) == {
        "baseline",
        "price_low",
        "electrolyzer_low",
    }
    assert set(comparison["scenario"]) == {"reference", "red_hourly"}
    assert comparison.loc[
        comparison["case_id"] == "baseline", "lcoh_change_percent_vs_baseline"
    ].abs().max() == pytest.approx(0.0)
    assert (
        pd.read_csv(artifacts.output_directory / "case_inputs" / "price_low.csv")[
            "electricity_price"
        ]
        == 50.0
    ).all()
    assert (
        pd.read_csv(
            artifacts.output_directory / "case_inputs" / "electrolyzer_low.csv"
        )["electricity_price"]
        == 100.0
    ).all()
    assert (
        DEFAULT_CONFIG.technologies.electrolyzer.capex_eur_per_kw.value
        == pytest.approx(1_297.0)
    )
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["number_of_cases"] == 3
    assert metadata["number_of_optimization_runs"] == 6
    assert metadata["baseline_parameter_values"]["uniform_real_wacc_fraction"] == pytest.approx(0.11)


def test_existing_results_require_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    cases_path = tmp_path / "cases.csv"
    output_path = tmp_path / "results"
    write_feasible_input(input_path)
    write_cases(cases_path)
    run_h2_sensitivity(
        input_path,
        cases_path,
        output_path,
        solver_backend="scipy-highs",
    )
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_h2_sensitivity(
            input_path,
            cases_path,
            output_path,
            solver_backend="scipy-highs",
        )


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "keine Sensitivitätsanalyse gestartet" in capsys.readouterr().out
