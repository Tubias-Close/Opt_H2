"""Tests der unabhängigen Abschlussvalidierung."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_h2_scenarios import run_h2_scenarios
from validate_h2_results import main, validate_h2_results


def write_feasible_input(path: Path) -> None:
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=24, freq="h"),
            "pv_capacity_factor": np.ones(24),
            "wind_capacity_factor": np.zeros(24),
            "electricity_price": np.full(24, 50.0),
            "grid_emission_factor": np.full(24, 300.0),
            "h2_demand": np.full(24, 100.0),
        }
    ).to_csv(path, index=False)


def test_independent_validation_passes_complete_scenario_export(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_feasible_input(input_path)
    scenario_results = run_h2_scenarios(
        input_path,
        tmp_path / "scenarios",
        solver_backend="scipy-highs",
    )
    artifacts = validate_h2_results(
        scenario_results.output_directory,
        expected_hours=24,
    )
    assert artifacts.all_checks_passed is True
    checks = pd.read_csv(artifacts.checks_path)
    assert checks["passed"].all()
    assert set(checks["scenario"]) == {
        "all",
        "reference",
        "red_monthly",
        "red_hourly",
    }
    samples = pd.read_csv(artifacts.samples_path)
    assert len(samples) == 9
    assert samples["electricity_balance_residual_mwh"].abs().max() < 1e-6
    assert samples["hydrogen_balance_residual_kg"].abs().max() < 1e-4
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert report["all_checks_passed"] is True
    assert report["failed_checks"] == 0


def test_existing_validation_requires_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_feasible_input(input_path)
    scenario_results = run_h2_scenarios(
        input_path,
        tmp_path / "scenarios",
        solver_backend="scipy-highs",
    )
    validate_h2_results(scenario_results.output_directory, expected_hours=24)
    with pytest.raises(FileExistsError, match="--overwrite"):
        validate_h2_results(scenario_results.output_directory, expected_hours=24)


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "keine Ergebnisvalidierung" in capsys.readouterr().out
