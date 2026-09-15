"""Tests des einheitlichen H2-Szenarienlaufs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_h2_scenarios import main, run_h2_scenarios


def write_feasible_input(path: Path) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=24, freq="h"),
            "pv_capacity_factor": np.ones(24),
            "wind_capacity_factor": np.zeros(24),
            "electricity_price": np.full(24, 50.0),
            "grid_emission_factor": np.full(24, 300.0),
            "h2_demand": np.full(24, 100.0),
        }
    )
    data.to_csv(path, index=False)
    return data


def test_one_call_writes_three_separate_reproducible_scenarios(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.csv"
    write_feasible_input(input_path)
    output_root = tmp_path / "comparison"

    artifacts = run_h2_scenarios(
        input_path,
        output_root,
        solver_backend="scipy-highs",
    )

    assert list(artifacts.scenario_runs) == ["S0", "S1", "S2"]
    assert artifacts.comparison_path.is_file()
    assert artifacts.metadata_path.is_file()
    comparison = pd.read_csv(artifacts.comparison_path)
    assert comparison["scenario_id"].tolist() == ["S0", "S1", "S2"]
    assert comparison["scenario"].tolist() == [
        "reference",
        "red_monthly",
        "red_hourly",
    ]
    assert comparison.loc[0, "lcoh_change_percent_vs_s0"] == pytest.approx(0.0)
    assert comparison["red_iii_ghg_compliant"].tolist() == [True, True, True]
    expected_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert comparison["input_sha256"].nunique() == 1
    assert comparison.loc[0, "input_sha256"] == expected_hash
    for directory_name in ("S0_reference", "S1_red_monthly", "S2_red_hourly"):
        scenario_directory = output_root / directory_name
        assert (scenario_directory / "summary.csv").is_file()
        assert (scenario_directory / "hourly_operation.csv").is_file()
        assert (scenario_directory / "validated_input.csv").is_file()
        assert (scenario_directory / "run_metadata.json").is_file()

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "1.0"
    assert metadata["input"]["sha256"] == expected_hash
    assert [entry["scenario_id"] for entry in metadata["scenarios"]] == [
        "S0",
        "S1",
        "S2",
    ]


def test_off_grid_is_added_only_when_requested(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_feasible_input(input_path)

    artifacts = run_h2_scenarios(
        input_path,
        tmp_path / "comparison",
        include_off_grid=True,
        solver_backend="scipy_highs",
    )

    comparison = pd.read_csv(artifacts.comparison_path)
    assert comparison["scenario_id"].tolist() == ["S0", "S1", "S2", "S3"]
    assert comparison.iloc[-1]["scenario"] == "off_grid"
    assert (artifacts.output_directory / "S3_off_grid" / "summary.csv").is_file()


def test_existing_batch_requires_explicit_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_feasible_input(input_path)
    output_root = tmp_path / "comparison"
    first = run_h2_scenarios(
        input_path,
        output_root,
        solver_backend="scipy-highs",
    )
    original_comparison = first.comparison_path.read_bytes()

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_h2_scenarios(
            input_path,
            output_root,
            solver_backend="scipy-highs",
        )
    assert first.comparison_path.read_bytes() == original_comparison

    second = run_h2_scenarios(
        input_path,
        output_root,
        overwrite=True,
        solver_backend="scipy-highs",
    )
    assert second.comparison_path.is_file()


def test_no_arguments_prints_help_without_starting_a_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "--input" in output
    assert "kein Szenarienlauf gestartet" in output


def test_missing_input_returns_controlled_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    return_code = main(
        [
            "--input",
            str(tmp_path / "missing.csv"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert return_code == 1
    assert "nicht gefunden" in capsys.readouterr().err
