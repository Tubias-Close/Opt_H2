"""Funktions- und Kommandozeilentests für run_single_site_h2.py."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_single_site_h2 import H2RunError, main, run_single_site_h2


def write_24_hour_input(path: Path) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=24, freq="h"),
            "pv_capacity_factor": np.zeros(24),
            "wind_capacity_factor": np.zeros(24),
            "electricity_price": np.full(24, 50.0),
            "grid_emission_factor": np.full(24, 300.0),
            "h2_demand": np.full(24, 100.0),
        }
    )
    data.to_csv(path, index=False)
    return data


def write_red_24_hour_input(path: Path) -> pd.DataFrame:
    data = write_24_hour_input(path)
    data["pv_capacity_factor"] = np.r_[np.ones(12), np.zeros(12)]
    data["electricity_price"] = np.zeros(24)
    data.to_csv(path, index=False)
    return data


def test_python_function_writes_complete_reproducible_result(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_24_hour_input(input_path)
    output_directory = tmp_path / "result"

    artifacts = run_single_site_h2(input_path, output_directory)

    assert artifacts.output_directory == output_directory.resolve()
    assert artifacts.summary_path.is_file()
    assert artifacts.hourly_operation_path.is_file()
    assert artifacts.validated_input_path.is_file()
    assert artifacts.metadata_path.is_file()

    summary = pd.read_csv(artifacts.summary_path)
    hourly_operation = pd.read_csv(artifacts.hourly_operation_path)
    validated_input = pd.read_csv(artifacts.validated_input_path)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    expected_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert len(summary) == 1
    assert summary.loc[0, "solver_status"] == "optimal"
    assert summary.loc[0, "solver_name"] == "gurobi"
    assert summary.loc[0, "scenario"] == "reference"
    assert summary.loc[0, "number_of_hours"] == 24
    assert summary.loc[0, "input_sha256"] == expected_hash
    assert summary.loc[0, "lcoh_eur_per_kg_h2"] == pytest.approx(
        artifacts.result.lcoh_eur_per_kg_h2
    )
    assert len(hourly_operation) == 24
    assert len(validated_input) == 24
    assert metadata["schema_version"] == "1.3"
    assert metadata["input"]["sha256"] == expected_hash
    assert metadata["input"]["time_zone"] == "UTC"
    assert metadata["result"]["optimality_gap_fraction"] == pytest.approx(0.0)
    assert metadata["result"]["solver_name"] == "gurobi"
    emissions = metadata["result"]["emissions"]
    assert emissions["regulatory_intensity_kg_co2e_per_kg_h2"] == pytest.approx(
        artifacts.result.regulatory_emission_intensity_kg_co2e_per_kg_h2
    )
    assert emissions["red_iii_ghg_compliant"] is False
    assert (
        summary.loc[0, "red_iii_maximum_product_intensity_kg_co2e_per_kg_h2"]
        == pytest.approx(3.384)
    )
    assert metadata["result"]["balance_diagnostics"][
        "max_electricity_balance_residual_mwh"
    ] <= 1e-6
    assert metadata["output_files"]["summary"] == "summary.csv"
    assert "technologies.electrolyzer.capex_eur_per_kw" in metadata[
        "model_parameters"
    ]


def test_monthly_red_run_exports_independent_temporal_check(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_red_24_hour_input(input_path)
    output_directory = tmp_path / "red_monthly"

    artifacts = run_single_site_h2(
        input_path,
        output_directory,
        scenario="red_monthly",
        solver_backend="scipy-highs",
    )

    summary = pd.read_csv(artifacts.summary_path)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert summary.loc[0, "scenario"] == "red_monthly"
    assert bool(summary.loc[0, "red_iii_temporal_compliant"])
    assert summary.loc[0, "red_iii_temporal_mode"] == "monthly"
    assert summary.loc[0, "max_red_iii_temporal_deficit_mwh"] <= 1e-6
    check = metadata["result"]["red_iii_temporal_check"]
    assert check["mode"] == "monthly"
    assert check["compliant"] is True
    assert check["number_of_correlation_periods"] == 1
    emissions = metadata["result"]["emissions"]
    assert emissions["operational_grid_emissions_kg_co2e_per_year"] > 0.0
    assert emissions["regulatory_emissions_kg_co2e_per_year"] == pytest.approx(
        0.0, abs=1e-7
    )
    assert emissions["red_iii_ghg_compliant"] is True


def test_hourly_red_run_exports_independent_temporal_check(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_red_24_hour_input(input_path)
    output_directory = tmp_path / "red_hourly"

    artifacts = run_single_site_h2(
        input_path,
        output_directory,
        scenario="red_hourly",
        solver_backend="scipy-highs",
    )

    summary = pd.read_csv(artifacts.summary_path)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert summary.loc[0, "scenario"] == "red_hourly"
    assert bool(summary.loc[0, "red_iii_temporal_compliant"])
    assert summary.loc[0, "red_iii_temporal_mode"] == "hourly"
    assert summary.loc[0, "red_iii_correlation_periods"] == 24
    assert summary.loc[0, "max_red_iii_temporal_deficit_mwh"] <= 1e-6
    check = metadata["result"]["red_iii_temporal_check"]
    assert check["mode"] == "hourly"
    assert check["compliant"] is True
    assert check["number_of_correlation_periods"] == 24
    assert "Niedrigpreis-Ausnahme" in check["scope_note"]


def test_existing_outputs_require_explicit_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_24_hour_input(input_path)
    output_directory = tmp_path / "result"
    first = run_single_site_h2(input_path, output_directory)

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_single_site_h2(input_path, output_directory)

    second = run_single_site_h2(
        input_path,
        output_directory,
        overwrite=True,
    )
    assert second.summary_path == first.summary_path
    assert second.summary_path.is_file()


def test_input_file_is_never_overwritten_as_an_output(tmp_path: Path) -> None:
    input_path = tmp_path / "summary.csv"
    original_bytes = write_24_hour_input(input_path).to_csv(index=False).encode()
    assert input_path.read_bytes() == original_bytes

    with pytest.raises(H2RunError, match="denselben Pfad"):
        run_single_site_h2(
            input_path,
            tmp_path,
            overwrite=True,
        )
    assert input_path.read_bytes() == original_bytes


def test_main_runs_same_24_hour_case(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = tmp_path / "input.csv"
    write_24_hour_input(input_path)
    output_directory = tmp_path / "result"

    return_code = main(
        [
            "--input",
            str(input_path),
            "--scenario",
            "reference",
            "--output-dir",
            str(output_directory),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "Modelllauf erfolgreich abgeschlossen" in captured.out
    assert "LCOH:" in captured.out
    assert (output_directory / "summary.csv").is_file()


def test_command_line_process_runs_successfully(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script_path = repository_root / "run_single_site_h2.py"
    if not script_path.is_file():
        repository_root = Path(__file__).resolve().parent
        script_path = repository_root / "run_single_site_h2.py"
    input_path = tmp_path / "input.csv"
    write_24_hour_input(input_path)
    output_directory = tmp_path / "cli_result"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input",
            str(input_path),
            "--scenario",
            "reference",
            "--output-dir",
            str(output_directory),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Modelllauf erfolgreich abgeschlossen" in completed.stdout
    assert (output_directory / "summary.csv").is_file()
    assert (output_directory / "hourly_operation.csv").is_file()
    assert (output_directory / "validated_input.csv").is_file()
    assert (output_directory / "run_metadata.json").is_file()


def test_no_arguments_prints_help_without_starting_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    return_code = main([])
    captured = capsys.readouterr()
    assert return_code == 0
    assert "usage:" in captured.out
    assert "Es wurde noch kein Modelllauf gestartet" in captured.out


def test_missing_input_file_returns_clear_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_input = tmp_path / "missing.csv"
    return_code = main(
        [
            "--input",
            str(missing_input),
            "--output-dir",
            str(tmp_path / "result"),
        ]
    )
    captured = capsys.readouterr()
    assert return_code == 1
    assert "Eingabedatei wurde nicht gefunden" in captured.err
    assert not (tmp_path / "result").exists()
