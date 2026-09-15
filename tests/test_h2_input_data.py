"""Grenzfalltests für das standardisierte stündliche H2-Eingabeformat."""

import numpy as np
import pandas as pd
import pytest

from config_h2 import DEFAULT_CONFIG
from h2_input_data import (
    HOURLY_COLUMN_UNITS,
    REQUIRED_HOURLY_COLUMNS,
    HourlyInputError,
    validate_hourly_input,
)


def synthetic_24_hours() -> pd.DataFrame:
    """Erzeuge einen kleinen, vollständig nachvollziehbaren Eingabefall."""

    hours = np.arange(24)
    pv_profile = np.maximum(0.0, np.sin((hours - 6) * np.pi / 12.0))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01 00:00", periods=24, freq="h"
            ),
            "pv_capacity_factor": pv_profile,
            "wind_capacity_factor": np.full(24, 0.45),
            "electricity_price": np.linspace(40.0, 100.0, 24),
            "grid_emission_factor": np.full(24, 300.0),
            "h2_demand": np.full(
                24, DEFAULT_CONFIG.study.h2_demand_kg_per_hour
            ),
        }
    )


def test_synthetic_24_hour_input_is_canonical_and_valid() -> None:
    raw = synthetic_24_hours()
    validated = validate_hourly_input(raw, expected_hours=24)

    assert validated is not raw
    assert tuple(validated.columns) == REQUIRED_HOURLY_COLUMNS
    assert str(validated["timestamp"].dtype) == "datetime64[us, UTC]"
    assert validated.attrs["column_units"] == dict(HOURLY_COLUMN_UNITS)
    assert validated.attrs["time_zone"] == "UTC"
    assert validated.attrs["frequency"] == "1h"
    assert validated["h2_demand"].sum() == pytest.approx(10_000.0)


def test_validation_does_not_modify_original_dataframe() -> None:
    raw = synthetic_24_hours()
    original_dtype = raw["timestamp"].dtype
    validate_hourly_input(raw, expected_hours=24)
    assert raw["timestamp"].dtype == original_dtype
    assert raw.attrs == {}


def test_additional_columns_are_preserved() -> None:
    raw = synthetic_24_hours()
    raw["source_label"] = "synthetic"
    validated = validate_hourly_input(raw)
    assert (validated["source_label"] == "synthetic").all()


def test_negative_electricity_prices_are_allowed() -> None:
    raw = synthetic_24_hours()
    raw.loc[3, "electricity_price"] = -25.0
    validated = validate_hourly_input(raw)
    assert validated.loc[3, "electricity_price"] == -25.0


@pytest.mark.parametrize("missing_column", REQUIRED_HOURLY_COLUMNS)
def test_missing_required_column_is_rejected(missing_column: str) -> None:
    raw = synthetic_24_hours().drop(columns=missing_column)
    with pytest.raises(HourlyInputError, match="Pflichtspalten fehlen"):
        validate_hourly_input(raw)


def test_duplicate_column_name_is_rejected() -> None:
    raw = synthetic_24_hours()
    raw.columns = [*raw.columns[:-1], "grid_emission_factor"]
    with pytest.raises(HourlyInputError, match="Spaltennamen"):
        validate_hourly_input(raw)


def test_duplicate_timestamp_is_rejected() -> None:
    raw = synthetic_24_hours()
    raw.loc[5, "timestamp"] = raw.loc[4, "timestamp"]
    with pytest.raises(HourlyInputError, match="doppelten Zeitstempel"):
        validate_hourly_input(raw)


def test_missing_hour_is_rejected() -> None:
    raw = synthetic_24_hours().drop(index=5).reset_index(drop=True)
    with pytest.raises(HourlyInputError, match="lückenlos stündlich"):
        validate_hourly_input(raw)


def test_unsorted_timestamps_are_rejected() -> None:
    raw = synthetic_24_hours()
    raw.loc[[4, 5], "timestamp"] = raw.loc[[5, 4], "timestamp"].to_numpy()
    with pytest.raises(HourlyInputError, match="aufsteigend sortiert"):
        validate_hourly_input(raw)


@pytest.mark.parametrize("bad_timestamp", ["kein-datum", None])
def test_invalid_timestamp_is_rejected(bad_timestamp: object) -> None:
    raw = synthetic_24_hours()
    raw["timestamp"] = raw["timestamp"].astype(object)
    raw.loc[2, "timestamp"] = bad_timestamp
    with pytest.raises(HourlyInputError, match="timestamp"):
        validate_hourly_input(raw)


@pytest.mark.parametrize(
    "column",
    [
        "pv_capacity_factor",
        "wind_capacity_factor",
        "electricity_price",
        "grid_emission_factor",
        "h2_demand",
    ],
)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_numeric_values_are_rejected(column: str, bad_value: float) -> None:
    raw = synthetic_24_hours()
    raw.loc[7, column] = bad_value
    with pytest.raises(HourlyInputError, match="NaN oder einen unendlichen Wert"):
        validate_hourly_input(raw)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("pv_capacity_factor", -0.001),
        ("pv_capacity_factor", 1.001),
        ("wind_capacity_factor", -0.001),
        ("wind_capacity_factor", 1.001),
    ],
)
def test_capacity_factor_outside_unit_interval_is_rejected(
    column: str, bad_value: float
) -> None:
    raw = synthetic_24_hours()
    raw.loc[8, column] = bad_value
    with pytest.raises(HourlyInputError, match="zwischen 0 und 1"):
        validate_hourly_input(raw)


@pytest.mark.parametrize(
    "column", ["grid_emission_factor", "h2_demand"]
)
def test_negative_emissions_or_demand_are_rejected(column: str) -> None:
    raw = synthetic_24_hours()
    raw.loc[9, column] = -0.1
    with pytest.raises(HourlyInputError, match="darf nicht negativ"):
        validate_hourly_input(raw)


def test_zero_total_h2_demand_is_rejected() -> None:
    raw = synthetic_24_hours()
    raw["h2_demand"] = 0.0
    with pytest.raises(HourlyInputError, match="größer als null"):
        validate_hourly_input(raw)


def test_wrong_expected_number_of_hours_is_rejected() -> None:
    with pytest.raises(HourlyInputError, match="Erwartet wurden 168 Stunden"):
        validate_hourly_input(synthetic_24_hours(), expected_hours=168)


@pytest.mark.parametrize("expected_hours", [0, -1])
def test_nonpositive_expected_hours_are_rejected(expected_hours: int) -> None:
    with pytest.raises(ValueError, match="positiv"):
        validate_hourly_input(
            synthetic_24_hours(), expected_hours=expected_hours
        )


def test_noninteger_expected_hours_are_rejected() -> None:
    with pytest.raises(TypeError, match="ganze Zahl"):
        validate_hourly_input(synthetic_24_hours(), expected_hours=24.0)  # type: ignore[arg-type]
