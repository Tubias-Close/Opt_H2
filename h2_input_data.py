"""Standardisierte stündliche Eingabedaten für das H2-Optimierungsmodell.

Dieses Modul trennt die Datenprüfung von den späteren Datenquellen. Es ist
unerheblich, ob ein DataFrame aus einem künstlichen Testfall, einer CSV-Datei,
PVGIS oder dem ursprünglichen Modell stammt. Nach erfolgreicher Validierung
besitzt er dieselben Spalten, Einheiten, eine stündliche UTC-Zeitachse und
eindeutige Wertebereiche.

Beim Import werden keine Dateien geöffnet und keine Verzeichnisse angelegt.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

import numpy as np
import pandas as pd


TIMESTAMP_COLUMN: Final[str] = "timestamp"
NUMERIC_HOURLY_COLUMNS: Final[tuple[str, ...]] = (
    "pv_capacity_factor",
    "wind_capacity_factor",
    "electricity_price",
    "grid_emission_factor",
    "h2_demand",
)
REQUIRED_HOURLY_COLUMNS: Final[tuple[str, ...]] = (
    TIMESTAMP_COLUMN,
    *NUMERIC_HOURLY_COLUMNS,
)
HOURLY_COLUMN_UNITS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "timestamp": "UTC",
        "pv_capacity_factor": "fraction",
        "wind_capacity_factor": "fraction",
        "electricity_price": "EUR/MWh",
        "grid_emission_factor": "kg_CO2e/MWh",
        "h2_demand": "kg_H2/h",
    }
)
EXPECTED_TIME_STEP: Final[pd.Timedelta] = pd.Timedelta(hours=1)


class HourlyInputError(ValueError):
    """Verständlicher Validierungsfehler für stündliche Modelldaten."""


def validate_hourly_input(
    data: pd.DataFrame,
    *,
    expected_hours: int | None = None,
) -> pd.DataFrame:
    """Prüfe und vereinheitliche einen stündlichen H2-Eingabedatensatz.

    Parameters
    ----------
    data:
        DataFrame mit mindestens den in ``REQUIRED_HOURLY_COLUMNS`` genannten
        Spalten. Zusätzliche Metadatenspalten bleiben erhalten.
    expected_hours:
        Optionale erwartete Zeilenzahl, beispielsweise 24 für einen Test oder
        8.760 für den vollständigen Basisfall.

    Returns
    -------
    pandas.DataFrame
        Eine Kopie der Eingabe. Zeitstempel sind als UTC normalisiert und die
        fünf Modellgrößen als ``float`` gespeichert. Einheiten und Frequenz
        stehen zusätzlich in ``DataFrame.attrs``.

    Notes
    -----
    Negative Strompreise sind zulässig. Kapazitätsfaktoren müssen zwischen
    null und eins liegen; Emissionsfaktor und H2-Nachfrage dürfen nicht
    negativ sein. Mindestens eine Stunde muss eine positive H2-Nachfrage
    enthalten.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data muss ein pandas.DataFrame sein.")
    if data.empty:
        raise HourlyInputError("Der stündliche Eingabedatensatz ist leer.")
    if data.columns.duplicated().any():
        duplicate_columns = data.columns[data.columns.duplicated()].tolist()
        raise HourlyInputError(
            f"Spaltennamen dürfen nicht doppelt vorkommen: {duplicate_columns}."
        )

    missing_columns = [
        column for column in REQUIRED_HOURLY_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise HourlyInputError(
            "Pflichtspalten fehlen: " + ", ".join(missing_columns) + "."
        )

    _validate_expected_hours(expected_hours)
    if expected_hours is not None and len(data) != expected_hours:
        raise HourlyInputError(
            f"Erwartet wurden {expected_hours} Stunden, vorhanden sind {len(data)}."
        )

    validated = data.copy(deep=True)
    validated[TIMESTAMP_COLUMN] = _parse_utc_timestamps(validated[TIMESTAMP_COLUMN])
    _validate_time_axis(validated[TIMESTAMP_COLUMN])

    for column in NUMERIC_HOURLY_COLUMNS:
        validated[column] = _parse_finite_numeric_column(validated[column], column)

    for column in ("pv_capacity_factor", "wind_capacity_factor"):
        outside_range = ~validated[column].between(0.0, 1.0, inclusive="both")
        if outside_range.any():
            row = int(np.flatnonzero(outside_range.to_numpy())[0])
            value = validated[column].iloc[row]
            raise HourlyInputError(
                f"{column} muss zwischen 0 und 1 liegen; Zeile {row} enthält {value}."
            )

    _require_nonnegative(validated, "grid_emission_factor")
    _require_nonnegative(validated, "h2_demand")
    if validated["h2_demand"].sum() <= 0.0:
        raise HourlyInputError(
            "Die gesamte H2-Nachfrage muss größer als null sein."
        )

    validated.attrs = dict(data.attrs)
    validated.attrs["column_units"] = dict(HOURLY_COLUMN_UNITS)
    validated.attrs["time_zone"] = "UTC"
    validated.attrs["frequency"] = "1h"
    return validated


def _validate_expected_hours(expected_hours: int | None) -> None:
    if expected_hours is None:
        return
    if isinstance(expected_hours, bool) or not isinstance(expected_hours, int):
        raise TypeError("expected_hours muss eine ganze Zahl oder None sein.")
    if expected_hours <= 0:
        raise ValueError("expected_hours muss positiv sein.")


def _parse_utc_timestamps(values: pd.Series) -> pd.Series:
    try:
        timestamps = pd.to_datetime(values, errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise HourlyInputError(
            "timestamp enthält mindestens einen ungültigen Datumswert."
        ) from exc
    if timestamps.isna().any():
        raise HourlyInputError("timestamp enthält fehlende Datumswerte.")
    return timestamps


def _validate_time_axis(timestamps: pd.Series) -> None:
    duplicated = timestamps.duplicated(keep=False)
    if duplicated.any():
        first_duplicate = timestamps[duplicated].iloc[0]
        raise HourlyInputError(
            f"timestamp enthält einen doppelten Zeitstempel: {first_duplicate}."
        )
    if not timestamps.is_monotonic_increasing:
        raise HourlyInputError("timestamp muss streng aufsteigend sortiert sein.")

    time_steps = timestamps.diff().iloc[1:]
    wrong_steps = time_steps != EXPECTED_TIME_STEP
    if wrong_steps.any():
        position = int(np.flatnonzero(wrong_steps.to_numpy())[0]) + 1
        previous_timestamp = timestamps.iloc[position - 1]
        current_timestamp = timestamps.iloc[position]
        raise HourlyInputError(
            "Die Zeitreihe muss lückenlos stündlich sein; zwischen "
            f"{previous_timestamp} und {current_timestamp} liegt keine Stunde."
        )


def _parse_finite_numeric_column(values: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        raise HourlyInputError(f"{column} muss numerisch sein und darf keine Bool-Werte enthalten.")
    try:
        numeric = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise HourlyInputError(f"{column} muss ausschließlich Zahlen enthalten.") from exc

    finite = np.isfinite(numeric.to_numpy())
    if not finite.all():
        row = int(np.flatnonzero(~finite)[0])
        raise HourlyInputError(
            f"{column} enthält in Zeile {row} NaN oder einen unendlichen Wert."
        )
    return numeric


def _require_nonnegative(data: pd.DataFrame, column: str) -> None:
    negative = data[column] < 0.0
    if negative.any():
        row = int(np.flatnonzero(negative.to_numpy())[0])
        value = data[column].iloc[row]
        raise HourlyInputError(
            f"{column} darf nicht negativ sein; Zeile {row} enthält {value}."
        )


__all__ = [
    "EXPECTED_TIME_STEP",
    "HOURLY_COLUMN_UNITS",
    "HourlyInputError",
    "NUMERIC_HOURLY_COLUMNS",
    "REQUIRED_HOURLY_COLUMNS",
    "TIMESTAMP_COLUMN",
    "validate_hourly_input",
]
