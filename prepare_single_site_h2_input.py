"""Erzeuge geprüfte stündliche H2-Eingabedaten für eine Koordinate.

Der Aufbereiter verbindet drei klar getrennte Datenquellen:

* PVGIS-TMY-Wetter und die bestehenden PV-/Windmodelle,
* den versionierten länderspezifischen Gewerbestrompreis oder eine eigene
  stündliche Preisreihe,
* einen versionierten länderspezifischen Netzemissionsfaktor, einen
  ausdrücklich übergebenen Faktor oder eine eigene stündliche Reihe.

Die nicht versionierten LCA-Dateien des Ammoniakmodells werden nicht benötigt.
Ohne Argumente zeigt das Skript eine Hilfe, damit es auch aus Spyder heraus
ohne ``SystemExit: 2`` geöffnet und geprüft werden kann.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from config_h2 import DEFAULT_CONFIG, ModelConfig, SiteConfig
from h2_input_data import HOURLY_COLUMN_UNITS, HourlyInputError, validate_hourly_input


REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_PRICE_FILE: Final[Path] = (
    REPOSITORY_ROOT / "input_data" / "gpp_2025_country_pages_numeric.json"
)
DEFAULT_COUNTRY_FACTOR_FILE: Final[Path] = (
    REPOSITORY_ROOT / "input_data" / "h2_country_factors.json"
)
DEFAULT_COUNTRY_SHAPEFILE: Final[Path] = (
    REPOSITORY_ROOT
    / "input_data"
    / "ne_110m_admin_0_countries"
    / "ne_110m_admin_0_countries.shp"
)
WEATHER_COLUMNS: Final[tuple[str, ...]] = (
    "temp_air",
    "ghi",
    "dni",
    "dhi",
    "wind_speed",
    "pressure",
)
PVGIS_SOURCE_URL: Final[str] = "https://re.jrc.ec.europa.eu/pvg_tools/en/"
NATURAL_EARTH_SOURCE_URL: Final[str] = "https://www.naturalearthdata.com/"


class SiteDataPreparationError(RuntimeError):
    """Verständlicher Fehler bei der standortbezogenen Datenaufbereitung."""


@dataclass(frozen=True, slots=True)
class WeatherDataset:
    """Wetterdaten und die tatsächlich von der Quelle verwendete Koordinate."""

    data: pd.DataFrame
    used_latitude: float
    used_longitude: float
    elevation_m: float | None
    source: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CountryInfo:
    """Ergebnis der geografischen Länderzuordnung."""

    iso2: str
    name: str
    source: str


@dataclass(frozen=True, slots=True)
class PreparationArtifacts:
    """Pfade und Daten einer erfolgreich abgeschlossenen Aufbereitung."""

    input_path: Path
    metadata_path: Path
    data: pd.DataFrame
    metadata: Mapping[str, object]


WeatherProvider = Callable[..., WeatherDataset]
ProfileGenerator = Callable[
    [pd.DataFrame, float, float, int], tuple[Sequence[float], Sequence[float]]
]
CountryResolver = Callable[[float, float], CountryInfo]


def prepare_single_site_h2_input(
    latitude: float,
    longitude: float,
    output_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    country_code: str | None = None,
    weather_file: str | Path | None = None,
    electricity_price_eur_per_mwh: float | None = None,
    electricity_price_file: str | Path | None = None,
    grid_emission_factor_kg_per_mwh: float | None = None,
    grid_emission_file: str | Path | None = None,
    grid_emission_source: str | None = None,
    spatial_resolution_deg: float | None = None,
    h2_demand_kg_per_day: float | None = None,
    overwrite: bool = False,
    config: ModelConfig = DEFAULT_CONFIG,
    expected_hours: int | None = None,
    weather_provider: WeatherProvider | None = None,
    profile_generator: ProfileGenerator | None = None,
    country_resolver: CountryResolver | None = None,
) -> PreparationArtifacts:
    """Bereite einen standardisierten Stunden-Datensatz für einen Standort auf.

    Die optionalen Provider dienen der klaren Trennung von Netzwerkzugriff,
    Physik und Validierung. Dadurch sind die fachlichen Grenzfälle vollständig
    ohne Internetzugriff testbar.
    """

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite muss ein boolescher Wert sein.")
    config.validate()
    site = SiteConfig(latitude=float(latitude), longitude=float(longitude))
    site.validate()
    if spatial_resolution_deg is None:
        model_latitude = site.latitude
        model_longitude = site.longitude
        location_mapping_mode = "exact_requested_coordinate"
    else:
        model_latitude, model_longitude = snap_to_grid_cell_center(
            site.latitude,
            site.longitude,
            spatial_resolution_deg,
        )
        location_mapping_mode = "nearest_regular_grid_cell_center"

    number_of_hours = (
        int(config.study.number_of_time_steps.value)
        if expected_hours is None
        else _positive_integer(expected_hours, "expected_hours")
    )
    profile_year = int(config.study.profile_calendar_year.value)
    target_index = pd.date_range(
        f"{profile_year}-01-01 00:00", periods=number_of_hours, freq="h", tz="UTC"
    )

    output = Path(output_path).expanduser().resolve()
    metadata_output = (
        output.with_name(f"{output.stem}_metadata.json")
        if metadata_path is None
        else Path(metadata_path).expanduser().resolve()
    )
    source_paths = [
        Path(item).expanduser().resolve()
        for item in (weather_file, electricity_price_file, grid_emission_file)
        if item is not None
    ]
    _prepare_output_paths(
        output,
        metadata_output,
        source_paths=source_paths,
        overwrite=overwrite,
    )

    if weather_file is not None:
        weather = load_weather_csv(
            weather_file,
            requested_latitude=model_latitude,
            requested_longitude=model_longitude,
        )
    else:
        provider = weather_provider or fetch_pvgis_tmy
        weather = provider(
            latitude=model_latitude,
            longitude=model_longitude,
            start_year=int(config.study.weather_start_year.value),
            end_year=int(config.study.weather_end_year.value),
            profile_year=profile_year,
        )

    normalized_weather, weather_time_metadata = _normalize_weather_data(
        weather.data,
        target_index=target_index,
        expected_hours=number_of_hours,
    )
    _validate_coordinate(weather.used_latitude, weather.used_longitude)

    generator = profile_generator or generate_renewable_profiles
    pv_raw, wind_raw = generator(
        normalized_weather,
        weather.used_latitude,
        weather.used_longitude,
        profile_year,
    )
    pv_profile = _validated_capacity_factor(
        pv_raw, "pv_capacity_factor", number_of_hours
    )
    wind_profile = _validated_capacity_factor(
        wind_raw, "wind_capacity_factor", number_of_hours
    )

    if country_code is None:
        resolver = country_resolver or resolve_country_from_natural_earth
        country = resolver(model_latitude, model_longitude)
    else:
        country = country_info_from_code(country_code)

    prices, price_metadata = _build_electricity_prices(
        target_index,
        country=country,
        constant_value=electricity_price_eur_per_mwh,
        series_file=electricity_price_file,
    )
    emissions, emission_metadata = _build_grid_emissions(
        target_index,
        country=country,
        constant_value=grid_emission_factor_kg_per_mwh,
        series_file=grid_emission_file,
        source_description=grid_emission_source,
    )

    demand_per_day = (
        float(config.study.h2_demand_kg_per_day.value)
        if h2_demand_kg_per_day is None
        else _positive_finite(h2_demand_kg_per_day, "h2_demand_kg_per_day")
    )
    raw_input = pd.DataFrame(
        {
            "timestamp": target_index,
            "pv_capacity_factor": pv_profile,
            "wind_capacity_factor": wind_profile,
            "electricity_price": prices,
            "grid_emission_factor": emissions,
            "h2_demand": np.full(number_of_hours, demand_per_day / 24.0),
        }
    )
    try:
        validated = validate_hourly_input(raw_input, expected_hours=number_of_hours)
    except HourlyInputError as exc:
        raise SiteDataPreparationError(
            f"Der erzeugte H2-Eingabedatensatz ist ungültig: {exc}"
        ) from exc

    distance_km = haversine_distance_km(
        site.latitude,
        site.longitude,
        weather.used_latitude,
        weather.used_longitude,
    )
    metadata: dict[str, object] = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "site": {
            "requested_latitude": site.latitude,
            "requested_longitude": site.longitude,
            "model_location_latitude": model_latitude,
            "model_location_longitude": model_longitude,
            "location_mapping_mode": location_mapping_mode,
            "spatial_resolution_deg": spatial_resolution_deg,
            "distance_requested_to_model_location_km": haversine_distance_km(
                site.latitude,
                site.longitude,
                model_latitude,
                model_longitude,
            ),
            "used_weather_latitude": weather.used_latitude,
            "used_weather_longitude": weather.used_longitude,
            "distance_requested_to_weather_point_km": distance_km,
            "elevation_m": weather.elevation_m,
            "country_code": country.iso2,
            "country_name": country.name,
            "country_assignment_source": country.source,
        },
        "time_axis": {
            "profile_calendar_year": profile_year,
            "number_of_hours": number_of_hours,
            "time_zone": "UTC",
            "frequency": "1h",
            **weather_time_metadata,
        },
        "sources": {
            "weather": _json_safe(dict(weather.source)),
            "renewable_profiles": {
                "implementation": "calculate_renewable_yield.py",
                "pv_model": "pv_profile_generator_tmy",
                "wind_model": "wind_profile_generator_tmy, V90/2000, 80 m",
                "spatial_scope": "model_coordinate_or_selected_grid_cell",
            },
            "electricity_price": price_metadata,
            "grid_emission_factor": emission_metadata,
            "h2_demand": {
                "value_kg_h2_per_day": demand_per_day,
                "source": config.study.h2_demand_kg_per_day.source,
                "reference_year": config.study.h2_demand_kg_per_day.reference_year,
                "spatial_scope": "scenario_assumption_not_location_dataset",
            },
        },
        "column_units": dict(HOURLY_COLUMN_UNITS),
        "output": {
            "input_file": str(output),
            "metadata_file": str(metadata_output),
        },
        "limitations": [
            "Der automatische Preis ist ein länderspezifischer Gewerbestrompreis, der konstant auf Stunden abgebildet wird; er ist kein Day-Ahead-Preis.",
            "Der Netzemissionsfaktor wird nicht aus den fehlenden LCA-Dateien des Ammoniakmodells übernommen; verfügbare Länderwerte stammen aus der versionierten H2-Länderdatei.",
            "PVGIS verwendet intern gerasterte Wetterdaten; als verwendete Koordinate wird die von der API gemeldete Koordinate dokumentiert.",
        ],
    }

    _write_csv_atomic(validated, output)
    metadata["output"]["input_sha256"] = _sha256(output)  # type: ignore[index]
    _write_json_atomic(metadata, metadata_output)
    return PreparationArtifacts(output, metadata_output, validated, metadata)


def fetch_pvgis_tmy(
    *,
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
    profile_year: int,
) -> WeatherDataset:
    """Lade ein PVGIS-TMY und unterstütze aktuelle sowie ältere pvlib-Rückgaben."""

    _validate_coordinate(latitude, longitude)
    try:
        import pvlib

        response = pvlib.iotools.get_pvgis_tmy(
            latitude,
            longitude,
            startyear=start_year,
            endyear=end_year,
            map_variables=True,
            coerce_year=profile_year,
            timeout=60,
        )
    except Exception as exc:
        raise SiteDataPreparationError(
            "PVGIS konnte für die Koordinate "
            f"({latitude}, {longitude}) keine TMY-Wetterdaten liefern. "
            "Prüfe Internetzugang und PVGIS-Datenabdeckung."
        ) from exc

    if not isinstance(response, (tuple, list)) or len(response) not in (2, 4):
        raise SiteDataPreparationError(
            f"Unerwartetes PVGIS-Rückgabeformat: {type(response).__name__}."
        )
    if len(response) == 2:
        data, metadata = response
    else:
        data, months_selected, inputs, legacy_metadata = response
        metadata = dict(legacy_metadata or {})
        metadata["inputs"] = inputs
        metadata["months_selected"] = months_selected
    if not isinstance(data, pd.DataFrame):
        raise SiteDataPreparationError("PVGIS hat keine tabellarischen Wetterdaten geliefert.")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    location = metadata.get("inputs", {})
    location = location.get("location", {}) if isinstance(location, Mapping) else {}
    used_latitude = _mapping_float(location, "latitude", latitude)
    used_longitude = _mapping_float(location, "longitude", longitude)
    elevation = _mapping_optional_float(location, "elevation")
    months = metadata.get("months_selected")
    source = {
        "name": "PVGIS Typical Meteorological Year (TMY)",
        "url": PVGIS_SOURCE_URL,
        "weather_selection_years": [start_year, end_year],
        "profile_calendar_year": profile_year,
        "months_selected": _json_safe(months),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return WeatherDataset(data, used_latitude, used_longitude, elevation, source)


def load_weather_csv(
    path: str | Path,
    *,
    requested_latitude: float,
    requested_longitude: float,
) -> WeatherDataset:
    """Lade eine lokale Wetter-CSV mit Zeitstempel und PVGIS-kompatiblen Feldern."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Wetterdatei nicht gefunden: {source_path}")
    try:
        data = pd.read_csv(source_path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise SiteDataPreparationError(
            f"Wetterdatei konnte nicht gelesen werden: {source_path}"
        ) from exc
    return WeatherDataset(
        data=data,
        used_latitude=requested_latitude,
        used_longitude=requested_longitude,
        elevation_m=None,
        source={
            "name": "Lokale Wetter-CSV",
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
    )


def generate_renewable_profiles(
    weather_data: pd.DataFrame,
    latitude: float,
    longitude: float,
    profile_year: int,
) -> tuple[Sequence[float], Sequence[float]]:
    """Nutze die geprüften Erzeugungsfunktionen des ursprünglichen Modells."""

    try:
        from calculate_renewable_yield import (
            pv_profile_generator_tmy,
            wind_profile_generator_tmy,
        )

        model_weather = weather_data.copy()
        model_weather.index = model_weather.index.tz_localize(None)
        pv_profile = pv_profile_generator_tmy(model_weather, latitude, longitude)
        wind_profile = wind_profile_generator_tmy(
            model_weather,
            turbine_spec="V90/2000",
            hub_height=80,
            offshore=False,
            assessment_year=profile_year,
        )
    except Exception as exc:
        raise SiteDataPreparationError(
            "PV- oder Windprofil konnte aus den Wetterdaten nicht berechnet werden."
        ) from exc
    return pv_profile, wind_profile


def resolve_country_from_natural_earth(
    latitude: float,
    longitude: float,
    shapefile_path: str | Path = DEFAULT_COUNTRY_SHAPEFILE,
) -> CountryInfo:
    """Bestimme ein Land ohne zusätzliche GIS-Bibliothek aus Natural Earth."""

    _validate_coordinate(latitude, longitude)
    shp_path = Path(shapefile_path).expanduser().resolve()
    dbf_path = shp_path.with_suffix(".dbf")
    if not shp_path.is_file() or not dbf_path.is_file():
        raise SiteDataPreparationError(
            "Natural-Earth-Ländergrenzen fehlen. Übergib --country-code ausdrücklich."
        )
    attributes = _read_dbf_records(dbf_path)
    for record_index, bbox, rings in _iter_shapefile_polygons(shp_path):
        if record_index >= len(attributes):
            break
        if not _bbox_contains(bbox, longitude, latitude):
            continue
        if _point_in_polygon(longitude, latitude, rings):
            record = attributes[record_index]
            iso2 = _iso2_from_natural_earth_record(record)
            name = record.get("ADMIN") or record.get("NAME_EN") or iso2
            return CountryInfo(
                iso2=iso2,
                name=name,
                source=f"Natural Earth 1:110m ({NATURAL_EARTH_SOURCE_URL})",
            )
    raise SiteDataPreparationError(
        "Die Koordinate konnte keiner Landfläche zugeordnet werden. "
        "Für Küsten-, Insel- oder Offshore-Punkte --country-code angeben."
    )


def country_info_from_code(country_code: str) -> CountryInfo:
    """Normalisiere einen ISO-Alpha-2- oder Alpha-3-Code auf Alpha-2."""

    if not isinstance(country_code, str) or not country_code.strip():
        raise ValueError("country_code darf nicht leer sein.")
    code = country_code.strip().upper()
    if code in {"XK", "XKX"}:
        return CountryInfo("XK", "Kosovo", "Explizite Benutzereingabe")
    try:
        import pycountry

        country = (
            pycountry.countries.get(alpha_2=code)
            if len(code) == 2
            else pycountry.countries.get(alpha_3=code)
        )
    except Exception as exc:
        raise SiteDataPreparationError("pycountry ist für die ISO-Prüfung erforderlich.") from exc
    if country is None:
        raise ValueError(f"Unbekannter ISO-Ländercode: {country_code}")
    return CountryInfo(country.alpha_2, country.name, "Explizite Benutzereingabe")


def haversine_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Berechne die Großkreisdistanz zweier WGS84-Koordinaten."""

    for latitude, longitude in (
        (latitude_1, longitude_1),
        (latitude_2, longitude_2),
    ):
        _validate_coordinate(latitude, longitude)
    radius_km = 6371.0088
    lat_1, lat_2 = math.radians(latitude_1), math.radians(latitude_2)
    delta_lat = lat_2 - lat_1
    delta_lon = math.radians(longitude_2 - longitude_1)
    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2.0) ** 2
    )
    return radius_km * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def snap_to_grid_cell_center(
    latitude: float,
    longitude: float,
    resolution_deg: float = 1.0,
) -> tuple[float, float]:
    """Ordne eine Koordinate dem Mittelpunkt eines regulären Gradrasters zu.

    Das 1-Grad-Raster des ursprünglichen Ammoniakmodells besitzt Mittelpunkte
    bei ``..., -21.5, -20.5, ...``. Die Funktion bildet eine frei gewählte
    Koordinate reproduzierbar auf dasselbe Rasterprinzip ab.
    """

    _validate_coordinate(latitude, longitude)
    resolution = _finite_number(resolution_deg, "spatial_resolution_deg")
    if not 0.0 < resolution <= 180.0:
        raise ValueError("spatial_resolution_deg muss in (0, 180] liegen.")
    snapped_latitude = (
        math.floor(float(latitude) / resolution) * resolution + resolution / 2.0
    )
    snapped_longitude = (
        math.floor(float(longitude) / resolution) * resolution + resolution / 2.0
    )
    snapped_latitude = min(90.0 - resolution / 2.0, max(-90.0 + resolution / 2.0, snapped_latitude))
    snapped_longitude = min(180.0 - resolution / 2.0, max(-180.0 + resolution / 2.0, snapped_longitude))
    _validate_coordinate(snapped_latitude, snapped_longitude)
    return snapped_latitude, snapped_longitude


def _normalize_weather_data(
    data: pd.DataFrame,
    *,
    target_index: pd.DatetimeIndex,
    expected_hours: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise SiteDataPreparationError("Die Wetterzeitreihe ist leer oder nicht tabellarisch.")
    missing = [column for column in WEATHER_COLUMNS if column not in data.columns]
    if missing:
        raise SiteDataPreparationError(
            "Wetterspalten fehlen: " + ", ".join(missing) + "."
        )
    if "timestamp" in data.columns:
        timestamp_values = data["timestamp"]
    elif isinstance(data.index, pd.DatetimeIndex):
        timestamp_values = pd.Series(data.index, index=data.index)
    else:
        raise SiteDataPreparationError(
            "Wetterdaten benötigen eine timestamp-Spalte oder einen DatetimeIndex."
        )
    try:
        source_index = pd.DatetimeIndex(pd.to_datetime(timestamp_values, utc=True, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise SiteDataPreparationError("Die Wetterzeitstempel sind ungültig.") from exc
    if source_index.has_duplicates:
        raise SiteDataPreparationError("Die Wetterzeitreihe enthält doppelte Zeitstempel.")
    if not source_index.is_monotonic_increasing:
        raise SiteDataPreparationError("Die Wetterzeitreihe muss aufsteigend sortiert sein.")
    if len(data) != expected_hours:
        raise SiteDataPreparationError(
            f"Erwartet wurden {expected_hours} Wetterstunden, vorhanden sind {len(data)}."
        )
    if len(source_index) > 1 and not (source_index[1:] - source_index[:-1] == pd.Timedelta(hours=1)).all():
        raise SiteDataPreparationError("Die Wetterzeitreihe muss lückenlos stündlich sein.")

    normalized = pd.DataFrame(index=target_index)
    for column in WEATHER_COLUMNS:
        try:
            values = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise SiteDataPreparationError(f"Wetterspalte {column} muss numerisch sein.") from exc
        if not np.isfinite(values).all():
            raise SiteDataPreparationError(f"Wetterspalte {column} enthält fehlende Werte.")
        normalized[column] = values
    corrected_small_negative_values: dict[str, int] = {}
    for column in ("ghi", "dni", "dhi", "wind_speed"):
        negative = normalized[column] < 0.0
        if (normalized[column] < -1.0).any():
            raise SiteDataPreparationError(
                f"Wetterspalte {column} enthält einen Wert kleiner als -1,0 und ist ungültig."
            )
        corrected_small_negative_values[column] = int(negative.sum())
        normalized.loc[negative, column] = 0.0
    if (normalized["pressure"] <= 0.0).any():
        raise SiteDataPreparationError("Wetterspalte pressure muss positiv sein.")
    time_metadata = {
        "weather_source_start_utc": source_index[0].isoformat(),
        "weather_source_end_utc": source_index[-1].isoformat(),
        "weather_index_alignment": (
            "unchanged"
            if source_index.equals(target_index)
            else "positionally_mapped_to_profile_calendar_year"
        ),
        "negative_weather_values_between_minus_one_and_zero_set_to_zero": corrected_small_negative_values,
    }
    return normalized, time_metadata


def _validated_capacity_factor(
    values: Sequence[float],
    name: str,
    expected_hours: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) != expected_hours:
        raise SiteDataPreparationError(
            f"{name} enthält {len(array)} statt {expected_hours} Stunden."
        )
    if not np.isfinite(array).all():
        raise SiteDataPreparationError(f"{name} enthält NaN oder unendliche Werte.")
    tolerance = 1e-9
    if (array < -tolerance).any() or (array > 1.0 + tolerance).any():
        raise SiteDataPreparationError(f"{name} muss vollständig zwischen 0 und 1 liegen.")
    return np.clip(array, 0.0, 1.0)


def _build_electricity_prices(
    target_index: pd.DatetimeIndex,
    *,
    country: CountryInfo,
    constant_value: float | None,
    series_file: str | Path | None,
) -> tuple[np.ndarray, dict[str, object]]:
    if constant_value is not None and series_file is not None:
        raise ValueError(
            "Strompreis entweder als Konstante oder als Datei übergeben, nicht beides."
        )
    if series_file is not None:
        values, metadata = _load_hourly_series(
            series_file,
            value_column="electricity_price",
            target_index=target_index,
            allow_negative=True,
        )
        metadata["type"] = "user_hourly_series"
        metadata["spatial_scope"] = "user_declared"
        return values, metadata
    if constant_value is not None:
        value = _finite_number(constant_value, "electricity_price_eur_per_mwh")
        return np.full(len(target_index), value), {
            "type": "user_constant",
            "value_eur_per_mwh": value,
            "spatial_scope": "user_declared",
            "temporal_mapping": "constant_for_all_hours",
        }

    if not DEFAULT_PRICE_FILE.is_file():
        raise SiteDataPreparationError(
            f"Versionierte Strompreisdatei fehlt: {DEFAULT_PRICE_FILE}"
        )
    try:
        price_data = json.loads(DEFAULT_PRICE_FILE.read_text(encoding="utf-8"))
        numeric_map = price_data["numeric_map"]
        price_eur_per_kwh = float(numeric_map[country.iso2])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SiteDataPreparationError(
            f"Kein belastbarer automatischer Strompreis für {country.iso2}. "
            "Nutze --electricity-price oder --electricity-price-file."
        ) from exc
    if not math.isfinite(price_eur_per_kwh) or price_eur_per_kwh <= 0.0:
        raise SiteDataPreparationError(
            f"Ungültiger automatischer Strompreis für {country.iso2}: {price_eur_per_kwh}."
        )
    details = price_data.get("results", {}).get(country.iso2, {})
    price_eur_per_mwh = price_eur_per_kwh * 1000.0
    return np.full(len(target_index), price_eur_per_mwh), {
        "type": "country_business_retail_price_proxy",
        "country_code": country.iso2,
        "value_eur_per_kwh_in_source": price_eur_per_kwh,
        "value_eur_per_mwh_in_model": price_eur_per_mwh,
        "source_url": details.get("url") if isinstance(details, Mapping) else None,
        "source_file": str(DEFAULT_PRICE_FILE),
        "source_file_sha256": _sha256(DEFAULT_PRICE_FILE),
        "usd_to_eur_used_by_source_file": price_data.get("USD_TO_EUR_used"),
        "spatial_scope": "country",
        "temporal_mapping": "country_value_repeated_for_all_hours",
        "suitability": "reference_cost_proxy_not_day_ahead_price",
    }


def _build_grid_emissions(
    target_index: pd.DatetimeIndex,
    *,
    country: CountryInfo,
    constant_value: float | None,
    series_file: str | Path | None,
    source_description: str | None,
) -> tuple[np.ndarray, dict[str, object]]:
    if constant_value is not None and series_file is not None:
        raise ValueError(
            "Netzemissionen entweder als Konstante oder Datei übergeben, nicht beides."
        )
    if series_file is not None:
        values, metadata = _load_hourly_series(
            series_file,
            value_column="grid_emission_factor",
            target_index=target_index,
            allow_negative=False,
        )
        metadata["type"] = "user_hourly_series"
        metadata["spatial_scope"] = "user_declared"
        if source_description:
            metadata["source_description"] = source_description
        return values, metadata
    if constant_value is None:
        if not DEFAULT_COUNTRY_FACTOR_FILE.is_file():
            raise SiteDataPreparationError(
                "Genau eine Netzemissionsquelle ist erforderlich; die versionierte "
                f"Länderdatei fehlt: {DEFAULT_COUNTRY_FACTOR_FILE}"
            )
        try:
            factor_data = json.loads(
                DEFAULT_COUNTRY_FACTOR_FILE.read_text(encoding="utf-8")
            )
            entry = factor_data["countries"][country.iso2]
            value = float(entry["grid_emission_factor_kg_co2e_per_mwh"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SiteDataPreparationError(
                "Genau eine Netzemissionsquelle ist erforderlich. Für "
                f"{country.iso2} ist kein versionierter Länderwert vorhanden; nutze "
                "--grid-emission-factor oder --grid-emission-file."
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise SiteDataPreparationError(
                f"Ungültiger Netzemissionsfaktor für {country.iso2}: {value}."
            )
        return np.full(len(target_index), value), {
            "type": "country_annual_average",
            "country_code": country.iso2,
            "value_kg_co2e_per_mwh": value,
            "data_year": entry.get("data_year"),
            "source_name": entry.get("source_name"),
            "source_url": entry.get("source_url"),
            "source_file": str(DEFAULT_COUNTRY_FACTOR_FILE),
            "source_file_sha256": _sha256(DEFAULT_COUNTRY_FACTOR_FILE),
            "spatial_scope": "country_national_grid",
            "temporal_mapping": "annual_country_value_repeated_for_all_hours",
        }

    value = _finite_number(constant_value, "grid_emission_factor_kg_per_mwh")
    if value < 0.0:
        raise ValueError("grid_emission_factor_kg_per_mwh darf nicht negativ sein.")
    return np.full(len(target_index), value), {
        "type": "user_constant",
        "value_kg_co2e_per_mwh": value,
        "source_description": source_description or "Vom Nutzer übergebene Modellannahme",
        "spatial_scope": "user_declared",
        "temporal_mapping": "constant_for_all_hours",
    }


def _load_hourly_series(
    path: str | Path,
    *,
    value_column: str,
    target_index: pd.DatetimeIndex,
    allow_negative: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Zeitreihendatei nicht gefunden: {source_path}")
    try:
        data = pd.read_csv(source_path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise SiteDataPreparationError(
            f"Zeitreihendatei konnte nicht gelesen werden: {source_path}"
        ) from exc
    missing = [column for column in ("timestamp", value_column) if column not in data.columns]
    if missing:
        raise SiteDataPreparationError(
            f"In {source_path.name} fehlen: {', '.join(missing)}."
        )
    try:
        source_index = pd.DatetimeIndex(pd.to_datetime(data["timestamp"], utc=True, errors="raise"))
        values = pd.to_numeric(data[value_column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise SiteDataPreparationError(
            f"Zeitstempel oder {value_column} in {source_path.name} sind ungültig."
        ) from exc
    if len(data) != len(target_index):
        raise SiteDataPreparationError(
            f"{source_path.name} enthält {len(data)} statt {len(target_index)} Stunden."
        )
    if source_index.has_duplicates or not source_index.is_monotonic_increasing:
        raise SiteDataPreparationError(
            f"Zeitachse in {source_path.name} ist doppelt oder unsortiert."
        )
    if len(source_index) > 1 and not (source_index[1:] - source_index[:-1] == pd.Timedelta(hours=1)).all():
        raise SiteDataPreparationError(
            f"Zeitachse in {source_path.name} ist nicht lückenlos stündlich."
        )
    if not np.isfinite(values).all():
        raise SiteDataPreparationError(f"{value_column} enthält fehlende Werte.")
    if not allow_negative and (values < 0.0).any():
        raise SiteDataPreparationError(f"{value_column} darf nicht negativ sein.")
    alignment = "exact_timestamp_match" if source_index.equals(target_index) else "positionally_mapped_to_profile_calendar_year"
    return values, {
        "source_file": str(source_path),
        "source_file_sha256": _sha256(source_path),
        "source_start_utc": source_index[0].isoformat(),
        "source_end_utc": source_index[-1].isoformat(),
        "temporal_mapping": alignment,
    }


def _read_dbf_records(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if len(raw) < 33:
        raise SiteDataPreparationError(f"Ungültige DBF-Datei: {path}")
    record_count = struct.unpack_from("<I", raw, 4)[0]
    header_length = struct.unpack_from("<H", raw, 8)[0]
    record_length = struct.unpack_from("<H", raw, 10)[0]
    fields: list[tuple[str, int]] = []
    offset = 32
    while offset + 32 <= header_length and raw[offset] != 0x0D:
        name = raw[offset : offset + 11].split(b"\x00", 1)[0].decode("ascii")
        length = raw[offset + 16]
        fields.append((name, length))
        offset += 32
    records: list[dict[str, str]] = []
    position = header_length
    for _ in range(record_count):
        record_raw = raw[position : position + record_length]
        position += record_length
        if len(record_raw) != record_length:
            raise SiteDataPreparationError(f"Abgeschnittene DBF-Datei: {path}")
        field_position = 1
        record: dict[str, str] = {}
        for name, length in fields:
            value_raw = record_raw[field_position : field_position + length]
            field_position += length
            record[name] = value_raw.decode("utf-8", errors="replace").strip(" \t\r\n\x00")
        records.append(record)
    return records


def _iter_shapefile_polygons(
    path: Path,
) -> Sequence[tuple[int, tuple[float, float, float, float], list[list[tuple[float, float]]]]]:
    polygons: list[
        tuple[int, tuple[float, float, float, float], list[list[tuple[float, float]]]]
    ] = []
    with path.open("rb") as handle:
        header = handle.read(100)
        if len(header) != 100 or struct.unpack_from(">I", header, 0)[0] != 9994:
            raise SiteDataPreparationError(f"Ungültige Shapefile-Datei: {path}")
        record_index = 0
        while True:
            record_header = handle.read(8)
            if not record_header:
                break
            if len(record_header) != 8:
                raise SiteDataPreparationError(f"Abgeschnittene Shapefile-Datei: {path}")
            _, content_words = struct.unpack(">II", record_header)
            content = handle.read(content_words * 2)
            if len(content) != content_words * 2:
                raise SiteDataPreparationError(f"Abgeschnittene Shapefile-Datei: {path}")
            shape_type = struct.unpack_from("<I", content, 0)[0]
            if shape_type in (5, 15, 25):
                bbox = struct.unpack_from("<4d", content, 4)
                number_of_parts, number_of_points = struct.unpack_from("<2I", content, 36)
                parts = list(struct.unpack_from(f"<{number_of_parts}I", content, 44))
                point_offset = 44 + 4 * number_of_parts
                points = [
                    struct.unpack_from("<2d", content, point_offset + 16 * i)
                    for i in range(number_of_points)
                ]
                rings: list[list[tuple[float, float]]] = []
                for part_number, start in enumerate(parts):
                    end = parts[part_number + 1] if part_number + 1 < len(parts) else len(points)
                    rings.append(points[start:end])
                polygons.append((record_index, bbox, rings))
            record_index += 1
    return polygons


def _bbox_contains(
    bbox: tuple[float, float, float, float], longitude: float, latitude: float
) -> bool:
    xmin, ymin, xmax, ymax = bbox
    return xmin <= longitude <= xmax and ymin <= latitude <= ymax


def _point_in_polygon(
    longitude: float,
    latitude: float,
    rings: Sequence[Sequence[tuple[float, float]]],
) -> bool:
    inside = False
    for ring in rings:
        if len(ring) < 3:
            continue
        ring_inside = False
        previous_x, previous_y = ring[-1]
        for current_x, current_y in ring:
            if _point_on_segment(longitude, latitude, previous_x, previous_y, current_x, current_y):
                return True
            crosses = (current_y > latitude) != (previous_y > latitude)
            if crosses:
                intersection_x = (
                    (previous_x - current_x)
                    * (latitude - current_y)
                    / (previous_y - current_y)
                    + current_x
                )
                if longitude < intersection_x:
                    ring_inside = not ring_inside
            previous_x, previous_y = current_x, current_y
        if ring_inside:
            inside = not inside
    return inside


def _point_on_segment(
    x: float, y: float, x1: float, y1: float, x2: float, y2: float
) -> bool:
    tolerance = 1e-10
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > tolerance:
        return False
    return (
        min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
    )


def _iso2_from_natural_earth_record(record: Mapping[str, str]) -> str:
    for field in ("ISO_A2_EH", "ISO_A2"):
        code = record.get(field, "").strip().upper()
        if len(code) == 2 and code != "-9":
            return code
    for field in ("ISO_A3_EH", "ADM0_A3", "ISO_A3"):
        alpha3 = record.get(field, "").strip().upper()
        if alpha3 == "XKX":
            return "XK"
        if len(alpha3) == 3 and alpha3 != "-99":
            try:
                import pycountry

                country = pycountry.countries.get(alpha_3=alpha3)
            except Exception as exc:
                raise SiteDataPreparationError("pycountry ist für die ISO-Zuordnung erforderlich.") from exc
            if country is not None:
                return country.alpha_2
    raise SiteDataPreparationError(
        f"Natural-Earth-Datensatz enthält keinen nutzbaren ISO-Code für {record.get('ADMIN', 'unbekannt')}."
    )


def _prepare_output_paths(
    output: Path,
    metadata_output: Path,
    *,
    source_paths: Sequence[Path],
    overwrite: bool,
) -> None:
    if output == metadata_output:
        raise ValueError("Eingabe-CSV und Metadaten benötigen verschiedene Pfade.")
    for source_path in source_paths:
        if output == source_path or metadata_output == source_path:
            raise SiteDataPreparationError("Eine Quelldatei darf nicht überschrieben werden.")
    existing = [path for path in (output, metadata_output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Ausgabedatei existiert bereits: "
            + ", ".join(str(path) for path in existing)
            + ". Nutze --overwrite zum Ersetzen."
        )
    for path in (output, metadata_output):
        if path.exists() and not path.is_file():
            raise SiteDataPreparationError(f"Ausgabepfad ist keine Datei: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv_atomic(data: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        data.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(data: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _mapping_float(mapping: Mapping[str, object], key: str, fallback: float) -> float:
    try:
        value = float(mapping.get(key, fallback))
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def _mapping_optional_float(mapping: Mapping[str, object], key: str) -> float | None:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _validate_coordinate(latitude: float, longitude: float) -> None:
    lat = _finite_number(latitude, "latitude")
    lon = _finite_number(longitude, "longitude")
    if not -90.0 <= lat <= 90.0:
        raise ValueError("latitude muss zwischen -90 und 90 Grad liegen.")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("longitude muss zwischen -180 und 180 Grad liegen.")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} muss eine reelle Zahl sein.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} muss eine reelle Zahl sein.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} muss endlich sein.")
    return number


def _positive_finite(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} muss positiv sein.")
    return number


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} muss eine ganze Zahl sein.")
    if value <= 0:
        raise ValueError(f"{name} muss positiv sein.")
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Erzeugt für eine Koordinate eine validierte stündliche Eingabe-CSV "
            "für das H2-Optimierungsmodell."
        ),
        epilog=(
            "Beispiel (eine Zeile): python prepare_single_site_h2_input.py "
            "--latitude 50.78 --longitude 6.08 "
            "--grid-emission-factor 350 --grid-emission-source \"Quelle, Jahr\" "
            "--output input_data\\h2_aachen.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--latitude", required=True, type=float, help="Breitengrad in WGS84.")
    parser.add_argument("--longitude", required=True, type=float, help="Längengrad in WGS84.")
    parser.add_argument("--output", required=True, type=Path, help="Zielpfad der H2-Eingabe-CSV.")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optionaler JSON-Zielpfad; Standard: <output>_metadata.json.",
    )
    parser.add_argument(
        "--country-code",
        help="Optionaler ISO-Alpha-2/3-Code; sinnvoll für Offshore- oder Grenzpunkte.",
    )
    parser.add_argument(
        "--weather-file",
        type=Path,
        help="Optionale lokale Wetter-CSV statt eines PVGIS-Abrufs.",
    )
    price_group = parser.add_mutually_exclusive_group()
    price_group.add_argument(
        "--electricity-price",
        type=float,
        help="Konstanter Netzstrompreis in EUR/MWh statt der automatischen Länderzuordnung.",
    )
    price_group.add_argument(
        "--electricity-price-file",
        type=Path,
        help="CSV mit timestamp und electricity_price in EUR/MWh.",
    )
    emission_group = parser.add_mutually_exclusive_group()
    emission_group.add_argument(
        "--grid-emission-factor",
        type=float,
        help="Konstanter Netzemissionsfaktor in kg CO2e/MWh.",
    )
    emission_group.add_argument(
        "--grid-emission-file",
        type=Path,
        help="CSV mit timestamp und grid_emission_factor in kg CO2e/MWh.",
    )
    parser.add_argument(
        "--grid-emission-source",
        help="Quellenangabe für den übergebenen Netzemissionsfaktor oder die Datei.",
    )
    parser.add_argument(
        "--spatial-resolution",
        type=float,
        help=(
            "Optionale Rasterweite in Grad. Mit 1 wird die Koordinate wie im "
            "ursprünglichen Ammoniakmodell einem 1-Grad-Pixelmittelpunkt zugeordnet."
        ),
    )
    parser.add_argument(
        "--h2-demand-kg-per-day",
        type=float,
        help="Optionale Tagesnachfrage; Standardwert stammt aus config_h2.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ersetzt vorhandene Ausgabe-CSV und Metadaten.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return 0
    parsed = parser.parse_args(arguments)
    try:
        artifacts = prepare_single_site_h2_input(
            parsed.latitude,
            parsed.longitude,
            parsed.output,
            metadata_path=parsed.metadata_output,
            country_code=parsed.country_code,
            weather_file=parsed.weather_file,
            electricity_price_eur_per_mwh=parsed.electricity_price,
            electricity_price_file=parsed.electricity_price_file,
            grid_emission_factor_kg_per_mwh=parsed.grid_emission_factor,
            grid_emission_file=parsed.grid_emission_file,
            grid_emission_source=parsed.grid_emission_source,
            spatial_resolution_deg=parsed.spatial_resolution,
            h2_demand_kg_per_day=parsed.h2_demand_kg_per_day,
            overwrite=parsed.overwrite,
        )
    except (FileNotFoundError, FileExistsError, TypeError, ValueError, SiteDataPreparationError) as exc:
        print(f"Fehler bei der Standortdaten-Aufbereitung: {exc}", file=sys.stderr)
        return 1
    print(f"H2-Eingabedaten geschrieben: {artifacts.input_path}")
    print(f"Metadaten geschrieben: {artifacts.metadata_path}")
    print(f"Stunden: {len(artifacts.data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CountryInfo",
    "PreparationArtifacts",
    "SiteDataPreparationError",
    "WeatherDataset",
    "build_argument_parser",
    "country_info_from_code",
    "fetch_pvgis_tmy",
    "generate_renewable_profiles",
    "haversine_distance_km",
    "load_weather_csv",
    "main",
    "prepare_single_site_h2_input",
    "resolve_country_from_natural_earth",
    "snap_to_grid_cell_center",
]
