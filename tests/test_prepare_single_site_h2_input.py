"""Tests für die standortbezogene H2-Eingabedaten-Aufbereitung."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prepare_single_site_h2_input import (
    CountryInfo,
    SiteDataPreparationError,
    WeatherDataset,
    haversine_distance_km,
    main,
    prepare_single_site_h2_input,
    resolve_country_from_natural_earth,
    snap_to_grid_cell_center,
)


def test_namibia_coordinate_maps_to_original_one_degree_pixel() -> None:
    latitude, longitude = snap_to_grid_cell_center(-21.0800, 14.1610, 1.0)
    assert latitude == pytest.approx(-21.5)
    assert longitude == pytest.approx(14.5)


def fake_weather(
    *,
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
    profile_year: int,
    hours: int = 24,
) -> WeatherDataset:
    del start_year, end_year, profile_year
    index = pd.date_range("2015-01-01", periods=hours, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "temp_air": np.full(hours, 15.0),
            "ghi": np.maximum(0.0, np.sin((np.arange(hours) - 6) * np.pi / 12.0)) * 700.0,
            "dni": np.maximum(0.0, np.sin((np.arange(hours) - 6) * np.pi / 12.0)) * 500.0,
            "dhi": np.maximum(0.0, np.sin((np.arange(hours) - 6) * np.pi / 12.0)) * 200.0,
            "wind_speed": np.full(hours, 6.0),
            "pressure": np.full(hours, 101_325.0),
        },
        index=index,
    )
    return WeatherDataset(
        data=data,
        used_latitude=latitude + 0.05,
        used_longitude=longitude - 0.05,
        elevation_m=120.0,
        source={"name": "offline-test-weather"},
    )


def fake_profiles(
    weather: pd.DataFrame,
    latitude: float,
    longitude: float,
    profile_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    del latitude, profile_year
    hours = np.arange(len(weather))
    pv = np.maximum(0.0, np.sin((hours - 6) * np.pi / 12.0))
    wind = np.full(len(weather), 0.3 + abs(longitude) / 1000.0)
    return pv, wind


def country_resolver(latitude: float, longitude: float) -> CountryInfo:
    del latitude
    if longitude < 0:
        return CountryInfo("ES", "Spain", "test polygons")
    return CountryInfo("DE", "Germany", "test polygons")


def run_preparation(
    tmp_path: Path,
    *,
    latitude: float = 50.78,
    longitude: float = 6.08,
    weather_provider=fake_weather,
    profile_generator=fake_profiles,
    **kwargs: object,
):
    return prepare_single_site_h2_input(
        latitude,
        longitude,
        tmp_path / "input.csv",
        grid_emission_factor_kg_per_mwh=350.0,
        grid_emission_source="Testquelle 2025",
        expected_hours=24,
        weather_provider=weather_provider,
        profile_generator=profile_generator,
        country_resolver=country_resolver,
        **kwargs,
    )


def test_first_coordinate_creates_complete_validated_files(tmp_path: Path) -> None:
    artifacts = run_preparation(tmp_path)

    assert artifacts.input_path.is_file()
    assert artifacts.metadata_path.is_file()
    assert len(artifacts.data) == 24
    assert tuple(artifacts.data.columns) == (
        "timestamp",
        "pv_capacity_factor",
        "wind_capacity_factor",
        "electricity_price",
        "grid_emission_factor",
        "h2_demand",
    )
    assert artifacts.data["h2_demand"].sum() == pytest.approx(10_000.0)
    assert artifacts.data["electricity_price"].iloc[0] == pytest.approx(248.0)
    assert str(artifacts.data["timestamp"].dtype) == "datetime64[us, UTC]"

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["site"]["country_code"] == "DE"
    assert metadata["site"]["used_weather_latitude"] == pytest.approx(50.83)
    assert metadata["site"]["distance_requested_to_weather_point_km"] > 0.0
    assert metadata["sources"]["grid_emission_factor"]["source_description"] == "Testquelle 2025"
    assert len(metadata["output"]["input_sha256"]) == 64


def test_second_coordinate_uses_same_interface_and_different_profile(tmp_path: Path) -> None:
    first = run_preparation(tmp_path / "first", latitude=50.78, longitude=6.08)
    second = run_preparation(tmp_path / "second", latitude=40.42, longitude=-3.70)

    assert list(first.data.columns) == list(second.data.columns)
    assert not np.allclose(
        first.data["wind_capacity_factor"], second.data["wind_capacity_factor"]
    )
    assert second.metadata["site"]["country_code"] == "ES"


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(90.01, 0.0), (-90.01, 0.0), (0.0, 180.01), (0.0, -180.01)],
)
def test_invalid_coordinate_is_rejected(
    tmp_path: Path, latitude: float, longitude: float
) -> None:
    with pytest.raises(ValueError):
        run_preparation(tmp_path, latitude=latitude, longitude=longitude)


def test_incomplete_weather_series_is_rejected(tmp_path: Path) -> None:
    def incomplete_provider(**kwargs: object) -> WeatherDataset:
        return fake_weather(**kwargs, hours=23)  # type: ignore[arg-type]

    with pytest.raises(SiteDataPreparationError, match="24 Wetterstunden"):
        run_preparation(tmp_path, weather_provider=incomplete_provider)


def test_missing_weather_column_is_rejected(tmp_path: Path) -> None:
    def missing_provider(**kwargs: object) -> WeatherDataset:
        result = fake_weather(**kwargs)  # type: ignore[arg-type]
        return WeatherDataset(
            result.data.drop(columns="pressure"),
            result.used_latitude,
            result.used_longitude,
            result.elevation_m,
            result.source,
        )

    with pytest.raises(SiteDataPreparationError, match="pressure"):
        run_preparation(tmp_path, weather_provider=missing_provider)


def test_small_negative_pvgis_wind_value_is_set_to_zero(tmp_path: Path) -> None:
    def provider(**kwargs: object) -> WeatherDataset:
        result = fake_weather(**kwargs)  # type: ignore[arg-type]
        result.data.iloc[3, result.data.columns.get_loc("wind_speed")] = -0.07
        return result

    artifacts = run_preparation(tmp_path, weather_provider=provider)
    corrections = artifacts.metadata["time_axis"][
        "negative_weather_values_between_minus_one_and_zero_set_to_zero"
    ]
    assert corrections["wind_speed"] == 1


def test_material_negative_weather_value_is_rejected(tmp_path: Path) -> None:
    def provider(**kwargs: object) -> WeatherDataset:
        result = fake_weather(**kwargs)  # type: ignore[arg-type]
        result.data.iloc[3, result.data.columns.get_loc("wind_speed")] = -2.0
        return result

    with pytest.raises(SiteDataPreparationError, match="kleiner als -1,0"):
        run_preparation(tmp_path, weather_provider=provider)


def test_weather_source_failure_is_reported(tmp_path: Path) -> None:
    def unavailable_provider(**kwargs: object) -> WeatherDataset:
        del kwargs
        raise SiteDataPreparationError("outside source coverage")

    with pytest.raises(SiteDataPreparationError, match="outside source coverage"):
        run_preparation(tmp_path, weather_provider=unavailable_provider)


def test_grid_emission_source_is_mandatory(tmp_path: Path) -> None:
    with pytest.raises(SiteDataPreparationError, match="Genau eine Netzemissionsquelle"):
        prepare_single_site_h2_input(
            50.78,
            6.08,
            tmp_path / "input.csv",
            expected_hours=24,
            weather_provider=fake_weather,
            profile_generator=fake_profiles,
            country_resolver=country_resolver,
        )


def test_versioned_namibia_grid_factor_is_loaded_automatically(tmp_path: Path) -> None:
    artifacts = prepare_single_site_h2_input(
        -21.0800,
        14.1610,
        tmp_path / "input.csv",
        country_code="NA",
        spatial_resolution_deg=1.0,
        expected_hours=24,
        weather_provider=fake_weather,
        profile_generator=fake_profiles,
    )
    assert artifacts.data["grid_emission_factor"].iloc[0] == pytest.approx(220.0)
    assert artifacts.metadata["site"]["model_location_latitude"] == pytest.approx(-21.5)
    assert artifacts.metadata["site"]["model_location_longitude"] == pytest.approx(14.5)
    emissions = artifacts.metadata["sources"]["grid_emission_factor"]
    assert emissions["spatial_scope"] == "country_national_grid"
    assert emissions["data_year"] == 2024


def test_hourly_price_and_emission_files_are_aligned_by_position(tmp_path: Path) -> None:
    source_index = pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC")
    price_file = tmp_path / "prices.csv"
    emission_file = tmp_path / "emissions.csv"
    pd.DataFrame(
        {"timestamp": source_index, "electricity_price": np.arange(24) - 5.0}
    ).to_csv(price_file, index=False)
    pd.DataFrame(
        {"timestamp": source_index, "grid_emission_factor": np.arange(24) + 100.0}
    ).to_csv(emission_file, index=False)

    artifacts = prepare_single_site_h2_input(
        50.78,
        6.08,
        tmp_path / "input.csv",
        electricity_price_file=price_file,
        grid_emission_file=emission_file,
        expected_hours=24,
        weather_provider=fake_weather,
        profile_generator=fake_profiles,
        country_resolver=country_resolver,
    )

    assert artifacts.data["electricity_price"].iloc[0] == pytest.approx(-5.0)
    assert artifacts.data["grid_emission_factor"].iloc[-1] == pytest.approx(123.0)
    assert artifacts.metadata["sources"]["electricity_price"]["temporal_mapping"] == "positionally_mapped_to_profile_calendar_year"


def test_incomplete_hourly_series_is_rejected(tmp_path: Path) -> None:
    price_file = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=23, freq="h", tz="UTC"),
            "electricity_price": np.ones(23),
        }
    ).to_csv(price_file, index=False)
    with pytest.raises(SiteDataPreparationError, match="23 statt 24"):
        prepare_single_site_h2_input(
            50.78,
            6.08,
            tmp_path / "input.csv",
            electricity_price_file=price_file,
            grid_emission_factor_kg_per_mwh=350.0,
            expected_hours=24,
            weather_provider=fake_weather,
            profile_generator=fake_profiles,
            country_resolver=country_resolver,
        )


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    run_preparation(tmp_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        run_preparation(tmp_path)
    replaced = run_preparation(tmp_path, overwrite=True)
    assert replaced.input_path.is_file()


@pytest.mark.parametrize(
    ("latitude", "longitude", "expected_iso2"),
    [(52.52, 13.405, "DE"), (40.42, -3.70, "ES")],
)
def test_natural_earth_country_lookup(
    latitude: float, longitude: float, expected_iso2: str
) -> None:
    country = resolve_country_from_natural_earth(latitude, longitude)
    assert country.iso2 == expected_iso2


def test_ocean_coordinate_requires_explicit_country() -> None:
    with pytest.raises(SiteDataPreparationError, match="--country-code"):
        resolve_country_from_natural_earth(0.0, -140.0)


def test_haversine_distance_is_zero_for_same_coordinate() -> None:
    assert haversine_distance_km(50.78, 6.08, 50.78, 6.08) == pytest.approx(0.0)


def test_no_cli_arguments_print_help_without_systemexit(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out
