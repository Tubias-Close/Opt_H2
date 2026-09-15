"""Tests für Annahmen, Metadaten und Einheiten in config_h2.py."""

from dataclasses import replace

import pytest

from config_h2 import (
    DEFAULT_CONFIG,
    H2_LHV_KWH_PER_KG,
    ScalarParameter,
    Scenario,
    SiteConfig,
    default_model_config,
    iter_scalar_parameters,
    with_sensitivity_parameter,
    with_uniform_real_wacc,
)


def test_uniform_wacc_override_keeps_default_unchanged() -> None:
    updated = with_uniform_real_wacc(
        DEFAULT_CONFIG,
        0.11,
        source="Regionaler Testwert",
        reference_year=2025,
    )
    assert updated.technologies.pv.real_wacc_fraction.value == pytest.approx(0.11)
    assert updated.technologies.electrolyzer.real_wacc_fraction.value == pytest.approx(0.11)
    assert updated.technologies.h2_storage.real_wacc_fraction.value == pytest.approx(0.11)
    assert DEFAULT_CONFIG.technologies.pv.real_wacc_fraction.value == pytest.approx(0.05)


def test_sensitivity_override_changes_only_selected_parameter() -> None:
    updated = with_sensitivity_parameter(
        DEFAULT_CONFIG,
        "electrolyzer_capex_eur_per_kw",
        1_000.0,
        source="Methodischer Testfall",
        note="Nur für den Test",
    )
    assert updated.technologies.electrolyzer.capex_eur_per_kw.value == pytest.approx(
        1_000.0
    )
    assert updated.technologies.pv == DEFAULT_CONFIG.technologies.pv
    assert (
        DEFAULT_CONFIG.technologies.electrolyzer.capex_eur_per_kw.value
        == pytest.approx(1_297.0)
    )


def test_sensitivity_override_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Unbekannter"):
        with_sensitivity_parameter(
            DEFAULT_CONFIG,
            "invented_parameter",
            1.0,
            source="Methodischer Testfall",
        )


def test_default_configuration_is_valid() -> None:
    config = default_model_config()
    config.validate()
    assert config.scenario is Scenario.REFERENCE


def test_brandt_demand_is_converted_consistently() -> None:
    study = DEFAULT_CONFIG.study
    assert study.h2_demand_kg_per_hour == pytest.approx(10_000.0 / 24.0)
    assert study.h2_demand_kg_per_year == pytest.approx(3_650_000.0)


def test_electrolyzer_energy_units_and_efficiency() -> None:
    electrolyzer = DEFAULT_CONFIG.technologies.electrolyzer
    assert electrolyzer.specific_electricity_mwh_per_kg_h2 == pytest.approx(0.0525)
    assert electrolyzer.implied_lhv_efficiency_fraction == pytest.approx(
        H2_LHV_KWH_PER_KG / 52.5
    )


def test_capacity_cost_is_converted_from_kw_to_mw() -> None:
    pv = DEFAULT_CONFIG.technologies.pv
    assert pv.capex_eur_per_mw == pytest.approx(921_000.0)
    assert pv.fixed_opex_eur_per_mw_year == pytest.approx(15_100.0)


def test_compressor_energy_includes_both_efficiencies() -> None:
    compressor = DEFAULT_CONFIG.technologies.compressor
    assert compressor.specific_electricity_kwh_per_kg_h2 == pytest.approx(
        1.287 / (0.80 * 0.90)
    )


def test_every_scalar_has_unit_source_and_optional_valid_year() -> None:
    parameters = dict(iter_scalar_parameters(DEFAULT_CONFIG))
    assert parameters
    for name, parameter in parameters.items():
        assert name
        assert parameter.unit.strip()
        assert parameter.source.strip()
        assert parameter.reference_year is None or parameter.reference_year > 0


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(50.78, 6.08), (-90.0, -180.0), (90.0, 180.0)],
)
def test_site_accepts_any_valid_coordinate(latitude: float, longitude: float) -> None:
    SiteConfig(latitude=latitude, longitude=longitude).validate()


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(90.1, 0.0), (-90.1, 0.0), (0.0, 180.1), (0.0, -180.1)],
)
def test_site_rejects_invalid_coordinate(latitude: float, longitude: float) -> None:
    with pytest.raises(ValueError):
        SiteConfig(latitude=latitude, longitude=longitude).validate()


def test_invalid_wacc_is_rejected() -> None:
    config = default_model_config()
    invalid_wacc = ScalarParameter(
        1.0,
        "fraction",
        "Testwert für einen ungültigen Grenzfall",
        2023,
    )
    invalid_pv = replace(config.technologies.pv, real_wacc_fraction=invalid_wacc)
    invalid_technologies = replace(config.technologies, pv=invalid_pv)
    invalid_config = replace(config, technologies=invalid_technologies)
    with pytest.raises(ValueError, match="WACC"):
        invalid_config.validate()


def test_inconsistent_pressure_boundary_is_rejected() -> None:
    config = default_model_config()
    invalid_delivery_pressure = ScalarParameter(
        30.0,
        "bar",
        "Testwert für einen ungültigen Grenzfall",
    )
    invalid_study = replace(
        config.study,
        delivery_pressure_bar=invalid_delivery_pressure,
    )
    invalid_config = replace(config, study=invalid_study)
    with pytest.raises(ValueError, match="Übergabedruck"):
        invalid_config.validate()
