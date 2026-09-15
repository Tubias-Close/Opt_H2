"""Grenzfalltests für den unabhängigen RED-III-Prüfbaustein."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from red_iii_data import (
    DAY_AHEAD_PRICE_COLUMN,
    DEFAULT_RED_III_PARAMETERS,
    ELIGIBLE_RENEWABLE_ELECTRICITY_COLUMN,
    ETS_PRICE_COLUMN,
    LEGAL_CRITERIA,
    RFNBO_ELECTRICITY_COLUMN,
    EligibilityEvidence,
    TemporalCorrelation,
    additionality_is_required,
    assess_red_iii_compliance,
    build_correlation_periods,
    calculate_ghg_savings_fraction,
    low_price_exception_mask,
    operation_from_optimizer_output,
    renewable_asset_age_is_compliant,
    temporal_mode_for_date,
)


FULL_EVIDENCE = EligibilityEvidence(
    additionality_verified=True,
    geographical_correlation_verified=True,
    exclusive_allocation_verified=True,
)


def operation(
    required: list[float],
    eligible: list[float],
    *,
    start: str = "2029-01-01 00:00",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=len(required), freq="h"),
            RFNBO_ELECTRICITY_COLUMN: required,
            ELIGIBLE_RENEWABLE_ELECTRICITY_COLUMN: eligible,
        }
    )


def assess(
    data: pd.DataFrame,
    mode: TemporalCorrelation = TemporalCorrelation.HOURLY,
    **kwargs: object,
):
    return assess_red_iii_compliance(
        data,
        temporal_mode=mode,
        evidence=FULL_EVIDENCE,
        product_intensity_kg_co2e_per_kg_h2=3.0,
        **kwargs,
    )


def test_official_ghg_threshold_is_converted_consistently() -> None:
    rules = DEFAULT_RED_III_PARAMETERS
    assert rules.maximum_product_intensity_g_co2e_per_mj == pytest.approx(28.2)
    assert rules.maximum_product_intensity_kg_co2e_per_kg_h2 == pytest.approx(3.384)
    assert rules.low_grid_intensity_threshold_kg_co2e_per_mwh == pytest.approx(64.8)


def test_every_legal_criterion_has_source_article_and_interpretation() -> None:
    assert set(LEGAL_CRITERIA) >= {
        "additionality",
        "temporal_correlation",
        "geographical_correlation",
        "ghg_savings",
    }
    for criterion in LEGAL_CRITERIA.values():
        assert criterion.source_url.startswith("https://eur-lex.europa.eu/")
        assert criterion.article
        assert criterion.legal_rule
        assert criterion.model_interpretation


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2029-12-31 23:00Z", TemporalCorrelation.MONTHLY),
        ("2030-01-01 00:00Z", TemporalCorrelation.HOURLY),
    ],
)
def test_default_temporal_changeover(date: str, expected: TemporalCorrelation) -> None:
    assert temporal_mode_for_date(date) is expected


def test_member_state_can_switch_early_from_july_2027() -> None:
    start = "2027-07-01 00:00Z"
    assert temporal_mode_for_date(
        "2027-06-30 23:00Z", member_state_hourly_start=start
    ) is TemporalCorrelation.MONTHLY
    assert temporal_mode_for_date(
        "2027-07-01 00:00Z", member_state_hourly_start=start
    ) is TemporalCorrelation.HOURLY


def test_illegal_early_member_state_changeover_is_rejected() -> None:
    with pytest.raises(ValueError, match="2027-07-01"):
        temporal_mode_for_date(
            "2027-07-01", member_state_hourly_start="2027-06-30"
        )


def test_period_builder_returns_month_and_exact_hour() -> None:
    timestamps = pd.Series(
        pd.date_range("2029-01-31 23:00", periods=2, freq="h", tz="UTC")
    )
    assert build_correlation_periods(
        timestamps, TemporalCorrelation.MONTHLY
    ).tolist() == ["2029-01", "2029-02"]
    assert build_correlation_periods(
        timestamps, TemporalCorrelation.HOURLY
    ).tolist() == ["2029-01-31T23:00Z", "2029-02-01T00:00Z"]


def test_monthly_matching_allows_shifting_within_same_month() -> None:
    data = operation([2.0, 0.0], [0.0, 2.0])
    result = assess(data, TemporalCorrelation.MONTHLY)
    assert result.temporal_compliant
    assert result.is_compliant
    assert result.failed_periods == ()


def test_hourly_matching_rejects_same_shift() -> None:
    data = operation([2.0, 0.0], [0.0, 2.0])
    result = assess(data, TemporalCorrelation.HOURLY)
    assert not result.temporal_compliant
    assert not result.is_compliant
    assert result.failed_periods == ("2029-01-01T00:00Z",)


def test_monthly_matching_cannot_shift_between_months() -> None:
    data = operation(
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        start="2029-01-31 22:00",
    )
    result = assess(data, TemporalCorrelation.MONTHLY)
    assert not result.temporal_compliant
    assert result.failed_periods == ("2029-01",)


def test_exact_temporal_boundary_is_compliant() -> None:
    data = operation([1.0, 2.0], [1.0, 2.0])
    result = assess(data)
    assert result.is_compliant
    assert (result.period_results["balance_mwh"] == 0.0).all()


def test_day_ahead_fixed_price_boundary_uses_less_or_equal() -> None:
    data = operation([1.0, 1.0], [0.0, 0.0])
    data[DAY_AHEAD_PRICE_COLUMN] = [20.0, 20.01]
    assert low_price_exception_mask(data).tolist() == [True, False]


def test_ets_price_boundary_is_strictly_lower() -> None:
    data = operation([1.0, 1.0], [0.0, 0.0])
    data[DAY_AHEAD_PRICE_COLUMN] = [28.79, 28.8]
    data[ETS_PRICE_COLUMN] = [80.0, 80.0]
    assert low_price_exception_mask(data).tolist() == [True, False]


def test_low_price_exception_can_cover_an_hourly_shortage() -> None:
    data = operation([1.0, 1.0], [0.0, 1.0])
    data[DAY_AHEAD_PRICE_COLUMN] = [20.0, 100.0]
    without_exception = assess(data)
    with_exception = assess(data, use_low_price_exception=True)
    assert not without_exception.temporal_compliant
    assert with_exception.temporal_compliant


@pytest.mark.parametrize(
    "evidence",
    [
        EligibilityEvidence(False, True, True),
        EligibilityEvidence(True, False, True),
        EligibilityEvidence(True, True, False),
    ],
)
def test_each_external_evidence_is_required(evidence: EligibilityEvidence) -> None:
    result = assess_red_iii_compliance(
        operation([1.0], [1.0]),
        temporal_mode="hourly",
        evidence=evidence,
        product_intensity_kg_co2e_per_kg_h2=0.0,
    )
    assert not result.is_compliant


def test_ghg_limit_is_inclusive_and_above_limit_fails() -> None:
    data = operation([1.0], [1.0])
    at_limit = assess_red_iii_compliance(
        data,
        temporal_mode="hourly",
        evidence=FULL_EVIDENCE,
        product_intensity_kg_co2e_per_kg_h2=3.384,
    )
    above_limit = assess_red_iii_compliance(
        data,
        temporal_mode="hourly",
        evidence=FULL_EVIDENCE,
        product_intensity_kg_co2e_per_kg_h2=3.384001,
    )
    assert at_limit.ghg_compliant and at_limit.is_compliant
    assert not above_limit.ghg_compliant and not above_limit.is_compliant
    assert at_limit.ghg_savings_fraction == pytest.approx(0.70)


def test_ghg_savings_formula_uses_94_g_comparator() -> None:
    assert calculate_ghg_savings_fraction(0.0) == pytest.approx(1.0)
    assert calculate_ghg_savings_fraction(28.2) == pytest.approx(0.70)
    assert calculate_ghg_savings_fraction(94.0) == pytest.approx(0.0)


def test_additionality_transition_ends_in_2038() -> None:
    commissioning = "2027-12-31"
    assert not additionality_is_required("2037-12-31", commissioning)
    assert additionality_is_required("2038-01-01", commissioning)
    assert additionality_is_required(
        "2029-01-01", commissioning, capacity_added_after_2028=True
    )
    assert additionality_is_required("2029-01-01", "2028-01-01")


def test_renewable_asset_age_accepts_exactly_36_months() -> None:
    assert renewable_asset_age_is_compliant("2030-01-01", "2027-01-01")
    assert not renewable_asset_age_is_compliant("2030-01-01", "2026-12-31")


def test_optimizer_output_translation_includes_compression_and_not_grid() -> None:
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2029-01-01", periods=2, freq="h"),
            "pv_generation_mwh": [1.0, 2.0],
            "wind_generation_mwh": [3.0, 4.0],
            "grid_import_mwh": [9.0, 9.0],
            "electrolyzer_electricity_mwh": [3.5, 5.0],
            "compressor_electricity_mwh": [0.5, 1.0],
        }
    )
    translated = operation_from_optimizer_output(source)
    assert translated[RFNBO_ELECTRICITY_COLUMN].tolist() == [4.0, 6.0]
    assert translated[ELIGIBLE_RENEWABLE_ELECTRICITY_COLUMN].tolist() == [4.0, 6.0]


def test_checker_does_not_modify_input() -> None:
    data = operation([1.0], [1.0])
    original = data.copy(deep=True)
    assess(data)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda frame: frame.drop(columns=[RFNBO_ELECTRICITY_COLUMN]),
        lambda frame: frame.assign(**{RFNBO_ELECTRICITY_COLUMN: [-1.0, 1.0]}),
        lambda frame: frame.assign(
            **{ELIGIBLE_RENEWABLE_ELECTRICITY_COLUMN: [np.nan, 1.0]}
        ),
        lambda frame: frame.assign(timestamp=["2029-01-01 00:00", "bad"]),
    ],
)
def test_invalid_operation_is_rejected(mutator) -> None:
    with pytest.raises(ValueError):
        assess(mutator(operation([1.0, 1.0], [1.0, 1.0])))


def test_invalid_parameters_are_rejected() -> None:
    invalid = replace(DEFAULT_RED_III_PARAMETERS, minimum_ghg_savings_fraction=1.1)
    with pytest.raises(ValueError, match="THG-Mindesteinsparung"):
        invalid.validate()
