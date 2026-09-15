"""Quellenbasierte RED-III-Regeln für das vereinfachte H2-Modell.

Das Modul trennt regulatorische Parameter und eine unabhängige ex-post-
Prüfung vom Optimierungskern. Es modelliert den allgemeinen Strombezugsweg
nach Artikel 4 Absatz 4 der Delegierten Verordnung (EU) 2023/1184: Die als
vollständig erneuerbar beanspruchte Strommenge muss zusätzlich, zeitlich
korreliert, geografisch korreliert und eindeutig zugeordnet sein.

Die Prüfung ist eine wissenschaftliche Modellvereinfachung und keine
Zertifizierung. Insbesondere werden vertragliche und nationale Nachweise als
explizite boolesche Evidenz übergeben. Für Drittstaaten wird eine Gebotszone
durch das jeweils ähnlichste gleichwertige Konzept ersetzt (Art. 2 und
Erwägungsgrund 3 der Verordnung 2023/1184).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np
import pandas as pd

from config_h2 import H2_LHV_MJ_PER_KG


RED_III_SOURCE_URL: Final[str] = (
    "https://eur-lex.europa.eu/eli/dir/2023/2413/oj"
)
RFNBO_ELECTRICITY_SOURCE_URL: Final[str] = (
    "https://eur-lex.europa.eu/eli/reg_del/2023/1184/2024-06-10"
)
RFNBO_ELECTRICITY_ORIGINAL_SOURCE_URL: Final[str] = (
    "https://eur-lex.europa.eu/eli/reg_del/2023/1184/oj"
)
RFNBO_GHG_SOURCE_URL: Final[str] = (
    "https://eur-lex.europa.eu/eli/reg_del/2023/1185/oj"
)
LEGAL_SNAPSHOT_DATE: Final[str] = "2026-09-10"

TIMESTAMP_COLUMN: Final[str] = "timestamp"
RFNBO_ELECTRICITY_COLUMN: Final[str] = "rf_nbo_electricity_mwh"
ELIGIBLE_RENEWABLE_ELECTRICITY_COLUMN: Final[str] = (
    "eligible_renewable_electricity_mwh"
)
DAY_AHEAD_PRICE_COLUMN: Final[str] = "day_ahead_price_eur_per_mwh"
ETS_PRICE_COLUMN: Final[str] = "ets_price_eur_per_tco2e"


class TemporalCorrelation(str, Enum):
    """Im Modell untersuchte Zeitfenster nach Artikel 6."""

    MONTHLY = "monthly"
    HOURLY = "hourly"


class ElectricityEligibilityRoute(str, Enum):
    """Im Kern implementierter Rechtsweg für netzbezogenen Strom."""

    GENERAL_ARTICLE_4_4 = "general_article_4_4"


@dataclass(frozen=True, slots=True)
class LegalCriterion:
    """Nachvollziehbare Zuordnung einer Modellregel zur Rechtsquelle."""

    name: str
    source_url: str
    article: str
    legal_rule: str
    model_interpretation: str

    def __post_init__(self) -> None:
        for value in (
            self.name,
            self.source_url,
            self.article,
            self.legal_rule,
            self.model_interpretation,
        ):
            if not value.strip():
                raise ValueError("Rechtskriterien müssen vollständig dokumentiert sein.")


LEGAL_CRITERIA: Final[Mapping[str, LegalCriterion]] = MappingProxyType(
    {
        "scope_third_countries": LegalCriterion(
            "Geltung außerhalb der EU",
            RFNBO_ELECTRICITY_SOURCE_URL,
            "Art. 1 und Art. 2 Nr. 1",
            "Die Regeln gelten auch außerhalb der Union; gleichwertige "
            "Gebotszonenkonzepte sind zulässig.",
            "Für Namibia muss eine Gebotszone oder das ähnlichste verfügbare "
            "Markt-/Netzgebiet als externer Nachweis festgelegt werden.",
        ),
        "additionality": LegalCriterion(
            "Zusätzlichkeit",
            RFNBO_ELECTRICITY_SOURCE_URL,
            "Art. 5 und Art. 11",
            "Eigene Erzeugung oder PPA deckt die beanspruchte Strommenge; "
            "Erzeugungsanlage höchstens 36 Monate älter und grundsätzlich ohne "
            "Betriebs- oder Investitionsbeihilfe. Übergangsausnahme bis 2038.",
            "Der Kern erhält einen expliziten externen Nachweis. Alter und "
            "Übergangsregel können mit Hilfsfunktionen vorgeprüft werden.",
        ),
        "temporal_correlation": LegalCriterion(
            "Zeitliche Korrelation",
            RFNBO_ELECTRICITY_SOURCE_URL,
            "Art. 6",
            "Bis Ende 2029 monatliche, ab 2030 stündliche Korrelation; ein "
            "Mitgliedstaat darf ab 1. Juli 2027 früher stündlich prüfen.",
            "Monats- und Stundenfenster werden mengenbilanziell geprüft.",
        ),
        "low_price_exception": LegalCriterion(
            "Niedrigpreis-Ausnahme",
            RFNBO_ELECTRICITY_SOURCE_URL,
            "Art. 6 Abs. 3",
            "Die zeitliche Korrelation gilt in einer Stunde bei einem "
            "Day-Ahead-Preis von höchstens 20 EUR/MWh oder unter 0,36 mal dem "
            "ETS-Zertifikatspreis als erfüllt.",
            "Nur echte Day-Ahead- und ETS-Daten aktivieren die optionale "
            "Ausnahme; der Gewerbestrompreis-Proxy ist dafür unzulässig.",
        ),
        "geographical_correlation": LegalCriterion(
            "Geografische Korrelation",
            RFNBO_ELECTRICITY_SOURCE_URL,
            "Art. 7",
            "Erzeugung liegt in derselben, einer zulässigen verbundenen oder "
            "einer verbundenen Offshore-Gebotszone.",
            "Der Basiskern verwendet dieselbe Gebotszone beziehungsweise ein "
            "gleichwertiges Drittstaatenkonzept und verlangt externen Nachweis.",
        ),
        "allocation": LegalCriterion(
            "Strommengen und eindeutige Zuordnung",
            RFNBO_ELECTRICITY_SOURCE_URL,
            "Art. 8",
            "Stromverbrauch, erneuerbare Erzeugung und angewendeter "
            "Anrechnungsweg müssen belastbar dokumentiert werden.",
            "Der Prüfer verlangt eine stündliche, lückenlose Zeitreihe und "
            "einen extern bestätigten Ausschluss von Doppelzählung.",
        ),
        "ghg_savings": LegalCriterion(
            "Treibhausgaseinsparung",
            RFNBO_GHG_SOURCE_URL,
            "Anhang A Nr. 1, 2 und 5; RED Art. 25 Abs. 2",
            "Mindestens 70 Prozent Einsparung gegenüber 94 g CO2e/MJ; vollständig "
            "erneuerbarer Strom erhält 0 g CO2e/MJ Strom.",
            "Der Prüfer bewertet eine außerhalb dieses Moduls vollständig "
            "berechnete Produktintensität. Maschinenherstellung bleibt gemäß "
            "Anhang A Nr. 1 außerhalb der regulatorischen Rechnung.",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class RedIIIParameters:
    """Numerische, aus offiziellen EU-Quellen abgeleitete Regelparameter."""

    minimum_ghg_savings_fraction: float = 0.70
    fossil_comparator_g_co2e_per_mj: float = 94.0
    h2_lhv_mj_per_kg: float = H2_LHV_MJ_PER_KG
    monthly_rule_end: str = "2029-12-31T23:59:59Z"
    hourly_rule_start: str = "2030-01-01T00:00:00Z"
    earliest_member_state_hourly_start: str = "2027-07-01T00:00:00Z"
    additionality_transition_commissioning_cutoff: str = "2028-01-01T00:00:00Z"
    additionality_transition_end: str = "2038-01-01T00:00:00Z"
    maximum_renewable_asset_age_months: int = 36
    low_price_threshold_eur_per_mwh: float = 20.0
    ets_price_multiplier: float = 0.36
    high_renewable_share_threshold_fraction: float = 0.90
    low_grid_intensity_threshold_g_co2e_per_mj: float = 18.0

    @property
    def maximum_product_intensity_g_co2e_per_mj(self) -> float:
        return self.fossil_comparator_g_co2e_per_mj * (
            1.0 - self.minimum_ghg_savings_fraction
        )

    @property
    def maximum_product_intensity_kg_co2e_per_kg_h2(self) -> float:
        return (
            self.maximum_product_intensity_g_co2e_per_mj
            * self.h2_lhv_mj_per_kg
            / 1_000.0
        )

    @property
    def low_grid_intensity_threshold_kg_co2e_per_mwh(self) -> float:
        return self.low_grid_intensity_threshold_g_co2e_per_mj * 3.6

    def validate(self) -> None:
        if not 0.0 <= self.minimum_ghg_savings_fraction <= 1.0:
            raise ValueError("Die THG-Mindesteinsparung muss zwischen 0 und 1 liegen.")
        for value in (
            self.fossil_comparator_g_co2e_per_mj,
            self.h2_lhv_mj_per_kg,
            self.low_price_threshold_eur_per_mwh,
            self.ets_price_multiplier,
            self.low_grid_intensity_threshold_g_co2e_per_mj,
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError("RED-III-Zahlenwerte müssen endlich und nichtnegativ sein.")
        if not 0.0 <= self.high_renewable_share_threshold_fraction <= 1.0:
            raise ValueError("Der Erneuerbaren-Anteil muss zwischen 0 und 1 liegen.")
        if self.maximum_renewable_asset_age_months < 0:
            raise ValueError("Das Anlagenalter darf nicht negativ sein.")


DEFAULT_RED_III_PARAMETERS: Final[RedIIIParameters] = RedIIIParameters()
DEFAULT_RED_III_PARAMETERS.validate()


@dataclass(frozen=True, slots=True)
class EligibilityEvidence:
    """Extern zu belegende Voraussetzungen des vereinfachten Art.-4(4)-Wegs."""

    additionality_verified: bool
    geographical_correlation_verified: bool
    exclusive_allocation_verified: bool

    def __post_init__(self) -> None:
        for value in (
            self.additionality_verified,
            self.geographical_correlation_verified,
            self.exclusive_allocation_verified,
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError("Alle Nachweisfelder müssen boolesch sein.")


@dataclass(frozen=True, slots=True)
class RedIIIComplianceResult:
    """Prüfergebnis samt periodenweiser, exportierbarer Begründung."""

    is_compliant: bool
    temporal_compliant: bool
    additionality_compliant: bool
    geographical_correlation_compliant: bool
    exclusive_allocation_compliant: bool
    ghg_compliant: bool
    temporal_mode: TemporalCorrelation
    route: ElectricityEligibilityRoute
    product_intensity_g_co2e_per_mj: float
    product_intensity_kg_co2e_per_kg_h2: float
    ghg_savings_fraction: float
    failed_periods: tuple[str, ...]
    period_results: pd.DataFrame
    messages: tuple[str, ...]


def temporal_mode_for_date(
    production_date: object,
    *,
    member_state_hourly_start: object | None = None,
    parameters: RedIIIParameters = DEFAULT_RED_III_PARAMETERS,
) -> TemporalCorrelation:
    """Bestimme das geltende Zeitfenster aus Artikel 6."""

    parameters.validate()
    production = _as_utc_timestamp(production_date, "production_date")
    mandatory_hourly = pd.Timestamp(parameters.hourly_rule_start)

    if member_state_hourly_start is None:
        hourly_start = mandatory_hourly
    else:
        hourly_start = _as_utc_timestamp(
            member_state_hourly_start, "member_state_hourly_start"
        )
        earliest = pd.Timestamp(parameters.earliest_member_state_hourly_start)
        if hourly_start < earliest or hourly_start > mandatory_hourly:
            raise ValueError(
                "member_state_hourly_start muss zwischen 2027-07-01 und "
                "2030-01-01 liegen."
            )

    return (
        TemporalCorrelation.HOURLY
        if production >= hourly_start
        else TemporalCorrelation.MONTHLY
    )


def build_correlation_periods(
    timestamps: pd.Series,
    mode: TemporalCorrelation | str,
) -> pd.Series:
    """Erzeuge stabile Monats- oder Stundenschlüssel für eine UTC-Zeitreihe."""

    correlation_mode = _coerce_temporal_mode(mode)
    parsed = pd.to_datetime(timestamps, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("timestamps enthält ungültige Datumswerte.")

    if correlation_mode is TemporalCorrelation.MONTHLY:
        labels = parsed.dt.strftime("%Y-%m")
    else:
        labels = parsed.dt.strftime("%Y-%m-%dT%H:00Z")
    labels.name = "correlation_period"
    return labels


def low_price_exception_mask(
    operation: pd.DataFrame,
    *,
    ets_price_eur_per_tco2e: float | None = None,
    parameters: RedIIIParameters = DEFAULT_RED_III_PARAMETERS,
) -> pd.Series:
    """Prüfe die stündliche Niedrigpreis-Ausnahme aus Artikel 6 Absatz 3."""

    parameters.validate()
    if DAY_AHEAD_PRICE_COLUMN not in operation.columns:
        raise ValueError(
            f"Für die Niedrigpreis-Ausnahme fehlt {DAY_AHEAD_PRICE_COLUMN}."
        )
    day_ahead = _numeric_series(operation[DAY_AHEAD_PRICE_COLUMN], DAY_AHEAD_PRICE_COLUMN)
    fixed_threshold = day_ahead <= parameters.low_price_threshold_eur_per_mwh

    if ETS_PRICE_COLUMN in operation.columns:
        ets_price = _numeric_series(operation[ETS_PRICE_COLUMN], ETS_PRICE_COLUMN)
    elif ets_price_eur_per_tco2e is not None:
        if not isfinite(float(ets_price_eur_per_tco2e)) or ets_price_eur_per_tco2e < 0:
            raise ValueError("ets_price_eur_per_tco2e muss endlich und nichtnegativ sein.")
        ets_price = pd.Series(
            float(ets_price_eur_per_tco2e), index=operation.index, dtype=float
        )
    else:
        ets_price = None

    if ets_price is None:
        return fixed_threshold.rename("low_price_exception")
    ets_threshold = day_ahead < parameters.ets_price_multiplier * ets_price
    return (fixed_threshold | ets_threshold).rename("low_price_exception")


def calculate_ghg_savings_fraction(
    product_intensity_g_co2e_per_mj: float,
    *,
    parameters: RedIIIParameters = DEFAULT_RED_III_PARAMETERS,
) -> float:
    """Berechne die Einsparung gegenüber dem fossilen Vergleichswert."""

    parameters.validate()
    intensity = float(product_intensity_g_co2e_per_mj)
    if not isfinite(intensity) or intensity < 0.0:
        raise ValueError("Die Produktintensität muss endlich und nichtnegativ sein.")
    return 1.0 - intensity / parameters.fossil_comparator_g_co2e_per_mj


def additionality_is_required(
    production_date: object,
    rf_nbo_commissioning_date: object,
    *,
    capacity_added_after_2028: bool = False,
    parameters: RedIIIParameters = DEFAULT_RED_III_PARAMETERS,
) -> bool:
    """Bilde die Übergangsregel des Artikels 11 ab."""

    if not isinstance(capacity_added_after_2028, (bool, np.bool_)):
        raise TypeError("capacity_added_after_2028 muss boolesch sein.")
    production = _as_utc_timestamp(production_date, "production_date")
    commissioning = _as_utc_timestamp(
        rf_nbo_commissioning_date, "rf_nbo_commissioning_date"
    )
    if production < commissioning:
        raise ValueError("production_date darf nicht vor der Inbetriebnahme liegen.")
    if capacity_added_after_2028:
        return True
    cutoff = pd.Timestamp(parameters.additionality_transition_commissioning_cutoff)
    transition_end = pd.Timestamp(parameters.additionality_transition_end)
    return not (commissioning < cutoff and production < transition_end)


def renewable_asset_age_is_compliant(
    rf_nbo_commissioning_date: object,
    renewable_commissioning_date: object,
    *,
    parameters: RedIIIParameters = DEFAULT_RED_III_PARAMETERS,
) -> bool:
    """Prüfe, ob die EE-Anlage höchstens 36 Monate früher startete."""

    rf_nbo_date = _as_utc_timestamp(
        rf_nbo_commissioning_date, "rf_nbo_commissioning_date"
    )
    renewable_date = _as_utc_timestamp(
        renewable_commissioning_date, "renewable_commissioning_date"
    )
    earliest_allowed = rf_nbo_date - pd.DateOffset(
        months=parameters.maximum_renewable_asset_age_months
    )
    return bool(renewable_date >= earliest_allowed)


def assess_red_iii_compliance(
    operation: pd.DataFrame,
    *,
    temporal_mode: TemporalCorrelation | str,
    evidence: EligibilityEvidence,
    product_intensity_kg_co2e_per_kg_h2: float,
    use_low_price_exception: bool = False,
    ets_price_eur_per_tco2e: float | None = None,
    tolerance_mwh: float = 1e-9,
    parameters: RedIIIParameters = DEFAULT_RED_III_PARAMETERS,
) -> RedIIIComplianceResult:
    """Bewerte einen Betriebsplan unabhängig vom Optimierer.

    ``rf_nbo_electricity_mwh`` ist im derzeitigen Modell die Summe aus
    Elektrolyse- und Verdichtungsstrom. ``eligible_renewable_electricity_mwh``
    enthält nur tatsächlich erzeugte und diesem H2 eindeutig zugeordnete
    erneuerbare Strommengen. Netzstrom wird nicht automatisch angerechnet.
    """

    parameters.validate()
    mode = _coerce_temporal_mode(temporal_mode)
    if not isinstance(evidence, EligibilityEvidence):
        raise TypeError("evidence muss eine EligibilityEvidence-Instanz sein.")
    if not isinstance(use_low_price_exception, (bool, np.bool_)):
        raise TypeError("use_low_price_exception muss boolesch sein.")
    if not isfinite(float(tolerance_mwh)) or tolerance_mwh < 0.0:
        raise ValueError("tolerance_mwh muss endlich und nichtnegativ sein.")

    intensity_kg = float(product_intensity_kg_co2e_per_kg_h2)
    if not isfinite(intensity_kg) or intensity_kg < 0.0:
        raise ValueError(
            "product_intensity_kg_co2e_per_kg_h2 muss endlich und nichtnegativ sein."
        )

    data = _validated_operation(operation)
    periods = build_correlation_periods(data[TIMESTAMP_COLUMN], mode)
    if use_low_price_exception:
        exception = low_price_exception_mask(
            data,
            ets_price_eur_per_tco2e=ets_price_eur_per_tco2e,
            parameters=parameters,
        )
    else:
        exception = pd.Series(False, index=data.index, name="low_price_exception")

    working = pd.DataFrame(
        {
            "period": periods,
            "required_electricity_mwh": data[RFNBO_ELECTRICITY_COLUMN],
            "eligible_renewable_electricity_mwh": data[
                ELIGIBLE_RENEWABLE_ELECTRICITY_COLUMN
            ],
            "low_price_exception_electricity_mwh": np.where(
                exception, data[RFNBO_ELECTRICITY_COLUMN], 0.0
            ),
        }
    )
    period_results = (
        working.groupby("period", sort=True, as_index=False)
        .sum(numeric_only=True)
        .reset_index(drop=True)
    )
    period_results["total_eligible_electricity_mwh"] = (
        period_results["eligible_renewable_electricity_mwh"]
        + period_results["low_price_exception_electricity_mwh"]
    )
    period_results["balance_mwh"] = (
        period_results["total_eligible_electricity_mwh"]
        - period_results["required_electricity_mwh"]
    )
    period_results["compliant"] = (
        period_results["balance_mwh"] >= -float(tolerance_mwh)
    )

    temporal_compliant = bool(period_results["compliant"].all())
    intensity_g_per_mj = intensity_kg * 1_000.0 / parameters.h2_lhv_mj_per_kg
    ghg_savings = calculate_ghg_savings_fraction(
        intensity_g_per_mj, parameters=parameters
    )
    ghg_compliant = bool(
        intensity_g_per_mj
        <= parameters.maximum_product_intensity_g_co2e_per_mj + 1e-12
    )

    checks = (
        temporal_compliant,
        bool(evidence.additionality_verified),
        bool(evidence.geographical_correlation_verified),
        bool(evidence.exclusive_allocation_verified),
        ghg_compliant,
    )
    is_compliant = all(checks)
    failed_periods = tuple(
        period_results.loc[~period_results["compliant"], "period"].astype(str)
    )

    messages: list[str] = []
    if failed_periods:
        messages.append(
            f"Zeitliche Korrelation verletzt in {len(failed_periods)} Periode(n)."
        )
    if not evidence.additionality_verified:
        messages.append("Zusätzlichkeitsnachweis fehlt oder ist nicht erfüllt.")
    if not evidence.geographical_correlation_verified:
        messages.append("Nachweis der geografischen Korrelation fehlt.")
    if not evidence.exclusive_allocation_verified:
        messages.append("Eindeutige Strommengenallokation ist nicht nachgewiesen.")
    if not ghg_compliant:
        messages.append("Der regulatorische THG-Grenzwert wird überschritten.")
    if not messages:
        messages.append("Alle im vereinfachten Prüfer enthaltenen Kriterien sind erfüllt.")

    return RedIIIComplianceResult(
        is_compliant=is_compliant,
        temporal_compliant=temporal_compliant,
        additionality_compliant=bool(evidence.additionality_verified),
        geographical_correlation_compliant=bool(
            evidence.geographical_correlation_verified
        ),
        exclusive_allocation_compliant=bool(
            evidence.exclusive_allocation_verified
        ),
        ghg_compliant=ghg_compliant,
        temporal_mode=mode,
        route=ElectricityEligibilityRoute.GENERAL_ARTICLE_4_4,
        product_intensity_g_co2e_per_mj=intensity_g_per_mj,
        product_intensity_kg_co2e_per_kg_h2=intensity_kg,
        ghg_savings_fraction=ghg_savings,
        failed_periods=failed_periods,
        period_results=period_results,
        messages=tuple(messages),
    )


def operation_from_optimizer_output(hourly_operation: pd.DataFrame) -> pd.DataFrame:
    """Übersetze einen H2-Ergebnisdatensatz in die unabhängige Prüfschnittstelle."""

    required = {
        TIMESTAMP_COLUMN,
        "pv_generation_mwh",
        "wind_generation_mwh",
        "electrolyzer_electricity_mwh",
        "compressor_electricity_mwh",
    }
    missing = sorted(required - set(hourly_operation.columns))
    if missing:
        raise ValueError(
            "Im Optimierungsergebnis fehlen Spalten: " + ", ".join(missing)
        )
    translated = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: hourly_operation[TIMESTAMP_COLUMN].copy(),
            RFNBO_ELECTRICITY_COLUMN: (
                hourly_operation["electrolyzer_electricity_mwh"]
                + hourly_operation["compressor_electricity_mwh"]
            ),
            ELIGIBLE_RENEWABLE_ELECTRICITY_COLUMN: (
                hourly_operation["pv_generation_mwh"]
                + hourly_operation["wind_generation_mwh"]
            ),
        }
    )
    return _validated_operation(translated)


def _validated_operation(operation: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(operation, pd.DataFrame) or operation.empty:
        raise ValueError("operation muss ein nichtleerer DataFrame sein.")
    required = {
        TIMESTAMP_COLUMN,
        RFNBO_ELECTRICITY_COLUMN,
        ELIGIBLE_RENEWABLE_ELECTRICITY_COLUMN,
    }
    missing = sorted(required - set(operation.columns))
    if missing:
        raise ValueError("Pflichtspalten fehlen: " + ", ".join(missing))

    data = operation.copy(deep=True).reset_index(drop=True)
    timestamps = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("timestamp enthält ungültige Datumswerte.")
    if timestamps.duplicated().any():
        raise ValueError("timestamp darf keine Duplikate enthalten.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamp muss aufsteigend sortiert sein.")
    if len(timestamps) > 1:
        intervals = timestamps.diff().iloc[1:]
        if not (intervals == pd.Timedelta(hours=1)).all():
            raise ValueError("timestamp muss eine lückenlose Stundenreihe bilden.")
    data[TIMESTAMP_COLUMN] = timestamps

    for column in (RFNBO_ELECTRICITY_COLUMN, ELIGIBLE_RENEWABLE_ELECTRICITY_COLUMN):
        data[column] = _numeric_series(data[column], column)
        if (data[column] < 0.0).any():
            raise ValueError(f"{column} darf nicht negativ sein.")
    if float(data[RFNBO_ELECTRICITY_COLUMN].sum()) <= 0.0:
        raise ValueError("Der geprüfte Betriebsplan enthält keinen H2-Stromverbrauch.")
    return data


def _numeric_series(values: pd.Series, name: str) -> pd.Series:
    converted = pd.to_numeric(values, errors="coerce").astype(float)
    if converted.isna().any() or not np.isfinite(converted.to_numpy()).all():
        raise ValueError(f"{name} muss ausschließlich endliche Zahlen enthalten.")
    return converted


def _coerce_temporal_mode(mode: TemporalCorrelation | str) -> TemporalCorrelation:
    try:
        return mode if isinstance(mode, TemporalCorrelation) else TemporalCorrelation(mode)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in TemporalCorrelation)
        raise ValueError(f"Unbekannter temporal_mode. Erlaubt: {allowed}.") from exc


def _as_utc_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} ist kein gültiger Zeitpunkt.") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} ist kein gültiger Zeitpunkt.")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
