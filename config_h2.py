"""Zentrale, nebenwirkungsfreie Konfiguration des vereinfachten H2-Modells.

Das Modul definiert ausschließlich Annahmen, Einheiten und Metadaten. Beim
Import werden keine Excel-, CSV-, JSON- oder Wetterdateien geöffnet. Reale
Standortdaten werden in einem späteren Schritt über klar getrennte Loader
eingelesen.

Alle Geldwerte des Literatur-Basisfalls sind reale EUR des Jahres 2023. Die
technischen und wirtschaftlichen Basiswerte stammen aus Brandt et al. (2024),
Supplementary Tables 2--4. Abweichende Standort- oder Sensitivitätswerte
werden später ausdrücklich als Überschreibungen übergeben.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Final, Iterator


REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parent

HOURS_PER_DAY: Final[float] = 24.0
DAYS_PER_YEAR: Final[float] = 365.0
HOURS_PER_YEAR: Final[float] = HOURS_PER_DAY * DAYS_PER_YEAR
KILOWATT_PER_MEGAWATT: Final[float] = 1_000.0
KILOWATT_HOURS_PER_MEGAWATT_HOUR: Final[float] = 1_000.0
H2_LHV_MJ_PER_KG: Final[float] = 120.0
H2_LHV_KWH_PER_KG: Final[float] = H2_LHV_MJ_PER_KG / 3.6

BRANDT_ARTICLE_SOURCE: Final[str] = (
    "Brandt et al. (2024), Nature Energy 9, 703-713, "
    "doi:10.1038/s41560-024-01511-z"
)
BRANDT_SUPPLEMENT_SOURCE: Final[str] = (
    f"{BRANDT_ARTICLE_SOURCE}, Supplementary Tables 2-4"
)
MODEL_DEFINITION_SOURCE: Final[str] = "Eigene Modellfestlegung dieser Forschungsarbeit"
EXISTING_TMY_SOURCE: Final[str] = (
    "Bestehender Modellbaustein calculate_renewable_yield.py: PVGIS-TMY "
    "2007-2016, auf das Nicht-Schaltjahr 2025 abgebildet"
)


class Scenario(str, Enum):
    """Stromversorgungsszenarien, deren Regeln später implementiert werden."""

    REFERENCE = "reference"
    RED_MONTHLY = "red_monthly"
    RED_HOURLY = "red_hourly"
    OFF_GRID = "off_grid"


@dataclass(frozen=True, slots=True)
class ScalarParameter:
    """Skalarer Modellparameter mit prüfbarer Dokumentation."""

    value: float
    unit: str
    source: str
    reference_year: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("Parameterwerte müssen reelle Zahlen sein.")
        if not isfinite(float(self.value)):
            raise ValueError("Parameterwerte müssen endlich sein.")
        if not self.unit.strip():
            raise ValueError("Für jeden Parameter muss eine Einheit angegeben werden.")
        if not self.source.strip():
            raise ValueError("Für jeden Parameter muss eine Quelle angegeben werden.")
        if self.reference_year is not None and self.reference_year <= 0:
            raise ValueError("Das Bezugsjahr muss positiv sein.")


@dataclass(frozen=True, slots=True)
class PathConfig:
    """Repository-relative Pfade ohne automatischen Dateizugriff."""

    repository_root: Path = REPOSITORY_ROOT

    @property
    def input_directory(self) -> Path:
        return self.repository_root / "input_data"

    @property
    def output_directory(self) -> Path:
        return self.repository_root / "outputs_h2"

    @property
    def legacy_cost_workbook(self) -> Path:
        return self.input_directory / "technology_costs.xlsx"

    @property
    def wacc_database(self) -> Path:
        return self.input_directory / "SteffenEtAl2025_WACC_database.csv"

    def validate(self) -> None:
        if not self.repository_root.is_absolute():
            raise ValueError("repository_root muss ein absoluter Pfad sein.")


@dataclass(frozen=True, slots=True)
class SiteConfig:
    """Ein frei wählbarer Einzelstandort für genau einen Modelllauf."""

    latitude: float
    longitude: float
    label: str = "single_site"
    country_code: str | None = None
    bidding_zone: str | None = None

    def validate(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude muss zwischen -90 und 90 Grad liegen.")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude muss zwischen -180 und 180 Grad liegen.")
        if not self.label.strip():
            raise ValueError("label darf nicht leer sein.")
        if self.country_code is not None and len(self.country_code.strip()) not in (2, 3):
            raise ValueError("country_code muss ein ISO-Alpha-2- oder Alpha-3-Code sein.")


@dataclass(frozen=True, slots=True)
class StudyConfig:
    """Systemgrenze, Nachfrage und zeitliche Modellkonventionen."""

    h2_demand_kg_per_day: ScalarParameter = field(
        default_factory=lambda: ScalarParameter(
            10_000.0,
            "kg_H2/day",
            f"{BRANDT_ARTICLE_SOURCE}, Figure 4 und Supplementary Note 2",
            2023,
            "Konstante Durchschnittsnachfrage des Basisfalls.",
        )
    )
    time_step_hours: ScalarParameter = field(
        default_factory=lambda: ScalarParameter(
            1.0,
            "h",
            f"{BRANDT_ARTICLE_SOURCE}, System set-up",
            2023,
        )
    )
    number_of_time_steps: ScalarParameter = field(
        default_factory=lambda: ScalarParameter(
            8_760.0,
            "time_steps/year",
            f"{BRANDT_ARTICLE_SOURCE}, System set-up",
            2023,
            "Ein Nicht-Schaltjahr mit stündlicher Auflösung.",
        )
    )
    functional_unit_kg_h2: ScalarParameter = field(
        default_factory=lambda: ScalarParameter(
            1.0,
            "kg_H2",
            MODEL_DEFINITION_SOURCE,
            note="Ein Kilogramm gasförmiger Wasserstoff am Ausgang des Druckspeichers.",
        )
    )
    delivery_pressure_bar: ScalarParameter = field(
        default_factory=lambda: ScalarParameter(
            300.0,
            "bar",
            BRANDT_SUPPLEMENT_SOURCE,
            2023,
            "Nenndruck des modellierten Druckgastanks; Transport und Nutzung folgen außerhalb der Systemgrenze.",
        )
    )
    weather_start_year: ScalarParameter = field(
        default_factory=lambda: ScalarParameter(
            2007.0,
            "calendar_year",
            EXISTING_TMY_SOURCE,
        )
    )
    weather_end_year: ScalarParameter = field(
        default_factory=lambda: ScalarParameter(
            2016.0,
            "calendar_year",
            EXISTING_TMY_SOURCE,
        )
    )
    profile_calendar_year: ScalarParameter = field(
        default_factory=lambda: ScalarParameter(
            2025.0,
            "calendar_year",
            EXISTING_TMY_SOURCE,
            2025,
            "Technisches Indexjahr für 8.760 TMY-Stunden; kein einzelnes Wetterjahr.",
        )
    )

    @property
    def h2_demand_kg_per_hour(self) -> float:
        return self.h2_demand_kg_per_day.value / HOURS_PER_DAY

    @property
    def h2_demand_kg_per_year(self) -> float:
        return self.h2_demand_kg_per_day.value * DAYS_PER_YEAR

    def validate(self) -> None:
        if self.h2_demand_kg_per_day.value <= 0.0:
            raise ValueError("Die H2-Nachfrage muss positiv sein.")
        if self.time_step_hours.value <= 0.0:
            raise ValueError("Die Zeitschrittlänge muss positiv sein.")
        expected_hours = self.time_step_hours.value * self.number_of_time_steps.value
        if expected_hours != HOURS_PER_YEAR:
            raise ValueError("Der Basisfall muss genau 8.760 Stunden abdecken.")
        if self.functional_unit_kg_h2.value != 1.0:
            raise ValueError("Die funktionelle Einheit muss 1 kg H2 betragen.")
        if self.delivery_pressure_bar.value <= 0.0:
            raise ValueError("Der Übergabedruck muss positiv sein.")
        if self.weather_start_year.value > self.weather_end_year.value:
            raise ValueError("weather_start_year darf nicht nach weather_end_year liegen.")


@dataclass(frozen=True, slots=True)
class GenerationTechnology:
    """Kostenannahmen einer erneuerbaren Erzeugungstechnologie."""

    name: str
    capex_eur_per_kw: ScalarParameter
    fixed_opex_eur_per_kw_year: ScalarParameter
    variable_opex_eur_per_mwh: ScalarParameter
    lifetime_years: ScalarParameter
    real_wacc_fraction: ScalarParameter

    @property
    def capex_eur_per_mw(self) -> float:
        return self.capex_eur_per_kw.value * KILOWATT_PER_MEGAWATT

    @property
    def fixed_opex_eur_per_mw_year(self) -> float:
        return self.fixed_opex_eur_per_kw_year.value * KILOWATT_PER_MEGAWATT

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Jede Erzeugungstechnologie braucht einen Namen.")
        if self.capex_eur_per_kw.value < 0.0:
            raise ValueError(f"CAPEX von {self.name} darf nicht negativ sein.")
        if self.fixed_opex_eur_per_kw_year.value < 0.0:
            raise ValueError(f"Fixe OPEX von {self.name} dürfen nicht negativ sein.")
        if self.variable_opex_eur_per_mwh.value < 0.0:
            raise ValueError(f"Variable OPEX von {self.name} dürfen nicht negativ sein.")
        _validate_lifetime_and_wacc(self.lifetime_years, self.real_wacc_fraction, self.name)


@dataclass(frozen=True, slots=True)
class ElectrolyzerConfig:
    """Vereinfachter PEM-Elektrolyseur mit konstantem Strombedarf."""

    capex_eur_per_kw: ScalarParameter
    fixed_opex_eur_per_kw_year: ScalarParameter
    lifetime_years: ScalarParameter
    real_wacc_fraction: ScalarParameter
    specific_electricity_kwh_per_kg_h2: ScalarParameter
    water_consumption_kg_per_kg_h2: ScalarParameter
    outlet_pressure_bar: ScalarParameter

    @property
    def capex_eur_per_mw(self) -> float:
        return self.capex_eur_per_kw.value * KILOWATT_PER_MEGAWATT

    @property
    def fixed_opex_eur_per_mw_year(self) -> float:
        return self.fixed_opex_eur_per_kw_year.value * KILOWATT_PER_MEGAWATT

    @property
    def specific_electricity_mwh_per_kg_h2(self) -> float:
        return self.specific_electricity_kwh_per_kg_h2.value / KILOWATT_HOURS_PER_MEGAWATT_HOUR

    @property
    def implied_lhv_efficiency_fraction(self) -> float:
        return H2_LHV_KWH_PER_KG / self.specific_electricity_kwh_per_kg_h2.value

    def validate(self) -> None:
        if self.capex_eur_per_kw.value < 0.0 or self.fixed_opex_eur_per_kw_year.value < 0.0:
            raise ValueError("Elektrolyseurkosten dürfen nicht negativ sein.")
        _validate_lifetime_and_wacc(
            self.lifetime_years, self.real_wacc_fraction, "PEM-Elektrolyseur"
        )
        if self.specific_electricity_kwh_per_kg_h2.value <= H2_LHV_KWH_PER_KG:
            raise ValueError("Der Elektrolyseur-Strombedarf muss über dem H2-Heizwert liegen.")
        if self.water_consumption_kg_per_kg_h2.value <= 0.0:
            raise ValueError("Der Wasserbedarf muss positiv sein.")
        if self.outlet_pressure_bar.value <= 0.0:
            raise ValueError("Der Elektrolyseur-Ausgangsdruck muss positiv sein.")


@dataclass(frozen=True, slots=True)
class CompressorConfig:
    """Lineare Verdichtung vom Elektrolyseur zum Druckspeicher."""

    capex_eur_per_kw: ScalarParameter
    fixed_opex_fraction_per_year: ScalarParameter
    lifetime_years: ScalarParameter
    real_wacc_fraction: ScalarParameter
    isentropic_energy_kwh_per_kg_h2: ScalarParameter
    isentropic_efficiency_fraction: ScalarParameter
    mechanical_efficiency_fraction: ScalarParameter
    h2_loss_fraction: ScalarParameter
    inlet_pressure_bar: ScalarParameter
    outlet_pressure_bar: ScalarParameter

    @property
    def capex_eur_per_mw(self) -> float:
        return self.capex_eur_per_kw.value * KILOWATT_PER_MEGAWATT

    @property
    def fixed_opex_eur_per_kw_year(self) -> float:
        return self.capex_eur_per_kw.value * self.fixed_opex_fraction_per_year.value

    @property
    def specific_electricity_kwh_per_kg_h2(self) -> float:
        denominator = (
            self.isentropic_efficiency_fraction.value
            * self.mechanical_efficiency_fraction.value
        )
        return self.isentropic_energy_kwh_per_kg_h2.value / denominator

    @property
    def specific_electricity_mwh_per_kg_h2(self) -> float:
        return self.specific_electricity_kwh_per_kg_h2 / KILOWATT_HOURS_PER_MEGAWATT_HOUR

    def validate(self) -> None:
        if self.capex_eur_per_kw.value < 0.0:
            raise ValueError("Kompressor-CAPEX dürfen nicht negativ sein.")
        if not 0.0 <= self.fixed_opex_fraction_per_year.value < 1.0:
            raise ValueError("Kompressor-OPEX-Anteil muss zwischen 0 und 1 liegen.")
        _validate_lifetime_and_wacc(
            self.lifetime_years, self.real_wacc_fraction, "Kompressor"
        )
        if self.isentropic_energy_kwh_per_kg_h2.value <= 0.0:
            raise ValueError("Der isentrope Verdichtungsbedarf muss positiv sein.")
        for efficiency in (
            self.isentropic_efficiency_fraction,
            self.mechanical_efficiency_fraction,
        ):
            if not 0.0 < efficiency.value <= 1.0:
                raise ValueError("Kompressorwirkungsgrade müssen in (0, 1] liegen.")
        if not 0.0 <= self.h2_loss_fraction.value < 1.0:
            raise ValueError("Der H2-Verlustanteil muss zwischen 0 und 1 liegen.")
        if self.inlet_pressure_bar.value <= 0.0:
            raise ValueError("Der Kompressor-Eingangsdruck muss positiv sein.")
        if self.outlet_pressure_bar.value <= self.inlet_pressure_bar.value:
            raise ValueError("Der Kompressor-Ausgangsdruck muss über dem Eingangsdruck liegen.")


@dataclass(frozen=True, slots=True)
class HydrogenStorageConfig:
    """Druckgastank, dessen Kapazität in kg H2 dimensioniert wird."""

    capex_eur_per_kg_h2: ScalarParameter
    fixed_opex_fraction_per_year: ScalarParameter
    lifetime_years: ScalarParameter
    real_wacc_fraction: ScalarParameter
    nominal_pressure_bar: ScalarParameter

    def validate(self) -> None:
        if self.capex_eur_per_kg_h2.value < 0.0:
            raise ValueError("H2-Speicher-CAPEX dürfen nicht negativ sein.")
        if not 0.0 <= self.fixed_opex_fraction_per_year.value < 1.0:
            raise ValueError("H2-Speicher-OPEX-Anteil muss zwischen 0 und 1 liegen.")
        _validate_lifetime_and_wacc(
            self.lifetime_years, self.real_wacc_fraction, "H2-Druckspeicher"
        )
        if self.nominal_pressure_bar.value <= 0.0:
            raise ValueError("Der Speicherdruck muss positiv sein.")


@dataclass(frozen=True, slots=True)
class WaterConfig:
    """Wasserpreis für die variable H2-Produktionskostenrechnung."""

    price_eur_per_m3: ScalarParameter
    density_kg_per_m3: ScalarParameter = field(
        default_factory=lambda: ScalarParameter(
            1_000.0,
            "kg_H2O/m3_H2O",
            MODEL_DEFINITION_SOURCE,
            note="Gerundete Dichte zur Umrechnung des Wasserbedarfs.",
        )
    )

    def validate(self) -> None:
        if self.price_eur_per_m3.value < 0.0:
            raise ValueError("Der Wasserpreis darf nicht negativ sein.")
        if self.density_kg_per_m3.value <= 0.0:
            raise ValueError("Die Wasserdichte muss positiv sein.")


@dataclass(frozen=True, slots=True)
class TechnologyConfig:
    pv: GenerationTechnology
    wind_onshore: GenerationTechnology
    electrolyzer: ElectrolyzerConfig
    compressor: CompressorConfig
    h2_storage: HydrogenStorageConfig
    water: WaterConfig

    def validate(self) -> None:
        self.pv.validate()
        self.wind_onshore.validate()
        self.electrolyzer.validate()
        self.compressor.validate()
        self.h2_storage.validate()
        self.water.validate()
        if self.compressor.inlet_pressure_bar.value != self.electrolyzer.outlet_pressure_bar.value:
            raise ValueError("Kompressor-Eingang und Elektrolyseur-Ausgang müssen denselben Druck haben.")
        if self.h2_storage.nominal_pressure_bar.value > self.compressor.outlet_pressure_bar.value:
            raise ValueError("Der Speicherdruck darf den Kompressor-Ausgangsdruck nicht übersteigen.")


def _p(
    value: float,
    unit: str,
    *,
    source: str = BRANDT_SUPPLEMENT_SOURCE,
    reference_year: int | None = 2023,
    note: str = "",
) -> ScalarParameter:
    return ScalarParameter(value, unit, source, reference_year, note)


def default_technology_config() -> TechnologyConfig:
    """Erzeuge den konsistenten Literatur-Basisfall in realen EUR 2023."""

    pv = GenerationTechnology(
        name="utility_scale_pv",
        capex_eur_per_kw=_p(921.0, "EUR_2023/kW"),
        fixed_opex_eur_per_kw_year=_p(15.1, "EUR_2023/(kW*year)"),
        variable_opex_eur_per_mwh=_p(
            0.0,
            "EUR_2023/MWh",
            note="Keine separaten variablen PV-OPEX in Brandt et al.; im Basisfall auf null gesetzt.",
        ),
        lifetime_years=_p(30.0, "year"),
        real_wacc_fraction=_p(0.05, "fraction"),
    )
    wind = GenerationTechnology(
        name="onshore_wind",
        capex_eur_per_kw=_p(1_779.5, "EUR_2023/kW"),
        fixed_opex_eur_per_kw_year=_p(22.64, "EUR_2023/(kW*year)"),
        variable_opex_eur_per_mwh=_p(
            9.0,
            "EUR_2023/MWh",
            note="Umrechnung aus 0,009 EUR_2023/kWh.",
        ),
        lifetime_years=_p(25.0, "year"),
        real_wacc_fraction=_p(0.05, "fraction"),
    )
    electrolyzer = ElectrolyzerConfig(
        capex_eur_per_kw=_p(1_297.0, "EUR_2023/kW"),
        fixed_opex_eur_per_kw_year=_p(20.2, "EUR_2023/(kW*year)"),
        lifetime_years=_p(30.0, "year"),
        real_wacc_fraction=_p(0.07, "fraction"),
        specific_electricity_kwh_per_kg_h2=_p(
            52.5,
            "kWh/kg_H2",
            note="Nominaler Anlagenstrombedarf einschließlich Peripherie; konstante Effizienz im vereinfachten Kern.",
        ),
        water_consumption_kg_per_kg_h2=_p(14.0, "kg_H2O/kg_H2"),
        outlet_pressure_bar=_p(30.0, "bar"),
    )
    compressor = CompressorConfig(
        capex_eur_per_kw=_p(4_577.5, "EUR_2023/kW"),
        fixed_opex_fraction_per_year=_p(0.04, "fraction/year"),
        lifetime_years=_p(15.0, "year"),
        real_wacc_fraction=_p(0.07, "fraction"),
        isentropic_energy_kwh_per_kg_h2=_p(1.287, "kWh/kg_H2"),
        isentropic_efficiency_fraction=_p(0.80, "fraction"),
        mechanical_efficiency_fraction=_p(0.90, "fraction"),
        h2_loss_fraction=_p(0.005, "fraction"),
        inlet_pressure_bar=_p(30.0, "bar"),
        outlet_pressure_bar=_p(350.0, "bar"),
    )
    storage = HydrogenStorageConfig(
        capex_eur_per_kg_h2=_p(733.5, "EUR_2023/kg_H2"),
        fixed_opex_fraction_per_year=_p(0.02, "fraction/year"),
        lifetime_years=_p(25.0, "year"),
        real_wacc_fraction=_p(0.07, "fraction"),
        nominal_pressure_bar=_p(300.0, "bar"),
    )
    water = WaterConfig(price_eur_per_m3=_p(3.74, "EUR_2023/m3_H2O"))
    return TechnologyConfig(pv, wind, electrolyzer, compressor, storage, water)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Vollständige Konfiguration eines H2-Modelllaufs ohne Standortdaten."""

    scenario: Scenario = Scenario.REFERENCE
    paths: PathConfig = field(default_factory=PathConfig)
    study: StudyConfig = field(default_factory=StudyConfig)
    technologies: TechnologyConfig = field(default_factory=default_technology_config)

    def validate(self) -> None:
        if not isinstance(self.scenario, Scenario):
            raise ValueError("scenario muss ein Wert der Enum Scenario sein.")
        self.paths.validate()
        self.study.validate()
        self.technologies.validate()
        if (
            self.study.delivery_pressure_bar.value
            != self.technologies.h2_storage.nominal_pressure_bar.value
        ):
            raise ValueError("Übergabedruck und Nenndruck des H2-Speichers müssen übereinstimmen.")


def _validate_lifetime_and_wacc(
    lifetime_years: ScalarParameter,
    real_wacc_fraction: ScalarParameter,
    component_name: str,
) -> None:
    if lifetime_years.value <= 0.0:
        raise ValueError(f"Die Lebensdauer von {component_name} muss positiv sein.")
    if not 0.0 <= real_wacc_fraction.value < 1.0:
        raise ValueError(f"Der reale WACC von {component_name} muss zwischen 0 und 1 liegen.")


def iter_scalar_parameters(config: object, prefix: str = "") -> Iterator[tuple[str, ScalarParameter]]:
    """Durchlaufe alle dokumentierten Skalare einer verschachtelten Konfiguration."""

    if isinstance(config, ScalarParameter):
        yield prefix, config
        return
    if not is_dataclass(config) or isinstance(config, type):
        return
    for data_field in fields(config):
        child = getattr(config, data_field.name)
        child_prefix = f"{prefix}.{data_field.name}" if prefix else data_field.name
        yield from iter_scalar_parameters(child, child_prefix)


def default_model_config() -> ModelConfig:
    """Liefere eine validierte, unveränderliche Standardkonfiguration."""

    config = ModelConfig()
    config.validate()
    return config


def with_uniform_real_wacc(
    config: ModelConfig,
    real_wacc_fraction: float,
    *,
    source: str,
    reference_year: int | None = None,
    note: str = "",
) -> ModelConfig:
    """Überschreibe den realen WACC aller Investitionskomponenten nachvollziehbar.

    Das ursprüngliche Ammoniakmodell verwendet je Standort einen gemeinsamen
    Länder- beziehungsweise Regional-WACC. Diese Hilfsfunktion ermöglicht
    denselben Ansatz für einen konkreten H2-Fall, ohne den übertragbaren
    Literatur-Basisfall in ``DEFAULT_CONFIG`` zu verändern.
    """

    if isinstance(real_wacc_fraction, bool) or not isinstance(
        real_wacc_fraction, (int, float)
    ):
        raise TypeError("real_wacc_fraction muss eine reelle Zahl sein.")
    value = float(real_wacc_fraction)
    if not 0.0 <= value < 1.0:
        raise ValueError("real_wacc_fraction muss zwischen 0 und 1 liegen.")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Für die WACC-Überschreibung ist eine Quelle erforderlich.")

    parameter = ScalarParameter(
        value=value,
        unit="fraction",
        source=source.strip(),
        reference_year=reference_year,
        note=note,
    )
    technology = config.technologies
    updated_technology = replace(
        technology,
        pv=replace(technology.pv, real_wacc_fraction=parameter),
        wind_onshore=replace(technology.wind_onshore, real_wacc_fraction=parameter),
        electrolyzer=replace(
            technology.electrolyzer, real_wacc_fraction=parameter
        ),
        compressor=replace(technology.compressor, real_wacc_fraction=parameter),
        h2_storage=replace(technology.h2_storage, real_wacc_fraction=parameter),
    )
    updated = replace(config, technologies=updated_technology)
    updated.validate()
    return updated


CONFIG_SENSITIVITY_PARAMETERS: Final[tuple[str, ...]] = (
    "electrolyzer_capex_eur_per_kw",
    "electrolyzer_specific_electricity_kwh_per_kg_h2",
    "pv_capex_eur_per_kw",
    "wind_capex_eur_per_kw",
    "h2_storage_capex_eur_per_kg_h2",
    "uniform_real_wacc_fraction",
)


def with_sensitivity_parameter(
    config: ModelConfig,
    parameter_name: str,
    value: float,
    *,
    source: str,
    note: str = "",
) -> ModelConfig:
    """Erzeuge eine validierte Konfigurationskopie für einen Sensitivitätslauf.

    Die Funktion verändert niemals ``config``. Sie akzeptiert ausschließlich
    die ausdrücklich freigegebenen Parameter aus
    ``CONFIG_SENSITIVITY_PARAMETERS`` und erhält Einheit sowie Bezugsjahr des
    jeweiligen Basisparameters.
    """

    if not isinstance(parameter_name, str) or not parameter_name.strip():
        raise ValueError("parameter_name darf nicht leer sein.")
    normalized_name = parameter_name.strip()
    if normalized_name not in CONFIG_SENSITIVITY_PARAMETERS:
        allowed = ", ".join(CONFIG_SENSITIVITY_PARAMETERS)
        raise ValueError(
            f"Unbekannter Konfigurations-Sensitivitätsparameter "
            f"'{normalized_name}'. Erlaubt: {allowed}."
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Der Sensitivitätswert muss eine reelle Zahl sein.")
    numeric_value = float(value)
    if not isfinite(numeric_value):
        raise ValueError("Der Sensitivitätswert muss endlich sein.")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Für den Sensitivitätswert ist eine Quelle erforderlich.")

    if normalized_name == "uniform_real_wacc_fraction":
        return with_uniform_real_wacc(
            config,
            numeric_value,
            source=source,
            note=note,
        )

    technology = config.technologies
    if normalized_name == "electrolyzer_capex_eur_per_kw":
        original = technology.electrolyzer.capex_eur_per_kw
        updated_electrolyzer = replace(
            technology.electrolyzer,
            capex_eur_per_kw=_sensitivity_scalar(
                original, numeric_value, source=source, note=note
            ),
        )
        updated_technology = replace(
            technology, electrolyzer=updated_electrolyzer
        )
    elif normalized_name == "electrolyzer_specific_electricity_kwh_per_kg_h2":
        original = technology.electrolyzer.specific_electricity_kwh_per_kg_h2
        updated_electrolyzer = replace(
            technology.electrolyzer,
            specific_electricity_kwh_per_kg_h2=_sensitivity_scalar(
                original, numeric_value, source=source, note=note
            ),
        )
        updated_technology = replace(
            technology, electrolyzer=updated_electrolyzer
        )
    elif normalized_name == "pv_capex_eur_per_kw":
        original = technology.pv.capex_eur_per_kw
        updated_technology = replace(
            technology,
            pv=replace(
                technology.pv,
                capex_eur_per_kw=_sensitivity_scalar(
                    original, numeric_value, source=source, note=note
                ),
            ),
        )
    elif normalized_name == "wind_capex_eur_per_kw":
        original = technology.wind_onshore.capex_eur_per_kw
        updated_technology = replace(
            technology,
            wind_onshore=replace(
                technology.wind_onshore,
                capex_eur_per_kw=_sensitivity_scalar(
                    original, numeric_value, source=source, note=note
                ),
            ),
        )
    else:
        original = technology.h2_storage.capex_eur_per_kg_h2
        updated_technology = replace(
            technology,
            h2_storage=replace(
                technology.h2_storage,
                capex_eur_per_kg_h2=_sensitivity_scalar(
                    original, numeric_value, source=source, note=note
                ),
            ),
        )

    updated = replace(config, technologies=updated_technology)
    updated.validate()
    return updated


def _sensitivity_scalar(
    original: ScalarParameter,
    value: float,
    *,
    source: str,
    note: str,
) -> ScalarParameter:
    return ScalarParameter(
        value=value,
        unit=original.unit,
        source=source.strip(),
        reference_year=original.reference_year,
        note=note,
    )


DEFAULT_CONFIG: Final[ModelConfig] = default_model_config()


__all__ = [
    "BRANDT_ARTICLE_SOURCE",
    "BRANDT_SUPPLEMENT_SOURCE",
    "CONFIG_SENSITIVITY_PARAMETERS",
    "DAYS_PER_YEAR",
    "DEFAULT_CONFIG",
    "H2_LHV_KWH_PER_KG",
    "H2_LHV_MJ_PER_KG",
    "HOURS_PER_DAY",
    "HOURS_PER_YEAR",
    "ModelConfig",
    "PathConfig",
    "REPOSITORY_ROOT",
    "ScalarParameter",
    "Scenario",
    "SiteConfig",
    "StudyConfig",
    "TechnologyConfig",
    "default_model_config",
    "default_technology_config",
    "iter_scalar_parameters",
    "with_uniform_real_wacc",
    "with_sensitivity_parameter",
]
