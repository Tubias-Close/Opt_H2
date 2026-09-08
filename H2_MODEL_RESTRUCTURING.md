# Umstrukturierung zum Ein-Standort-Hydrogen-Modell

## Ziel

Dieses Repository bleibt ein Fork des ursprünglichen Ammoniak-Modells. Auf einem eigenen Branch entsteht daraus ein schlankes Modell zur kosten- und emissionsoptimierten Wasserstoffproduktion an genau einem räumlichen Standort (einem Pixel).

Die Forschungsfrage lautet:

> Welche Kombination aus PV, Wind, Batterie, Elektrolyse, Wasserstoffspeicher und Netzanschluss minimiert die Kosten und die Klimawirkung der Wasserstoffproduktion an einem gegebenen Standort?

## Git-Organisation

- Der Fork `MericVural/Opt_H2_Meric` bleibt das gemeinsame GitHub-Repository.
- `main` bewahrt den aktuellen Ammoniak-Modellstand als technische Referenz.
- Die H2-Entwicklung erfolgt auf `codex/h2-single-site`.
- Nach einer überprüften Version kann der Branch per Pull Request nach `main` zusammengeführt werden.

```powershell
git switch -c codex/h2-single-site
git push -u origin codex/h2-single-site
```

## Fachlicher Umfang

### Bestandteil des H2-Modells

- ein Standort: Name, ISO2-Code, Breitengrad und Längengrad;
- stündliche Zeitreihen über ein vollständiges Jahr (8.760 Stunden);
- PV- und Windkapazitätsfaktoren für genau diesen Standort;
- Netzstrompreis, Netz-THG-Intensität und länderspezifischer WACC;
- Elektrolyse, Batterie, H2-Speicher, PV, Wind und optionaler Netzanschluss;
- Netz-, Hybrid- und Inselbetriebs-Szenarien;
- Kosten- und Klimawirkungsbewertung;
- Mehrzieloptimierung und Pareto-Auswertung.

### Nicht Bestandteil

- Haber-Bosch-Synthese;
- Luftzerlegung (ASU) und Stickstoffbilanz;
- synthetischer Stickstoffbedarf und globale Stickstoffkarte;
- globale Rasteroptimierung;
- Ammoniak-spezifische Kosten, Emissionsfaktoren, LCA-Prozesse und Abbildungen.

## Zielgrößen und Einheiten

Die Ergebnisgrößen müssen spezifisch auf Wasserstoff bezogen sein:

| Größe | Einheit |
|---|---|
| Levelized cost of hydrogen (LCOH) | EUR/kg H2 |
| Klimawirkung | kg CO2-eq/kg H2 |
| Jahresproduktion | kg H2/a oder t H2/a |
| Elektrolyse- und Erzeugerkapazität | MW |
| Batterie- und H2-Speicherkapazität | MWh |
| Netzbezug und erneuerbare Erzeugung | MWh/a |

Eine jährliche H2-Zielmenge wird als exogene Vorgabe benötigt. Ohne eine Nachfragevorgabe oder einen H2-Verkaufspreis wäre die kostenminimale Lösung eine Anlage mit null Produktion.

## Modellformulierung

### Entscheidungsvariablen

Installierte Kapazitäten:

- PV und Wind in MW;
- Batterieenergie in MWh und Batterieleistung in MW;
- Elektrolyse in MW;
- H2-Speicher in MWh H2;
- Netzanschluss in MW.

Stündliche Variablen:

- Netzbezug und gegebenenfalls Netzeinspeisung;
- PV- und Winderzeugung;
- Batterie-Laden, Batterie-Entladen und Ladezustand;
- elektrische Aufnahme der Elektrolyse;
- H2-Erzeugung, H2-Auslieferung und H2-Speicherstand.

### Zentrale Nebenbedingungen

Strombilanz je Stunde:

```text
Netz + PV + Wind + Batterieentladung
= Elektrolyse + Batterieladung + Einspeisung
```

H2-Bilanz je Stunde:

```text
H2 aus Elektrolyse = H2-Auslieferung + Änderung des H2-Speichers
```

Weitere Bedingungen sind:

- Wetterprofil begrenzt PV- und Winderzeugung;
- Batterie- und H2-Speichergrenzen sowie zyklische Jahresrandbedingung;
- Wirkungsgrade von Batterie und Elektrolyse;
- Kapazitätsgrenzen;
- jährliche H2-Auslieferung entspricht der festgelegten Zielmenge;
- im Inselbetrieb ist der Netzanschluss null;
- im reinen Netzszenario sind PV- und Windkapazitäten null.

## Kosten und Klimawirkung

Die Jahreskosten umfassen Investitionsannuitäten, Ersatzinvestitionen, Betrieb und Wartung sowie Kosten des Netzstrombezugs. Aus ihnen folgt:

```text
LCOH = annualisierte Gesamtkosten / ausgelieferte Jahresmenge H2
```

Die Klimawirkung umfasst Netzstrom sowie die über Lebensdauer annualisierten Herstellungswirkungen von PV, Wind, Batterie, Elektrolyse, H2-Speicher und Netzanschluss:

```text
GHG_H2 = jährliche Gesamtemissionen / ausgelieferte Jahresmenge H2
```

## Kosten- und Umweltoptimierung

Kosten und Emissionen werden nicht unnormalisiert mit willkürlichen Gewichten addiert. Stattdessen wird eine Pareto-Kurve berechnet:

1. Kostenminimum ohne zusätzliche Emissionsobergrenze.
2. Emissionsminimum.
3. Kostenminimum unter mehreren THG-Obergrenzen zwischen beiden Extremen.

Jeder Punkt zeigt damit den Zielkonflikt zwischen LCOH und Klimawirkung. Die Auswertung berichtet Kapazitäten und stündliche Betriebszeitreihen für ausgewählte Pareto-Punkte.

## Vorgesehene Code-Struktur

```text
config_h2.py                  Standort, Nachfrage, Szenarien und Dateipfade
opt_hydrogen_functions.py     Gurobi-MILP für reines H2-System
run_single_site_h2.py         Erzeugt Eingabedaten und startet Szenarien/Pareto-Läufe
renewable_profiles.py         PV- und Windprofile für den Standort
energy_data_processor.py      Strompreise, WACC und Netz-THG; nur relevante Funktionen
plot_h2_results.py            Pareto-Kurve, Kapazitäten und Betriebszeitreihen
input_data/                   Kosten-, Preis- und Umweltfaktoren
results/                      Ergebnis-Tabellen und Zeitreihen
figs/                         Abbildungen
```

Der bestehende Ammoniak-Code bleibt zunächst unverändert. Die neuen H2-Dateien werden neben ihm aufgebaut. Erst nach vollständiger Validierung kann über das Entfernen nicht mehr benötigter Dateien entschieden werden.

## Umsetzungsreihenfolge

1. Standort, jährliche H2-Zielmenge und betrachtete Szenarien festlegen.
2. `config_h2.py` mit eindeutigen Einheiten und Eingabedateien erstellen.
3. Den H2-Optimierer ohne Haber-Bosch- und ASU-Komponenten implementieren.
4. Einen Ein-Standort-Runner mit PVGIS-Profilen, Strompreis, WACC und Netz-THG erstellen.
5. Kostenminimum und Emissionsminimum prüfen.
6. Pareto-Punkte mit THG-Obergrenzen berechnen.
7. Kapazitäten, LCOH, Emissionen und stündlichen Betrieb auswerten.
8. Optional: H2-spezifische Brightway-LCA ergänzen.

## Validierung

Vor wissenschaftlicher Auswertung werden mindestens diese Prüfungen durchgeführt:

- Strom- und H2-Bilanzen schließen in jeder Stunde.
- Jahresproduktion entspricht exakt der H2-Zielmenge.
- Im Inselbetrieb ist kein Netzstrombezug möglich.
- Im Netzszenario werden keine erneuerbaren Anlagen gebaut.
- Emissionsminimum darf keine höhere spezifische Klimawirkung als Kostenminimum haben.
- Pareto-Punkte müssen bei strengerer THG-Grenze nicht fallende Kosten aufweisen.
- Einheiten der Kosten- und Emissionsfaktoren werden gegen die Excel-Quelldaten dokumentiert.

## Noch festzulegende Eingaben

- Standort: Name, ISO2, Breitengrad, Längengrad;
- H2-Zielmenge in kg/a oder t/a;
- Zieljahr und Kostenbasis;
- zulässiger Netzbezug und mögliche Netzeinspeisung;
- ob eine vollständige LCA erforderlich ist oder zunächst nur Klimawirkung betrachtet wird.
