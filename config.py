import pandas as pd

"""Define vars"""
DAYS = 365
MJ_KG_H2 = 120
# Exchange rate used: 1 USD = 0.86 EUR (approx., Oct 2025 snapshot)
USD_TO_EUR = 0.86
MJ_kWh = 3.6

# Resoltuion of geospatial analysis:
DEG_RES = 1
T_DAY = 20  # tonnes NH3 production per day

ASSESSMENT_YEAR = 2025

##LOCATIONS selected
# country, iso2, latitude, longitude.
LOCATIONS = [
    [
        "Namibia (Erongo region)",
        "NA",
        -21.0,
        -14.0,
    ],  # Daures Green Hydrogen Village / Namibia
    # ["Netherlands (Schagen)", "NL", 52.75, 4.80],                         # Schagen / NW Netherlands
    # ["Spain (J. de la Frontera)", "ES", 36.85, -6.10],                 # Jerez / Andalucía
    # ["Argentina (Bahía Blanca)", "AR", -38.72, -62.27],          # Bahía Blanca / Pampa
    # ["Brazil (Rio region)", "BR", -22.50, -42.10],               # Macaé / nearby Rio de Janeiro
    # ["China (Shandong)", "CN", 37.20, 118.90],            # Shandong (central/east)
    # ["Australia (Victoria)", "AU", -37.90, 143.60], # Victoria: near Geelong / Melbourne region
    # ["USA (Midwest)", "US", 41.00, -95.20],     # Omaha / SW Iowa region
    # ["South Africa (near Bloemfontein)", "ZA", -28.50, 27.20], # Free State / near Bloemfontein (regional)
    # ["Nigeria (Kano region)", "NG", 12.00, 8],                        # Kano / northern Nigeria
    # ["India (Punjab)", "IN", 30.90, 75.86],                   # Ludhiana / Punjab
]

LOCATIONS_SENS = [
    [
        "Namibia (Erongo region)",
        "NA",
        -21.0,
        -14.0,
    ],  # Daures Green Hydrogen Village / Namibia
    # ["Netherlands (Schagen)", "NL", 52.75, 4.80],                         # Schagen / NW Netherlands
    # ["Spain (J. de la Frontera)", "ES", 36.85, -6.10],                 # Jerez / Andalucía
    # ["Argentina (Bahía Blanca)", "AR", -38.72, -62.27],          # Bahía Blanca / Pampa
    # [#"Brazil (Rio region)", "BR", -22.50, -42.10],               # Macaé / nearby Rio de Janeiro
    # ["China (Shandong)", "CN", 37.20, 118.90],            # Shandong (central/east)
    # ["Australia (Victoria)", "AU", -37.90, 143.60], # Victoria: near Geelong / Melbourne region
    # ["USA (Midwest)", "US", 41.00, -95.20],     # Omaha / SW Iowa region
    # ["South Africa (near Bloemfontein)", "ZA", -28.50, 27.20], # Free State / near Bloemfontein (regional)
    # ["Nigeria (Kano region)", "NG", 12.00, 8.59],                        # Kano / northern Nigeria
    # ["India (Punjab)", "IN", 30.90, 75.86],                   # Ludhiana / Punjab
]

LOCATIONS_SENS_PLUS = LOCATIONS_SENS + [
    ["Namibia (Erongo region)", "NA", -21.0, -14.0]
]

# KEYS
# You need a key for premise to generate prospective LCA databases
"""YOU WILL ALSO NEED TO HAVE A LOCAL LICENSE KEY FOR GUROBI, see: https://www.gurobi.com/solutions/licensing/ & /
    https://support.gurobi.com/hc/en-us/articles/12872879801105-How-do-I-retrieve-and-set-up-a-Gurobi-license-
"""

# Output file names
OUTPUT_FILE_XARRAY_INIT = f"processed_data/output_dataset_res_{DEG_RES}.pkl"
OUTPUT_FILE_XARRAY = f"processed_data/ds_processed_res_{DEG_RES}.pkl"

OUT_JSON_GHG = "input_data/ghg_factors.json"
OUT_JSON_GHG_FUTURE = "input_data/ghg_factors_future.json"
OUT_JSON_POWER_PRICES = "input_data/gpp_2025_country_pages_numeric.json"

FUTURE_POWER_PRICES = "input_data/lcoe_Liu_et_al_2025.xlsx"

# results
FILE_PATH_GLOBAL_RESULTS = f"results/global_results_new_{DEG_RES}.pkl"
FILE_PATH_GLOBAL_RESULTS_GRID = f"results/global_results_grid_{DEG_RES}.pkl"
TEMP_FILE = f"results/global_results_{DEG_RES}_TEMP.pkl"  # same output file

# Case study results
FILE_PATH_CASE_STUDIES = "results/case_studies.pkl"
FILE_PATH_CASE_STUDIES_LCA = "results/case_studies_lca.pkl"
FILE_PATH_SENS_ANALYSIS = "results/sensitivity_analysis_case_studies.pkl"

# Get cost data.
FILE_NAME_COSTS = r"input_data/technology_costs.xlsx"
COST_DATA = pd.read_excel(
    FILE_NAME_COSTS,
    sheet_name="costs",
    index_col="Parameter",
    usecols=[0, 1, 2],
    nrows=145,
)

# Decide on maximum capacity of technologies
N_DEMAND_THRESHOLD = 0  # example threshold
MAX_CAP_TECHS = 150  # MW
MAX_GRID_CAP = 100  # MW, Max grid connection capacity
MIN_ELECTROLYZER_CAP = (
    0  # MW, if installed, the minimum capacity installed of the electrolyzer.
)

# If using a brightway project when calculating LCA impacts yourself
# This can be enabled in case:
# 1. brightway2 and premise are installed
# 2. One has access to ecoinvent.
# Alternatively, one can set to False, uses exported life cycle GHG emission data (without other impacts)
CALC_ALL_LCA_IMPACTS = True
GENERATE_NEW_LCA_DB = True

USER_NAME = "ifuham1028"
PROJECT_NAME = "ecoinvent-3.12-cutoff_v2"
DB_NAME = "ecoinvent-3.12-cutoff_remind-SSP1-PkBudg650_2030_08_full"  # Temp DB name for LCA activity
EI_VERSION = "3.12"
NAME_REF_DB = "ecoinvent_{}_reference".format(EI_VERSION).replace(".", "")
NAME_FUTURE_DB = "ecoinvent-3.12-cutoff_remind-SSP1-PkBudg650_2050_08_full"
DB_NAME_INIT = "ecoinvent-{}-cutoff".format(EI_VERSION)
BIOSPHERE_DB = "ecoinvent-{}-biosphere".format(EI_VERSION)

CC_IMPACT_NG_NH3 = 2.8  # tCO2/tNH3 #based on Boyce et al. https://pdf.sciencedirectassets.com/313379/1-s2.0-S2405844023X0019X/1-s2.0-S2405844024035783/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIFuXGB8lniv3DflHoC%2Ffzw9MYWBjUr2x3DP5LHSAcEdCAiEAtlH3YoHBi%2BeEQ0tZEcXXPm1L0ubBLULSF3AprYE3hHgqvAUIrv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDEl3pXSg13mNkpGh5yqQBc0hMXTK5GT6Itb6MXFks2lU3qlC28S358m1gxwJxvULcuasAI5wg%2BQGa7I6FwTWktsLriCtNBzaSX4g5OveiLDyl4p9QR61SVjRZDG6etcVCcgiIAzoIJtP0ajEx%2BcqCCeL4kwbQSO4fm81cR1kCzyM4juV%2FJsxKaLS1lTkBbdIPRttVolV4JJeAQu4Eqrz43r4PB6%2Ftisyi1exSjOVsdd1%2BSGK0H%2F2FN81TqXoUlmoPz6B%2FWJXB0UoXBt6tGl06aKn%2Bu9WYyKEVCTz%2BpMfJegtUkT5qDEi%2BXU0VWphed%2BSGtpNWe%2Bpj7p9znWyZvtaRj6BowFrjMP1AGRzEdMEaspSEuOpokJV2vFjx7StEN02t%2F0XeIztyENRpyvGSvgVY%2FJuqMyKe7rjxWf37nB8vo42BK2E2OYejw9u%2BywhvN%2Bbi3oDw1%2BoyqtsKkoK%2B2M%2FYxUUTSzeI%2BS0fPBFNj36nasRJxD8fQlebya0kJZijpj6gnZqMu6%2BHqLM70eZO%2Fcxrz9PHgsgV%2B5Cete24ehrJj03aLOzcAEFtgT4kPhXHgLF6pxfS%2FfVGF7YnEPA8Yv3SSGsBnmtyF8H12jjTYO6DcxI1uprHMgDBX2WUC0eaP4oZrHVAqTzsbBlmlkDrwvRZ5v4Z6t7pjig1xCR6g8X8BhyCIilA%2BX6sooaeCpzJjMpH77xxGggpNH7CURDbOz4yCLrqwooqlMW8PQB6gE%2BnTVGMWX3sO4hJtI1qgZzEFFT6L1zYvj38XBSLFligd1ioDyVT71GUc7PUWcxTvAIyG1%2F%2BXxi2hS3mA71mf86HjXSAFrhso4gEY%2FQlZjP%2BZcQpBMqwC6vwtXuEkWINh3RGjL7wOwQ3TKmtTF7JOV6pbHCMNi6%2F8cGOrEBtu1FJh3JbHgHzBoebuv1WBZwey%2F9KSWvtX1RD8jXOvulxTG1DrewsBa9ANkYbcZ0OewZtC3yCB9BBQMipwXmBuwOeOYgAyqPFktTgzcpXZ1DvEEZGDW49mXD%2B%2FI8%2FwtBKTGgp%2FhW3Zi8u6QZrEdiYwBqcp3RmxpBWIhH9AsXL2O5aePEsyeoUdNhQluZ6PRuYypnrQcCrblvCe1OngdGEux%2FKQICQPOkhkJabQg9nzEB&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251027T223419Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQFU4NIPU%2F20251027%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=1c5ac10db3dd4e18f90d316296c6d2b8103855fce12f24fb147905ffba47608d&hash=74426e2a552e89e6cc6941721fca456183e1c03cbaf9e4b5b9c5598ef8c9f236&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2405844024035783&tid=spdf-3a6704dd-c916-43e2-861c-de7b965ab484&sid=f6f12eb61d3cd046527b86268de828a92247gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=15165a5d5750015e04&rr=9955a8fdba572716&cc=us
CC_METHOD = (
    "ecoinvent-3.12",
    "IPCC 2021 (incl. biogenic CO2)",
    "climate change: total (incl. biogenic CO2, incl. SLCFs)",
    "global warming potential (GWP100)",
)  # standard method to calculate CC impacts.

NAME_CC_COL = f"lca_impact_{CC_METHOD[-1]}"  # name of the climate change method we are using, to be used in the contribution arrays and post-processing
