import uuid
from functools import partial

import bw2calc as bc
import bw2io as bi
import numpy as np
from premise import *


class CheckedGMRESLCA(bc.JacobiGMRESLCA):
    """Use GMRES and reject solutions that fail the residual tolerance."""

    def solve_linear_system(self, demand=None):
        rhs = self.demand_array if demand is None else demand
        solution = super().solve_linear_system(demand=rhs)

        residual = np.linalg.norm(self.technosphere_matrix @ solution - rhs)
        tolerance = max(self.atol, self.rtol * np.linalg.norm(rhs))

        if (
            not np.all(np.isfinite(solution))
            or not np.isfinite(residual)
            or residual > tolerance
        ):
            # Do not reuse an unconverged solution as a starting guess.
            self.guess = None
            raise RuntimeError(
                "GMRES failed to converge: "
                f"residual={residual:.3e}, "
                f"required<={tolerance:.3e}, "
                f"maxiter={self.maxiter}, restart={self.restart}"
            )

        return solution


from collections import defaultdict

import bw2data as bd
import pandas as pd
from bw2io.importers.base_lci import LCIImporter

# Name for the MES to be created
from bw2io.strategies import add_database_name, csv_restore_tuples

from config import (
    ASSESSMENT_YEAR,
    BIOSPHERE_DB,
    CC_METHOD,
    DB_NAME,
    DB_NAME_INIT,
    EI_VERSION,
    MJ_KG_H2,
    NAME_REF_DB,
    PROJECT_NAME,
    USER_NAME,
    MJ_kWh,
)
from mapping import contribution_mapping_system, my_methods  # import mappings
from private_keys import KEY_PREMISE, USER_PW

FOREGROUND_DB = "db_ammonia_system"


def import_additional_lcias():
    """
    Import additional LCIA methods related to water consumption and land transformation.

    This function imports and applies LCIA methods for water consumption and creates a new environmental impact
    category for land transformation. It utilizes functions from the 'premise_gwp' and 'bw_recipe_2016' packages.

    Steps:
    1. Adds premise global warming potential (GWP) methods using 'add_premise_gwp'.
    2. Retrieves the biosphere database using 'get_biosphere_database'.
    3. Creates and applies LCIA methods for water consumption.
    4. Defines a new `LCIA' method for land transformation and writes impact factors.

    Returns:
    None

    Example:
    >>> import_additional_lcias()
    """
    # define project
    bd.projects.set_current(PROJECT_NAME)  # Creating/accessing the project

    """
    bd.bw2setup() #Importing elementary flows, LCIA methods and some other data

    # Step 1: Add premise global warming potential (GWP) methods
    add_premise_gwp()

    # Step 2: Retrieve the biosphere database
    biosphere = get_biosphere_database()

    # Step 3: Create and apply LCIA methods for water consumption
    gw = WaterConsumption(None, biosphere)
    gw.apply_strategies()
    gw.write_methods(overwrite=True)
    gw.data[0]

    # Step 4: Create a new LCIA method for land transformation
    my_cfs_land = []

    for bio in Database("biosphere3"):
        if "Transformation, from" in bio['name'] and "square meter" == bio['unit']:   
            line = (bio.key, 1)
            my_cfs_land.append(line)

    my_method = Method(("Own method", "Land transformation", "Land transformation"))
    my_metadata = {"unit": "m2", "meaning": "to represent land transformation"}
    my_method.register(**my_metadata)
    my_method.write(my_cfs_land)
    """


def import_ecoinvent_database(db_name=DB_NAME_INIT):
    """
    Import an Ecoinvent database and create default LCIA methods if not already imported.

    This function imports an Ecoinvent database using the specified database name and location path.
    It checks whether the database is already imported and, if not, imports it, applies strategies,
    provides statistics, and writes the database. Additionally, it creates default LCIA methods and core migrations.

    Parameters:
    - db_name (str): Database name for the Ecoinvent dataset.
    - location_path (str): Location path to the datasets subfolder of the unzipped Ecoinvent file.
    - overwrite (bool, optional): If True, overwrites existing LCIA methods. Default is False.

    Returns:
    None

    Example:
    >>> import_ecoinvent_database("ecoinvent_init", "/path/to/ecoinvent/datasets", overwrite=True)
    """
    bd.projects.set_current(PROJECT_NAME)
    # Check if the database is already imported
    if db_name in bd.databases:
        print(f"{db_name} has already been imported.")
    else:
        bi.import_ecoinvent_release(EI_VERSION, "cutoff", USER_NAME, USER_PW)
        # Import and process the Ecoinvent database
        # ei_importer = bd.SingleOutputEcospold2Importer(location_path, db_name)
        # ei_importer.apply_strategies()
        # ei_importer.statistics()
        # ei_importer.write_database()
        # bw2io.create_default_lcia_methods(overwrite=overwrite)
        # bw2io.create_core_migrations()


def generate_future_ei_dbs(
    scenarios=["SSP2-Base", "SSP2-PkBudg1150", "SSP2-PkBudg500"],
    iam="remind",
    start_yr=2025,
    end_yr=2050,
    step=15,
    endstring="base",
):
    """
    Generate Ecoinvent scenario models with specified parameters.

    This function generates Ecoinvent scenario models based on the specified year, scenario, and additional settings.
    It avoids adding duplicated databases by checking the existing databases in Brightway2.

    Parameters:
    - scenarios (list): The scenarios for which the models are generated. Default is:
                ["SSP2-Base",
                "SSP2-PkBudg1150",
                "SSP2-PkBudg500"] corresponding to baseline, 2 degrees C, and 1.5 degrees C.
    - iam (str): IAM chosen, can be 'remind' or 'image'.
    - start_yr (int): The starting year for the scenarios.
    - end_yr (int): The end year for the scenarios.
    - step (int): step between scenario years.
    - endstring (str, optional): A suffix to differentiate the generated databases. Default is "base".

    Returns:
    tuple: A tuple containing two lists -
        1. List of dictionaries specifying the models for the scenarios.
        2. List of database names generated based on the specified parameters.

    Example:
    >>> generate_future_ei_dbs("SSP2-Base")
    ([{'model': 'remind', 'pathway': 'SSP2-Base', 'year': 2030},
      {'model': 'remind', 'pathway': 'SSP2-Base', 'year': 2050}],
     ['ecoinvent_remind_SSP2-Base_2030_custom', 'ecoinvent_remind_SSP2-Base_2050_base'])
    """

    list_years = [
        start_yr + i * step
        for i in range(1, int((end_yr - start_yr) / step) + 1)
    ]

    list_spec_scenarios = []
    list_names = []

    for pt in scenarios:
        for yr in list_years:
            string_db = "ecoinvent_{}_{}_{}_{}".format(iam, pt, yr, endstring)

            if yr == start_yr and pt == "SSP2-Base":
                dict_spec = {
                    "model": iam,
                    "pathway": pt,
                    "year": yr,
                    "exclude": [
                        "update_electricity",
                        "update_cement",
                        "update_steel",
                        "update_dac",
                        "update_fuels",
                        "update_emissions",
                        "update_two_wheelers" "update_cars",
                        "update_trucks",
                        "update_buses",
                    ],
                }

                if string_db not in bd.databases:
                    list_spec_scenarios.append(dict_spec)
                    list_names.append(string_db)
                else:
                    print(
                        "Avoid duplicated db and therefore following db not added: '{}'".format(
                            string_db
                        )
                    )
            else:
                dict_spec = {"model": iam, "pathway": pt, "year": yr}

                if string_db not in bd.databases:
                    list_spec_scenarios.append(dict_spec)
                    list_names.append(string_db)
                else:
                    print(
                        "Avoid duplicated db and therefore following db not added: '{}'".format(
                            string_db
                        )
                    )

    return list_spec_scenarios, list_names


# ### generate the database which we are going to use, as premise include many novel datasets. Add some datasets that we generated ourselves.
def generate_reference_database():
    """
    Generate a reference database based on specified parameters.

    This function generates a reference database based on the specified parameters.
    It deletes the existing reference database with the same name if it exists and then creates a new one.

    Returns:
    None

    Example:
    >>> generate_reference_database()
    """
    bd.projects.set_current(PROJECT_NAME)
    clear_cache()
    # Delete old reference database with the same name
    for db_name in list(bd.databases):
        if NAME_REF_DB in db_name:
            print(
                "DB already exists, skipping creation of new reference database. If you want to create a new one, please delete the existing one with name '{}'".format(
                    db_name
                )
            )
            # del bd.databases[db_name]

    if NAME_REF_DB not in list(bd.databases):
        # Create a new reference database using NewDatabase
        ndb = NewDatabase(
            scenarios=[
                {
                    "model": "remind",
                    "pathway": "SSP2-NPi",
                    "year": "2025",
                    "exclude": [
                        "update_electricity",
                        "update_cement",
                        "update_steel",
                        "update_dac",
                        "update_fuels",
                        "update_emissions",
                    ],
                }
            ],
            source_db=DB_NAME_INIT,
            source_version=EI_VERSION,
            key=KEY_PREMISE,
            biosphere_name=BIOSPHERE_DB,
            additional_inventories=[
                {
                    "filepath": r"input_data\lci-add.xlsx",
                    "ecoinvent version": EI_VERSION,
                }
            ],
        )

        # Write the new reference database to Brightway2
        ndb.write_db_to_brightway(name=NAME_REF_DB)


def generate_prospective_lca_dbs(list_spec_scenarios, list_names):
    """
    Generate and update future LCA databases for prospective LCA.

    This function generates and updates future LCA databases based on specified scenarios if needed, and writes them to Brightway2.

    Parameters:
    - list_spec_scenarios (list): List of dictionaries specifying scenarios for new databases.
    - list_names (list): List of names specifying scenario names for new databases.

    Returns:
    None

    Example:
    >>> generate_and_update_lca_databases([{"model": "remind", "pathway": 'SSP2-Base', "year": "2035"}], ['ecoinvent_remind_SSP2-Base_2035_base'])
    """
    bd.projects.set_current(PROJECT_NAME)
    if len(list_spec_scenarios) > 0:
        ndb = NewDatabase(
            scenarios=list_spec_scenarios,
            source_db=DB_NAME_INIT,
            source_version=EI_VERSION,
            biosphere_name=BIOSPHERE_DB,
            key=KEY_PREMISE,
            additional_inventories=[
                {
                    "filepath": r"input_data\lci-add.xlsx",
                    "ecoinvent version": EI_VERSION,
                }
            ],
        )

        print("START UPDATING")
        ndb.update()

        print("START WRITING")
        ndb.write_db_to_brightway(name=list_names)


def get_low_voltage_grouped_locations(database_name):
    """
    Return all locations for activities in the given Brightway2 database
    where the activity name is 'market group for electricity, low voltage'
    and the reference product is 'electricity, low voltage'.
    """
    bd.projects.set_current(PROJECT_NAME)
    db = bd.Database(database_name)
    locations = [
        act["location"]
        for act in db
        if act["name"] == "market group for electricity, low voltage"
        and act["reference product"] == "electricity, low voltage"
    ]
    return locations


def get_tech_environmental_burdens(
    cost_dict, sec_db=NAME_REF_DB, lcia_method=CC_METHOD
):
    """
    Gets the environmental impact from each activity used in the MES.

    Args:
        cost_dict (dic): dictionary with cost data [-].
        ei_loc (str): ecoinvent location [-].
        sec_db (str): ecoinvent database used [-].
        lcia_method (str): Standard LCIA method used, here CC_METHOD

    Returns:
        dict_env_impacts (dict): dictionary with environmental impacts for the 'lcia_method' specified.
        env_impact (float): environmental burden factor for grid absorption from the grid.
        env_impact (float): environmental burden credit for grid injection to the grid.
    """
    # Use BW project created in create_db_lca_functions
    # define name of project
    bd.projects.set_current(PROJECT_NAME)  # Creating/accessing the project

    # Hydrogen storage vessel
    env_imp_h2_ves = get_activity_env(
        "high pressure hydrogen storage tank",
        "GLO",
        "high pressure hydrogen storage tank",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    ) / (
        MJ_KG_H2 / MJ_kWh
    )  # per vessel, 1 kg H2 storage, convert to kWh

    # Ground-mounted solar PV panels
    env_imp_pv = get_activity_env(
        "photovoltaic open ground installation, 570 kWp, multi-Si, on open ground",
        "RER",
        "photovoltaic open ground installation, 570 kWp, multi-Si, on open ground",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    ) / (
        570 * 0.895
    )  # per 570 kWp, however, consider degradation

    # onhore wind
    env_imp_wind_on = (
        get_activity_env(
            "market for wind turbine, 2MW, onshore",
            "GLO",
            "wind turbine, 2MW, onshore",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
        + get_activity_env(
            "market for wind turbine network connection, 2MW, onshore",
            "GLO",
            "wind turbine network connection, 2MW, onshore",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
    ) / 2000
    # electrolyzer
    env_imp_electr = (
        get_activity_env(
            "electrolyzer production, 1MWe, PEM, Stack",
            "RER",
            "electrolyzer, 1MWe, PEM, Stack",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
        + (cost_dict["electr_lt"] / cost_dict["electr_bos_lt"])
        * get_activity_env(
            "electrolyzer production, 1MWe, PEM, Balance of Plant",
            "RER",
            "electrolyzer, 1MWe, PEM, Balance of Plant",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
    ) / 1000

    # battery, here NMC, per kWh
    env_imp_bat_cap = get_activity_env(
        "market for battery capacity, Li-ion, NMC622, stationary",
        "GLO",
        "electricity storage capacity",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    )

    env_impact_grid_network = (
        get_activity_env(
            "wind turbine network connection construction, 4.5MW, onshore",
            "GLO",
            "wind turbine network connection, 4.5MW, onshore",
            sec_db,
            "",
            "",
            lcia_method=lcia_method,
        )
        / 4500
    )

    env_impact_asu = get_activity_env(
        "nitrogen production, infrastructure",
        "GLO",
        "nitrogen production, infrastructure",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    )

    env_impact_hb = get_activity_env(
        "ammonia production, infrastructure and catalyst",
        "GLO",
        "ammonia production, infrastructure and catalyst",
        sec_db,
        "",
        "",
        lcia_method=lcia_method,
    )

    ##############################################
    ############################################## from kg CO2 to tonne is -1e3, from kWh to Mwh is factor 1e3
    ##############################################

    dict_env_impacts = {
        "ghg_imp_h2_ves": env_imp_h2_ves,  # t/MWh
        "ghg_imp_pv": env_imp_pv,  # t/MWp
        "ghg_imp_wind_on": env_imp_wind_on,  # t/MWp
        "ghg_imp_electr": env_imp_electr,  # t/MW
        "ghg_imp_bat_cap": env_imp_bat_cap,  # t/MWhp
        "ghg_impact_grid_network": env_impact_grid_network,  # t/MW
        "ghg_imp_asu": env_impact_asu,  # t/t
        "ghg_imp_hb": env_impact_hb,  # t/t
    }

    if lcia_method != my_methods[0]:
        dict_env_impacts.keys.replace("ghg_imp", "env_imp").replace(
            "ghg", "env"
        )

    return dict_env_impacts


# ecoinvent_remind_SSP2-PkBudg1300_2030_all
def get_activity_env(
    name: str,
    location: str,
    ref_product: str,
    db: str,
    year: str,
    scenario: str,
    lcia_method=CC_METHOD,
) -> float:
    """
    Gets the environmental impact of an activity from a specified ecoinvent database.

    Args:
        name (str): activity name [-].
        location (str): activity location [-].
        ref_product (str): reference product [-].
        year (str): year of database [-].
        scenario (str): IAM scenario used [-].
        lcia_method (str): Standard LCIA method used, here CC_METHOD

    Returns:
        float: environmental impact.
        string: location of activity found.
    """

    if db == "":
        db_name = "ecoinvent_remind_{}_{}_all".format(scenario, year)
    else:
        db_name = db

    # For PV db, we don't have a reference product
    if ref_product == "":
        activity = [
            x
            for x in bd.Database(db_name)
            if name == x["name"] and location == x["location"]
        ][0]
    else:
        activity = [
            x
            for x in bd.Database(db_name)
            if name == x["name"]
            and location == x["location"]
            and ref_product == x["reference product"]
        ][0]
    lca = CheckedGMRESLCA(
        {mes: 1},
        method=lcia_method,
        rtol=1e-10,
        atol=0.0,
        restart=50,
        maxiter=1000,
    )
    lca.lci()
    lca.lcia()

    return lca.score


# ecoinvent_remind_SSP2-PkBudg1300_2030_all
def get_activity_env_elect(
    name: str,
    location: str,
    ref_product: str,
    db: str,
    year: str,
    scenario: str,
    lcia_method=CC_METHOD,
) -> float:
    """
    Gets the environmental impact of an electricity activity from a specified ecoinvent database.

    Args:
        name (str): activity name [-].
        location (str): activity location [-].
        ref_product (str): reference product [-].
        year (str): year of database [-].
        scenario (str): IAM scenario used [-].
        lcia_method (str): Standard LCIA method used, here CC_METHOD

    Returns:
        float: environmental impact.
    """

    if db == "":
        db_name = "ecoinvent_remind_{}_{}_all".format(scenario, year)
    else:
        db_name = db

    activity = [
        x
        for x in bd.Database(db)
        if name == x["name"]
        and location == x["location"]
        and ref_product == x["reference product"]
    ]

    if len(activity) < 1:
        # Check whether there is a market group activity for larger area
        activity = [
            x
            for x in bd.Database(db)
            if x["name"] == "market group for electricity, low voltage"
            and location == x["location"]
            and ref_product == x["reference product"]
        ]

        if len(activity) < 1:
            # Try to select GLO activity
            activity = [
                x
                for x in bd.Database(db)
                if x["name"] == "market group for electricity, low voltage"
                and "GLO" == x["location"]
                and ref_product == x["reference product"]
            ]

            if len(activity) < 1:
                # Try to select RoW activity
                activity = [
                    x
                    for x in bd.Database(db)
                    if name == x["name"]
                    and "RoW" == x["location"]
                    and ref_product == x["reference product"]
                ]
                if len(activity) == 1:
                    # Select this activity
                    activity = activity[0]
                else:
                    print("ERROR: No activity found for '{}'".format(name))
            else:
                # Select this activity
                activity = activity[0]

        elif len(activity) == 1:
            # Select this activity
            activity = activity[0]
        else:
            print("ERROR: More than 1 activity found for '{}'".format(name))

    elif len(activity) == 1:
        # Select this activity
        activity = activity[0]
    else:
        print("ERROR: More than 1 activity found for '{}'".format(name))

    lca = bc.LCA({activity: 1}, method=lcia_method)
    lca.lci()
    lca.lcia()

    return lca.score, activity["location"]


def create_process(location, year, exchanges):
    """
    Create a process for a multi-energy system.

    Parameters:
        location (str): The location of the multi-energy system.
        year (int): The year of the multi-energy system.
        exchanges (list): A list of exchanges associated with the process.

    Returns:
        dict: A dictionary representing the LCA process of the multi-energy system process.
    """

    name = "multi_energy_system_{}_{}".format(location, year)
    return {
        "name": name,
        "code": str(uuid.uuid4().hex),
        "unit": "unit",
        "reference product": name,
        "location": location,
        "exchanges": exchanges,
    }


def group_exchange_scores(lst):
    """
    Group exchange scores based on a list of exchanges and their scores.

    Parameters:
        lst (list): A list of tuples where each tuple contains an exchange and its associated score.

    Returns:
        dict: A dictionary with group labels as keys and corresponding LCIA scores as values.
    """

    # We will store results in a dict, with keys for group label and values of LCIA score.
    grouped_results = defaultdict(int)

    for exc, score in lst[1:]:
        grouped_results[
            contribution_mapping_system[exc.input["name"]]
        ] += score
    return grouped_results


def drop_empty_categories(db):
    """
    Drop categories with the value ('',) from the database.

    Parameters:
        db (list): A list of processes or datasets in a database.

    Returns:
        list: The modified database with empty categories removed.
    """

    DROP = ("",)
    for ds in db:
        if ds.get("categories") == DROP:
            del ds["categories"]
        for exc in ds.get("exchanges", []):
            if exc.get("categories") == DROP:
                del exc["categories"]
    return db


def strip_nonsense(db):
    """
    Strip leading and trailing spaces from strings in the database.

    Parameters:
        db (list): A list of processes or datasets in a database.

    Returns:
        list: The modified database with leading and trailing spaces removed from string values.
    """
    for ds in db:
        for key, value in ds.items():
            if isinstance(value, str):
                ds[key] = value.strip()
            for exc in ds.get("exchanges", []):
                for key, value in exc.items():
                    if isinstance(value, str):
                        exc[key] = value.strip()
    return db


def create_process_and_add(location, year, exchanges, sec_db):
    """Creates database with processes"""
    IMPORTER = LCIImporter(DB_NAME)  #
    IMPORTER.data = [create_process(location, year, exchanges)]

    IMPORTER.strategies = [
        partial(add_database_name, name=DB_NAME),
        csv_restore_tuples,
        drop_empty_categories,
        strip_nonsense,
    ]
    IMPORTER.apply_strategies()
    IMPORTER.match_database(
        sec_db,
        fields=("name", "unit", "location", "reference product", "database"),
    )
    # IMPORTER.match_database(DB_NAME_CONS, fields=('name','unit','location','reference product', 'database'))
    IMPORTER.match_database(fields=("name",))
    IMPORTER.statistics()
    IMPORTER.write_excel(only_unlinked=True)
    IMPORTER.write_database()


def get_activity_key(db_name, name, ref_product, location):
    """
    Return the Brightway key (database, code) for a uniquely matching activity.
    """
    matches = [
        act
        for act in bd.Database(db_name)
        if act["name"] == name
        and act["reference product"] == ref_product
        and act["location"] == location
    ]

    if len(matches) == 0:
        raise ValueError(
            f"No match found in '{db_name}' for:\n"
            f"  name={name}\n"
            f"  reference product={ref_product}\n"
            f"  location={location}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches found in '{db_name}' for:\n"
            f"  name={name}\n"
            f"  reference product={ref_product}\n"
            f"  location={location}\n"
            f"Matches: {[a.key for a in matches]}"
        )

    return matches[0].key


def write_mes_activities_to_db(activity_datasets, db_name, overwrite=True):
    """
    Write all MES foreground datasets to one Brightway database.
    Enforces internal consistency with the target db_name and resolves technosphere inputs.
    """
    fixed = {}

    for code, ds in activity_datasets.items():
        ds = ds.copy()
        ds["database"] = db_name
        ds["code"] = code

        fixed_exchanges = []

        for exc in ds.get("exchanges", []):
            exc = exc.copy()

            if exc.get("type") == "production":
                exc["input"] = (db_name, code)

            elif exc.get("type") == "technosphere":
                exc["input"] = get_activity_key(
                    db_name=exc["database"],
                    name=exc["name"],
                    ref_product=exc["reference product"],
                    location=exc["location"],
                )

            fixed_exchanges.append(exc)

        ds["exchanges"] = fixed_exchanges
        print(ds)
        fixed[(db_name, code)] = ds

    db = bd.Database(db_name)

    if overwrite:
        db.write(fixed)
    else:
        if len(db) == 0:
            db.write(fixed)
        else:
            raise ValueError(
                f"Database '{db_name}' already contains activities. "
                "Use overwrite=True or implement append logic."
            )


def build_mes_activity_dataset(
    parm,
    loc_elect,
    cap_wind_on,
    cap_pv,
    cap_bat_en,
    cap_h2_ves,
    cap_electrolyzer,
    cap_hb,
    cap_asu,
    cap_grid,
    summed_grid_abs,
    summed_grid_inj,
    sec_db,
    scenario_name,
    foreground_db,
    credit_env_export=False,
):
    """
    Build one Brightway foreground activity dataset in memory.
    """

    exchanges = []

    if cap_wind_on > 0:
        amount_wind = (
            cap_wind_on / 2 * (parm["project_lt"] / parm["wind_on_lt"])
        ) / parm["project_lt"]

        exchanges.append(
            {
                "name": "market for wind turbine, 2MW, onshore",
                "reference product": "wind turbine, 2MW, onshore",
                "database": sec_db,
                "location": "GLO",
                "type": "technosphere",
                "unit": "unit",
                "amount": amount_wind,
            }
        )

        exchanges.append(
            {
                "name": "market for wind turbine network connection, 2MW, onshore",
                "reference product": "wind turbine network connection, 2MW, onshore",
                "database": sec_db,
                "location": "GLO",
                "type": "technosphere",
                "unit": "unit",
                "amount": amount_wind,
            }
        )

    if cap_pv > 0:
        exchanges.append(
            {
                "name": "photovoltaic open ground installation, 570 kWp, multi-Si, on open ground",
                "reference product": "photovoltaic open ground installation, 570 kWp, multi-Si, on open ground",
                "database": sec_db,
                "location": "RER",
                "type": "technosphere",
                "unit": "unit",
                "amount": (
                    cap_pv
                    / (0.570 * 0.895)
                    * (parm["project_lt"] / parm["pv_lt"])
                )
                / parm["project_lt"],
            }
        )

    if cap_bat_en > 0:
        exchanges.append(
            {
                "name": "market for battery capacity, Li-ion, NMC622, stationary",
                "reference product": "electricity storage capacity",
                "database": sec_db,
                "location": "GLO",
                "type": "technosphere",
                "unit": "kilowatt hour",
                "amount": (
                    cap_bat_en * 1e3 * (parm["project_lt"] / parm["bat_en_lt"])
                )
                / parm["project_lt"],
            }
        )

    if cap_h2_ves > 0:
        exchanges.append(
            {
                "name": "high pressure hydrogen storage tank",
                "reference product": "high pressure hydrogen storage tank",
                "database": sec_db,
                "location": "GLO",
                "type": "technosphere",
                "unit": "kilogram",
                "amount": (
                    1e3
                    * (cap_h2_ves / (MJ_KG_H2 / MJ_kWh))
                    * (parm["project_lt"] / parm["h2_ves_lt"])
                )
                / parm["project_lt"],
            }
        )

    if cap_electrolyzer > 0:
        exchanges.append(
            {
                "name": "electrolyzer production, 1MWe, PEM, Stack",
                "reference product": "electrolyzer, 1MWe, PEM, Stack",
                "database": sec_db,
                "location": "RER",
                "type": "technosphere",
                "unit": "unit",
                "amount": (
                    cap_electrolyzer * (parm["project_lt"] / parm["electr_lt"])
                )
                / parm["project_lt"],
            }
        )

        exchanges.append(
            {
                "name": "electrolyzer production, 1MWe, PEM, Balance of Plant",
                "reference product": "electrolyzer, 1MWe, PEM, Balance of Plant",
                "database": sec_db,
                "location": "RER",
                "type": "technosphere",
                "unit": "unit",
                "amount": (
                    cap_electrolyzer
                    * (parm["project_lt"] / parm["electr_bos_lt"])
                )
                / parm["project_lt"],
            }
        )

    if summed_grid_abs > 0:
        if loc_elect in get_low_voltage_grouped_locations(sec_db):
            exchanges.append(
                {
                    "name": "market group for electricity, low voltage",
                    "reference product": "electricity, low voltage",
                    "database": sec_db,
                    "location": loc_elect,
                    "type": "technosphere",
                    "unit": "kilowatt hour",
                    "amount": 1e3 * summed_grid_abs,
                }
            )
        else:
            exchanges.append(
                {
                    "name": "market for electricity, low voltage",
                    "reference product": "electricity, low voltage",
                    "database": sec_db,
                    "location": loc_elect,
                    "type": "technosphere",
                    "unit": "kilowatt hour",
                    "amount": 1e3 * summed_grid_abs,
                }
            )

    if credit_env_export and summed_grid_inj > 0:
        if loc_elect in get_low_voltage_grouped_locations(sec_db):
            exchanges.append(
                {
                    "name": "market group for electricity, low voltage",
                    "reference product": "electricity, low voltage",
                    "database": sec_db,
                    "location": loc_elect,
                    "type": "technosphere",
                    "unit": "kilowatt hour",
                    "amount": -1e3 * summed_grid_inj,
                }
            )
        else:
            exchanges.append(
                {
                    "name": "market for electricity, low voltage",
                    "reference product": "electricity, low voltage",
                    "database": sec_db,
                    "location": loc_elect,
                    "type": "technosphere",
                    "unit": "kilowatt hour",
                    "amount": -1e3 * summed_grid_inj,
                }
            )

    if cap_grid > 0:
        exchanges.append(
            {
                "name": "wind turbine network connection construction, 4.5MW, onshore",
                "reference product": "wind turbine network connection, 4.5MW, onshore",
                "database": sec_db,
                "location": "GLO",
                "type": "technosphere",
                "unit": "unit",
                "amount": (
                    (cap_grid / 4.5) * (parm["project_lt"] / parm["grid_lt"])
                )
                / parm["project_lt"],
            }
        )

    if cap_asu > 0:
        exchanges.append(
            {
                "name": "nitrogen production, infrastructure",
                "reference product": "nitrogen production, infrastructure",
                "database": sec_db,
                "location": "GLO",
                "type": "technosphere",
                "unit": "kilogram",
                "amount": cap_asu,
            }
        )

    if cap_hb > 0:
        exchanges.append(
            {
                "name": "ammonia production, infrastructure and catalyst",
                "reference product": "ammonia production, infrastructure and catalyst",
                "database": sec_db,
                "location": "GLO",
                "type": "technosphere",
                "unit": "kilogram",
                "amount": cap_hb,
            }
        )

    activity_code = uuid.uuid4().hex
    activity_name = f"multi_energy_system_{loc_elect}_{sec_db}_{scenario_name}"

    dataset = {
        "name": activity_name,
        "reference product": "ammonia, at plant",
        "unit": "kilogram",
        "location": loc_elect,
        "database": foreground_db,
        "code": activity_code,
        "type": "process",
        "exchanges": [
            {
                "input": (foreground_db, activity_code),
                "amount": 1,
                "type": "production",
                "name": activity_name,
                "unit": "kilogram",
            },
            *exchanges,
        ],
    }

    return activity_code, dataset


def get_activity_key(db_name, name, ref_product, location):
    """
    Return the Brightway key (database, code) for a uniquely matching activity.
    """
    matches = [
        act
        for act in bd.Database(db_name)
        if act["name"] == name
        and act["reference product"] == ref_product
        and act["location"] == location
    ]

    if len(matches) == 0:
        raise ValueError(
            f"No match found in '{db_name}' for:\n"
            f"  name={name}\n"
            f"  reference product={ref_product}\n"
            f"  location={location}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches found in '{db_name}' for:\n"
            f"  name={name}\n"
            f"  reference product={ref_product}\n"
            f"  location={location}\n"
            f"Matches: {[a.key for a in matches]}"
        )

    return matches[0].key


def write_mes_activities_to_db(activity_datasets, db_name, overwrite=True):
    """
    Write all MES foreground datasets to one Brightway database.
    Enforces internal consistency with the target db_name and resolves technosphere inputs.
    """
    fixed = {}

    for code, ds in activity_datasets.items():
        ds = ds.copy()
        ds["database"] = db_name
        ds["code"] = code

        fixed_exchanges = []

        for exc in ds.get("exchanges", []):
            exc = exc.copy()

            if exc.get("type") == "production":
                exc["input"] = (db_name, code)

            elif exc.get("type") == "technosphere":
                exc["input"] = get_activity_key(
                    db_name=exc["database"],
                    name=exc["name"],
                    ref_product=exc["reference product"],
                    location=exc["location"],
                )

            fixed_exchanges.append(exc)

        ds["exchanges"] = fixed_exchanges
        fixed[(db_name, code)] = ds

    db = bd.Database(db_name)

    print("Start writing MES activities to database '{}'...".format(db_name))

    if overwrite:
        # 🔹 Delete existing database if it exists
        if db_name in bd.databases:
            del bd.databases[db_name]

        db.write(fixed)
    else:
        if len(db) == 0:
            db.write(fixed)
        else:
            raise ValueError(
                f"Database '{db_name}' already contains activities. "
                "Use overwrite=True or implement append logic."
            )


def environmental_lca(
    activity_code,
    foreground_db,
    sec_db,
    ghgs_opt,
    lcia_method=CC_METHOD,
    my_methods=None,
):
    """
    Calculate LCIA for one already-written MES foreground activity.
    """
    if my_methods is None:
        my_methods = [lcia_method]

    mes = bd.Database(foreground_db).get(activity_code)

    lca = bc.LCA({mes: 1}, method=lcia_method)
    lca.lci()
    lca.lcia()

    if "climate change" in str(lcia_method):
        if abs(ghgs_opt - lca.score) > 3:
            print("**************************************************")
            print(abs(ghgs_opt - lca.score))
            print(
                f"Difference between scores, initial calc score is '{ghgs_opt}' and LCA score here is '{lca.score}'"
            )

            lca.lcia(demand={mes.id: 1})
            for exc in mes.exchanges():
                if exc["type"] == "technosphere":
                    lca.lcia(demand={exc.input.id: exc["amount"]})
                    print(
                        f"{exc['name']}, amount: '{exc['amount']}', lca results: '{lca.score}'"
                    )
                elif exc["type"] == "biosphere":
                    cf = lca.characterization_matrix[
                        lca.biosphere_dict[exc.input.id], :
                    ].sum()
                    print(
                        f"{exc['name']}, amount: '{exc['amount']}', lca results: '{cf * exc['amount']}'"
                    )

            raise ValueError("ERROR: please check GHG calculation")

    result_array = [[] for _ in my_methods]

    for i, method in enumerate(my_methods):
        lca.switch_method(method)
        lca.lcia(demand={mes.id: 1})

        result_array[i].append(("total", lca.score))

        for exc in mes.exchanges():
            if exc["type"] == "technosphere":
                lca.lcia(demand={exc.input.id: exc["amount"]})
                result_array[i].append((exc, lca.score))
            elif exc["type"] == "biosphere":
                cf = lca.characterization_matrix[
                    lca.biosphere_dict[exc.input.id], :
                ].sum()
                result_array[i].append((exc, cf * exc["amount"]))

    for arr in result_array:
        if not np.allclose(arr[0][1], sum(o[1] for o in arr[1:])):
            print("Mismatch")

    grouped_array = [group_exchange_scores(arr) for arr in result_array]

    data_frames = []
    for i, group_data in enumerate(grouped_array):
        data_0 = dict(group_data)
        col_name = mes["name"]

        df_add = pd.DataFrame.from_dict(data_0, orient="index")
        df_add.rename(
            columns={"climate change total": col_name, 0: col_name},
            inplace=True,
        )
        df_add.index.names = ["contributor"]
        df_add["category"] = str(my_methods[i][2])
        df_add["year"] = ASSESSMENT_YEAR
        df_add["db_name"] = sec_db
        data_frames.append(df_add)

    df_total = pd.concat(data_frames, axis=0)
    return df_total.reset_index().set_index(
        ["category", "contributor", "year", "db_name"]
    )
