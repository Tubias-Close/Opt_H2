"""
Parallel case-study workflow for decentralized electric ammonia systems.

This script generates techno-economic and environmental results for multiple
ammonia production configurations across many locations and background
databases. It is designed to separate the computationally heavy optimization
step from the Brightway database-writing and LCA steps, so that optimization
can be parallelized safely while Brightway operations are performed once in
the main process.

Step-by-step overview
--------------------
1. Set up the Brightway project and load all required inputs.

2. Build a global list of optimization jobs.
   - Each job contains all information needed to run one optimization:
     country, ISO2 code, coordinates, scenario settings, selected background
     database, techno-economic data, environmental data, and electricity-price
     assumptions.

3. Run all optimization jobs in parallel.
   - A ProcessPoolExecutor distributes the optimization cases across the
     requested number of CPU cores.

4. Collect optimization outputs in the main process.
   - Infeasible cases are skipped with a warning.
   - Failed jobs are caught and reported without stopping the entire batch.
   - Successful optimization result tables are appended to a master list.
   - The returned optimization variables are checked for the required fields
     needed to build the foreground LCA activities.

5. Build all Brightway foreground datasets in memory.
   - For each successful case, a foreground ammonia-system activity is created
     using the optimization outputs.
   - These datasets are not written immediately; instead they are stored in
     one large dictionary in memory.

6. Write the foreground database once.
   - After all optimization jobs are finished, the full set of foreground
     activities is written to the Brightway foreground database in one step.
   - This avoids repeated database writes and keeps the workflow more robust
     and efficient.

7. Run life-cycle assessment for all successful cases.
   - After the foreground database has been written, LCAs are calculated for
     each stored activity.
   - The script uses the selected LCIA method(s) and checks climate-change
     scores against the optimization-derived GHG result for consistency.
   - Contribution results are grouped and stored in a structured dataframe.

8. Combine and save final outputs.
   - All techno-economic results are concatenated into one dataframe and saved
     as a pickle file.
   - All LCA contribution results are concatenated into a second dataframe and
     also saved as a pickle file.
   - If result generation is disabled, previously saved pickle files are loaded
     instead.

Why the workflow is structured this way
---------------------------------------
- Optimization is the most computationally expensive step, so it is run in
  parallel to reduce total runtime.
- Brightway database writing and foreground LCA calculations are handled only
  in the main process, because database operations are more fragile and should
  not be performed concurrently across workers.
- Building all jobs first and writing the foreground database once minimizes
  overhead and makes the workflow easier to debug and reproduce.

Main outputs
------------
- `totals_cost_all`:
    techno-economic results for all successful location-scenario-database cases.
- `totals_cost_all_lca`:
    life-cycle impact results and contribution breakdowns for the same cases.

Typical use case
----------------
Run this script when you want to evaluate a large set of ammonia-production
case studies over multiple countries, scenarios, and background databases.
"""

import concurrent.futures
import json
import pickle
import time

import bw2data
import pandas as pd

import calculate_renewable_yield as cry
import energy_data_processor as ep
import opt_ammonia_functions as opt

# import own Python files, vars, mappings, and functions
from config import (
    CC_METHOD,
    COST_DATA,
    FILE_PATH_CASE_STUDIES,
    FILE_PATH_CASE_STUDIES_LCA,
    FUTURE_POWER_PRICES,
    LOCATIONS,
    NAME_FUTURE_DB,
    NAME_REF_DB,
    PROJECT_NAME,
)
from create_db_lca_functions import (
    build_mes_activity_dataset,
    environmental_lca,
    write_mes_activities_to_db,
)
from energy_data_processor import country_to_iso2
from mapping import my_methods

# -----------------------------
# Global settings
# -----------------------------
bw2data.projects.set_current(PROJECT_NAME)

FOREGROUND_DB = "db_ammonia_system"
GEN_RESULTS = True

COST_DICT = COST_DATA[NAME_REF_DB].to_dict()
COST_DICT_FUTURE = COST_DATA[NAME_FUTURE_DB].to_dict()

ALL_DBS = [NAME_REF_DB, NAME_FUTURE_DB]

SCENARIOS = {
    # "grid_connected": {"autonomous_elect": False, "no_renewables": True},
    # "hybrid": {"autonomous_elect": False, "no_renewables": False},
    # "hybrid-green": {
    #    "autonomous_elect": False,
    #    "no_renewables": False,
    #    "hybrid_green": True,
    # },
    "off_grid": {"autonomous_elect": True, "no_renewables": False},
}

# -----------------------------
# Load input data
# -----------------------------
with open("input_data/dict_ghg_impacts.txt", "r") as file:
    DICT_GHG_IMPACTS = json.load(file)

with open("input_data/dict_ghg_impacts_future.txt", "r") as file:
    DICT_GHG_IMPACTS_FUTURE = json.load(file)

future_power_prices_df = pd.read_excel(
    FUTURE_POWER_PRICES,
    sheet_name="lcoe",
    index_col="country",
)[["2 degree_2050"]]

future_power_prices_df["iso2"] = future_power_prices_df.index.map(
    country_to_iso2
)
future_power_prices_df.reset_index(inplace=True)
future_power_prices_df = future_power_prices_df[["iso2", "2 degree_2050"]]
future_power_prices_df.set_index("iso2", inplace=True)
FUTURE_POWER_PRICES_DICT = future_power_prices_df["2 degree_2050"].to_dict()


# -----------------------------
# Worker
# -----------------------------
def run_single_case_ammonia(job):
    """
    Run one optimization case in a worker process.
    Returns a dict with status + outputs needed later in the main process.
    """
    try:
        (
            country,
            iso2,
            lat,
            lon,
            scenario_name,
            kwargs,
            db,
            cost_dict_in,
            dict_ghg_impacts_in,
            future_power_prices_dict,
        ) = job

        cost_dict = cost_dict_in.copy()
        dict_ghg_impacts = dict_ghg_impacts_in.copy()

        # Optimization limits
        dict_limits = ep.get_max_caps_regions()

        # Country-specific WACC / discount rate
        cost_dict["dr"] = ep.get_latest_avg_wacc(iso2)

        # Country-specific retail prices
        power_prices = (
            ep.get_elect_prices(iso2)
            if db == NAME_REF_DB
            else ep.get_elect_prices(iso2, price_file=future_power_prices_dict)
        )

        # Renewable CFs
        data_processor = cry.RenewableEnergyProcessor(lat, lon)
        cf_pv, cf_wind = data_processor.process_data()

        # Build hourly input dataframe
        df_data = pd.DataFrame(
            data={
                "pv_MW_array": cf_pv,
                "wind_MW_array_on": cf_wind,
                "grid_abs_price": power_prices,
                "rev_inj": 0,
            },
            index=pd.date_range("1/1/2023 00:00", periods=8760, freq="h"),
        )

        # Grid GHG intensities
        df_data["ghg_impact"] = ep.get_activity_env_elect_from_dict(
            iso2, db=db
        )
        df_data["ghg_impact_cons"] = 0

        # Run optimization only
        totals_cost_min, _, opt_vars = opt.opt_amm_electric_hb(
            df_data,
            1,
            0,
            cost_dict,
            dict_ghg_impacts,
            dict_limits,
            sec_db=db,
            export_alias=scenario_name,
            LOC_ELECT=iso2,
            consider_down_times=False,
            **kwargs,
        )

        # Handle infeasible cases
        if len(totals_cost_min) == 0 and len(opt_vars) == 0:
            return {
                "status": "infeasible",
                "country": country,
                "iso2": iso2,
                "scenario": scenario_name,
                "db_name": db,
            }

        totals_cost_min = totals_cost_min.copy()
        totals_cost_min["country"] = country
        totals_cost_min["iso2"] = iso2
        totals_cost_min["scenario"] = scenario_name
        totals_cost_min["db_name"] = db

        return {
            "status": "ok",
            "country": country,
            "iso2": iso2,
            "scenario": scenario_name,
            "db_name": db,
            "totals_cost_min": totals_cost_min,
            "opt_vars": opt_vars,
            "parm": cost_dict,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": repr(e),
            "country": job[0],
            "iso2": job[1],
            "scenario": job[4],
            "db_name": job[6],
        }


# -----------------------------
# Global job builder
# -----------------------------
def build_all_jobs(locations, scenarios, all_dbs):
    """
    Build all jobs across all background databases.
    """
    jobs = []

    for db in all_dbs:
        for country, iso2, lat, lon in locations:
            cost_dict = (
                COST_DICT.copy()
                if db == NAME_REF_DB
                else COST_DICT_FUTURE.copy()
            )
            dict_ghg_impacts = (
                DICT_GHG_IMPACTS.copy()
                if db == NAME_REF_DB
                else DICT_GHG_IMPACTS_FUTURE.copy()
            )

            for scenario_name, kwargs in scenarios.items():
                jobs.append(
                    (
                        country,
                        iso2,
                        lat,
                        lon,
                        scenario_name,
                        kwargs,
                        db,
                        cost_dict,
                        dict_ghg_impacts,
                        FUTURE_POWER_PRICES_DICT,
                    )
                )

    return jobs


# -----------------------------
# Main
# -----------------------------
def main_parallel_ammonia(max_workers=4, locations_subset=None):
    start_time = time.time()

    all_totals = []
    all_totals_lca = []

    if locations_subset is None:
        locations_subset = LOCATIONS

    if GEN_RESULTS:
        print("Building all jobs across all background databases...")
        jobs = build_all_jobs(
            locations=locations_subset,
            scenarios=SCENARIOS,
            all_dbs=ALL_DBS,
        )
        total_jobs = len(jobs)
        print(f"Prepared {total_jobs} jobs.")

        all_activity_datasets = {}
        all_run_meta = []
        finished = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(run_single_case_ammonia, job) for job in jobs
            ]

            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                finished += 1

                elapsed_time = time.time() - start_time
                elapsed_minutes = elapsed_time / 60
                avg_time_per_job = elapsed_time / finished
                remaining_minutes = (
                    avg_time_per_job * (total_jobs - finished) / 60
                )

                print(
                    f"\rFinished {finished}/{total_jobs} | "
                    f"Elapsed: {elapsed_minutes:.0f} min | "
                    f"Remaining: {remaining_minutes:.0f} min",
                    end="",
                )

                if res["status"] == "infeasible":
                    print(
                        f"\nWarning: infeasible optimization for "
                        f'{res["country"]} - {res["scenario"]} - {res["db_name"]}'
                    )
                    continue

                if res["status"] == "error":
                    print(
                        f"\nError for {res['country']} - {res['scenario']} - {res['db_name']}: "
                        f"{res['error']}"
                    )
                    continue

                country = res["country"]
                iso2 = res["iso2"]
                scenario_name = res["scenario"]
                db_name = res["db_name"]
                totals_cost_min = res["totals_cost_min"]
                opt_vars = res["opt_vars"]
                parm = res["parm"]

                all_totals.append(totals_cost_min)

                required_keys = [
                    "cap_wind_on",
                    "cap_pv",
                    "cap_bat_en",
                    "cap_h2_ves",
                    "cap_electrolyzer",
                    "cap_hb",
                    "cap_asu",
                    "cap_grid",
                    "summed_grid_abs",
                    "summed_grid_inj",
                    "ghgs_opt",
                ]
                missing = [k for k in required_keys if k not in opt_vars]
                if missing:
                    print(
                        f"\nWarning: opt_amm_electric_hb missing keys "
                        f"for {country} - {scenario_name} - {db_name}: {missing}"
                    )
                    continue

                build_vars = opt_vars.copy()
                build_vars["loc_elect"] = iso2
                build_vars["sec_db"] = db_name
                build_vars["credit_env_export"] = build_vars.get(
                    "credit_env_export", False
                )

                activity_code, dataset = build_mes_activity_dataset(
                    parm=parm,
                    loc_elect=build_vars["loc_elect"],
                    cap_wind_on=build_vars["cap_wind_on"],
                    cap_pv=build_vars["cap_pv"],
                    cap_bat_en=build_vars["cap_bat_en"],
                    cap_h2_ves=build_vars["cap_h2_ves"],
                    cap_electrolyzer=build_vars["cap_electrolyzer"],
                    cap_hb=build_vars["cap_hb"],
                    cap_asu=build_vars["cap_asu"],
                    cap_grid=build_vars["cap_grid"],
                    summed_grid_abs=build_vars["summed_grid_abs"],
                    summed_grid_inj=build_vars["summed_grid_inj"],
                    sec_db=build_vars["sec_db"],
                    scenario_name=f"{country}_{scenario_name}",
                    foreground_db=FOREGROUND_DB,
                    credit_env_export=build_vars["credit_env_export"],
                )

                all_activity_datasets[activity_code] = dataset

                all_run_meta.append(
                    {
                        "country": country,
                        "iso2": iso2,
                        "scenario": scenario_name,
                        "db_name": db_name,
                        "activity_code": activity_code,
                        "ghgs_opt": build_vars["ghgs_opt"],
                    }
                )

        print()
        if not all_activity_datasets:
            raise ValueError(
                "No feasible cases found; nothing to write to the foreground DB."
            )

        print("Writing Brightway foreground DB once...")
        write_mes_activities_to_db(
            activity_datasets=all_activity_datasets,
            db_name=FOREGROUND_DB,
            overwrite=True,
        )

        print("Running LCAs...")
        for i, meta in enumerate(all_run_meta, 1):
            lca_results = environmental_lca(
                activity_code=meta["activity_code"],
                foreground_db=FOREGROUND_DB,
                sec_db=meta["db_name"],
                ghgs_opt=meta["ghgs_opt"],
                lcia_method=CC_METHOD,
                my_methods=my_methods,
            )

            lca_results = lca_results.copy().reset_index()
            lca_results["country"] = meta["country"]
            lca_results["iso2"] = meta["iso2"]
            lca_results["scenario"] = meta["scenario"]

            lca_results.set_index(
                [
                    "country",
                    "iso2",
                    "scenario",
                    "category",
                    "contributor",
                    "db_name",
                    "year",
                ],
                inplace=True,
            )

            lca_results.rename(
                columns={lca_results.columns[0]: "results"}, inplace=True
            )
            all_totals_lca.append(lca_results)

            if i % 10 == 0 or i == len(all_run_meta):
                print(f"LCA done: {i}/{len(all_run_meta)}")

        if not all_totals:
            raise ValueError("No optimization results collected.")

        totals_cost_all = pd.concat(all_totals, ignore_index=True).set_index(
            ["country", "iso2", "scenario", "db_name"]
        )
        totals_cost_all.to_pickle(FILE_PATH_CASE_STUDIES)

        if not all_totals_lca:
            raise ValueError("No LCA results collected.")

        totals_cost_all_lca = pd.concat(all_totals_lca, axis=0)
        totals_cost_all_lca.to_pickle(FILE_PATH_CASE_STUDIES_LCA)

    else:
        with open(FILE_PATH_CASE_STUDIES, "rb") as file:
            totals_cost_all = pickle.load(file)

        with open(FILE_PATH_CASE_STUDIES_LCA, "rb") as file:
            totals_cost_all_lca = pickle.load(file)

    return totals_cost_all, totals_cost_all_lca


if __name__ == "__main__":
    totals_cost_all, totals_cost_all_lca = main_parallel_ammonia(
        max_workers=8,
        locations_subset=LOCATIONS,
    )
