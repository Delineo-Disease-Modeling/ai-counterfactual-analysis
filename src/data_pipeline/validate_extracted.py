# validate_extracted.py
# Quick sanity-check for one or more extracted_runX folders
# Usage: python validate_extracted.py --base_dir all_extracted_logs

import argparse
import glob
import json
import os
import sys

import pandas as pd


def expect(condition, msg):
    if not condition:
        raise AssertionError(msg)


REQUIRED_PARQUETS = {
    "agents.parquet",
    "agent_timelines.parquet",
    "infections.parquet",
    "facilities.parquet",
}
AGENT_COLS = {
    "simulation_run_id",
    "person_id",
    "age",
    "sex",
    "household_id",
    "vaccination_status",
    "vaccination_doses",
    "initial_infected",
}
TIMELINE_COLS = {
    "simulation_run_id",
    "timestep",
    "person_id",
    "current_location_id",
    "current_location_type",
    "is_masked",
    "infection_status",
    "infectious_variants",
    "symptomatic_variants",
}
INFECT_COLS = {
    "simulation_run_id",
    "timestep",
    "infector_id",
    "infectee_id",
    "location_id",
    "location_type",
    "variant",
    "infector_masked",
    "infectee_masked",
    "transmission_pair_age_diff",
}
FACILITY_COLS = {
    "simulation_run_id",
    "timestep",
    "location_id",
    "location_type",
    "capacity",
    "occupancy",
    "utilization_rate",
    "infectious_count",
    "symptomatic_count",
    "masked_count",
    "vaccinated_count",
}


def validate_folder(run_dir):
    missing = REQUIRED_PARQUETS - set(os.listdir(run_dir))
    expect(
        not missing,
        f"{run_dir}: missing parquet files {', '.join(missing)}",
    )

    agents = pd.read_parquet(os.path.join(run_dir, "agents.parquet"))
    timelines = pd.read_parquet(os.path.join(run_dir, "agent_timelines.parquet"))
    infections = pd.read_parquet(os.path.join(run_dir, "infections.parquet"))
    facilities = pd.read_parquet(os.path.join(run_dir, "facilities.parquet"))
    chains = json.load(open(os.path.join(run_dir, "infection_chains.json")))

    expect(set(agents.columns) == AGENT_COLS, f"{run_dir}: agents schema mismatch")
    expect(
        set(timelines.columns) == TIMELINE_COLS, f"{run_dir}: timelines schema mismatch"
    )
    expect(
        set(infections.columns) == INFECT_COLS, f"{run_dir}: infections schema mismatch"
    )
    expect(
        set(facilities.columns) == FACILITY_COLS, f"{run_dir}: facilities schema mismatch"
    )

    run_id = agents["simulation_run_id"].iloc[0]
    for df, name in [
        (timelines, "timelines"),
        (infections, "infections"),
        (facilities, "facilities"),
    ]:
        expect(
            df["simulation_run_id"].nunique() == 1
            and df["simulation_run_id"].iloc[0] == run_id,
            f"{run_dir}: inconsistent run_id in {name}",
        )

    expect(
        agents["person_id"].is_unique, f"{run_dir}: duplicate person_id in agents"
    )

    timeline_ids = set(timelines["person_id"].unique())
    agent_ids = set(agents["person_id"].unique())
    expect(
        timeline_ids <= agent_ids,
        f"{run_dir}: timelines contain {len(timeline_ids - agent_ids)} unknown person_ids",
    )

    infect_ids = set(infections["infector_id"].dropna()) | set(
        infections["infectee_id"].dropna()
    )
    expect(
        infect_ids <= agent_ids,
        f"{run_dir}: infections reference unknown person_ids",
    )

    loc_ids = set(facilities["location_id"].unique())
    infect_loc_ids = set(infections["location_id"].unique())
    expect(
        infect_loc_ids <= loc_ids,
        f"{run_dir}: infections reference {len(infect_loc_ids - loc_ids)} unknown location_ids",
    )

    if infections.empty:
        # If there are literally no infection events, an empty chain list is fine.
        expect(
            len(chains) == 0,
            f"{run_dir}: chains present but infections.parquet empty?"
        )
    else:
        if infections["infector_id"].notna().any():
            expect(
                len(chains) > 0,
                f"{run_dir}: infections present but infection_chains.json empty"
            )

    edge_count       = infections["infector_id"].notna().sum()
    chain_feasible   = edge_count > 0
    agent_feasible   = not timelines.empty
    facility_feasible = not facilities.empty

    summary_msg = (
        f"✓ {run_dir.split('/')[-1]}  | "
        f"chains: {'yes' if chain_feasible else 'no'}  "
        f"(edges={edge_count})  | "
        f"agents: {'yes' if agent_feasible else 'no'}  "
        f"(rows={len(timelines)})  | "
        f"facilities: {'yes' if facility_feasible else 'no'} "
        f"(rows={len(facilities)})"
    )
    print(summary_msg)
    print(f"✔ {run_dir} passed all checks\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="all_extracted_logs")
    args = parser.parse_args()

    folders = sorted(glob.glob(os.path.join(args.base_dir, "extracted_run*")))
    if not folders:
        sys.exit(f"No extracted runs found in {args.base_dir}")

    for f in folders:
        validate_folder(f)


if __name__ == "__main__":
    main()