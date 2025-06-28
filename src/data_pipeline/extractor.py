# extractor.py
# to convert simulator logs to structured format for clustering

import argparse
import json
import os
import re
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy


def shannon_entropy(items):
    counts = np.array(list(Counter(items).values()), dtype=float)
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(entropy(probs, base=2))


def longest_path_length_dag(G, source):
    try:
        lengths = {source: 0}
        for node in nx.topological_sort(G):
            for succ in G.successors(node):
                lengths[succ] = max(lengths.get(succ, 0), lengths[node] + 1)
        return max(lengths.values())
    except nx.NetworkXUnfeasible:
        # cyclic component: fall back to BFS depth from source
        lengths = {source: 0}
        queue = [source]
        while queue:
            u = queue.pop()
            for v in G.successors(u):
                if v not in lengths:
                    lengths[v] = lengths[u] + 1
                    queue.append(v)
        return max(lengths.values())


def build_tables(input_dir: str, run_id: str):
    person = pd.read_csv(os.path.join(input_dir, "person_logs.csv"))
    movement = pd.read_csv(os.path.join(input_dir, "movement_logs.csv"))
    infect = pd.read_csv(os.path.join(input_dir, "infection_logs.csv"))
    facility = pd.read_csv(os.path.join(input_dir, "location_logs.csv"))

    if (
    infect.empty
    or "infector_person_id" not in infect.columns
    or infect["infector_person_id"].notna().sum() == 0      # <-- add this
):
        chains_csv = os.path.join(input_dir, "infection_chains.csv")
        if os.path.exists(chains_csv):
            print(f"[i] {run_id}: using infection_chains.csv as infection log")
            infect = (
                pd.read_csv(chains_csv)
                .rename(
                    columns={
                        "infector_id":        "infector_person_id",
                        "infected_person_id": "infected_person_id",
                        "location_id":        "infection_location_id",
                        "variant":            "variant",
                        "timestep":           "timestep",
                    }
                )
                .assign(
                    infection_location_type=None,
                    infector_masked=None,
                    infected_masked=None,
                    transmission_pair_age_diff=None,
                )
            )

    agents = (
        person.sort_values("timestep")
        .drop_duplicates("person_id")
        .loc[
            :,
            [
                "person_id",
                "age",
                "sex",
                "household_id",
                "vaccination_status",
                "vaccination_doses",
            ],
        ]
        .copy()
    )

    seed_ids = (
        infect.loc[infect["infector_person_id"].isna(), "infected_person_id"]
        .dropna()
        .unique()
    )
    agents["initial_infected"] = agents["person_id"].isin(seed_ids)
    agents.insert(0, "simulation_run_id", run_id)

    agent_ts = person.loc[
        :,
        [
            "timestep",
            "person_id",
            "current_location_id",
            "current_location_type",
            "is_masked",
            "infection_status",
            "infectious_variants",
            "symptomatic_variants",
        ],
    ].copy()
    agent_ts.insert(0, "simulation_run_id", run_id)

    infections = (
        infect.rename(
            columns={
                "infector_person_id": "infector_id",
                "infected_person_id": "infectee_id",
                "infection_location_id": "location_id",
                "infection_location_type": "location_type",
                "infected_masked": "infectee_masked",
            }
        )[
            [
                "timestep",
                "infector_id",
                "infectee_id",
                "location_id",
                "location_type",
                "variant",
                "infector_masked",
                "infectee_masked",
                "transmission_pair_age_diff",
            ]
        ]
        .copy()
    )
    infections.insert(0, "simulation_run_id", run_id)

    facilities = facility[
        [
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
        ]
    ].copy()
    facilities.insert(0, "simulation_run_id", run_id)

    chains_json = build_infection_chains(infections, run_id)

    missing_ids = set(infect["infector_person_id"].dropna()) \
                | set(infect["infected_person_id"].dropna())
    missing_ids -= set(agents["person_id"])

    if missing_ids:
        print(f"[i] {run_id}: adding {len(missing_ids)} stub agents from fallback log")
        stub = pd.DataFrame({
            "simulation_run_id": run_id,
            "person_id": list(missing_ids),
            "age": np.nan,
            "sex": np.nan,
            "household_id": np.nan,
            "vaccination_status": np.nan,
            "vaccination_doses": np.nan,
            "initial_infected": False,
        })
        agents = pd.concat([agents, stub], ignore_index=True)

    return agents, agent_ts, infections, facilities, chains_json


def build_infection_chains(infections_df: pd.DataFrame, run_id: str):
    G = nx.from_pandas_edgelist(
        infections_df.dropna(subset=["infector_id"]),
        source="infector_id",
        target="infectee_id",
        edge_attr=["timestep", "location_id"],
        create_using=nx.DiGraph(),
    )

    chains_out = []
    component_id = 0
    for component in nx.weakly_connected_components(G):
        sub = G.subgraph(component).copy()
        roots = [n for n in sub.nodes if sub.in_degree(n) == 0]
        if not roots:
            edges_sorted = sorted(
                sub.edges(data=True), key=lambda e: e[2]["timestep"]
            )
            roots = [edges_sorted[0][0]]
        root = min(roots)
        edges = [
            {
                "src": u,
                "dst": v,
                "t": int(d["timestep"]),
                "loc": int(d["location_id"]),
            }
            for u, v, d in sub.edges(data=True)
        ]
        timesteps = [e["t"] for e in edges]
        locs = [e["loc"] for e in edges]
        summary = {
            "length": len(sub.nodes),
            "depth": longest_path_length_dag(sub, root),
            "n_secondary": sub.out_degree(root),
            "facility_entropy": round(shannon_entropy(locs), 4),
            "span_minutes": (max(timesteps) - min(timesteps)) if timesteps else 0,
        }
        chain_dict = {
            "simulation_run_id": run_id,
            "chain_id": f"{run_id}-{component_id}",
            "root_case": int(root),
            "nodes": sorted(int(n) for n in sub.nodes),
            "edges": edges,
            "summary": summary,
        }
        chains_out.append(chain_dict)
        component_id += 1
    return chains_out


def write_outputs(
    agents: pd.DataFrame,
    agent_ts: pd.DataFrame,
    infections: pd.DataFrame,
    facilities: pd.DataFrame,
    chains_json: list,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    agents.to_parquet(os.path.join(output_dir, "agents.parquet"), index=False)
    agent_ts.to_parquet(
        os.path.join(output_dir, "agent_timelines.parquet"), index=False
    )
    infections.to_parquet(
        os.path.join(output_dir, "infections.parquet"), index=False
    )
    facilities.to_parquet(
        os.path.join(output_dir, "facilities.parquet"), index=False
    )
    with open(os.path.join(output_dir, "infection_chains.json"), "w") as f:
        json.dump(chains_json, f, indent=2)


def process_all_runs(raw_dir: str, output_base: str):
    """
    Iterate over every subdirectory in `raw_dir` named run<digits>,
    extract & standardize, and write each to `output_base/extracted_<run_id>/`.
    """
    if not os.path.isdir(raw_dir):
        raise ValueError(f"{raw_dir} is not a directory")
    os.makedirs(output_base, exist_ok=True)

    pattern = re.compile(r"^run\d+$")
    for run_folder in sorted(os.listdir(raw_dir)):
        if not pattern.match(run_folder):
            # skip anything not named like run1, run2, ...
            continue

        run_path = os.path.join(raw_dir, run_folder)
        print(f"[+] Processing {run_folder} …")
        agents, agent_ts, infections, facilities, chains_json = build_tables(run_path, run_folder)
        out_dir = os.path.join(output_base, f"extracted_{run_folder}")
        write_outputs(agents, agent_ts, infections, facilities, chains_json, output_dir=out_dir)
        print(f"[✓] Written to {out_dir}")

    print(f"[✔] All runs processed into {output_base}/")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-extract all simulator runs under a raw/ directory"
    )
    parser.add_argument(
        "--raw_dir",
        default="raw",
        help="Path to directory containing run1/, run2/, …"
    )
    parser.add_argument(
        "--output_base",
        default="all_extracted_logs",
        help="Where to write extracted_<run_id>/ subfolders"
    )
    args = parser.parse_args()

    process_all_runs(args.raw_dir, args.output_base)


if __name__ == "__main__":
    main()