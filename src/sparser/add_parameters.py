# Script to add infection parameters to infection_logs.csv for each run in the simulation logs.
# This script goes through every run folder in data > raw and adds infection parameters from interventions.json to infection_logs.csv
# this can be run after the simulation runs are complete to ensure that all infection parameters are included in the logs.

import json 
import pandas as pd 
import os

for (root, dirs, files) in os.walk("../../data/raw/"):
    for dir_name in dirs:
        run_folder = os.path.join(root, dir_name)
        interventions_file = os.path.join(run_folder, "interventions.json")
        infection_logs_file = os.path.join(run_folder, "infection_logs.csv")

        if not os.path.exists(infection_logs_file):
            print(f"Skipping {run_folder} - no infection logs found.")
            continue

        if not os.path.exists(interventions_file):
            print(f"Skipping {run_folder} - no interventions file found.")
            continue

        with open(interventions_file, 'r') as f:
            interventions = json.load(f)

        # Read infection logs
        infection_df = pd.read_csv(infection_logs_file)

        # Add intervention parameters to the DataFrame
        for key, value in interventions.items():
            infection_df[key] = value

        # Save the updated DataFrame back to CSV
        infection_df.to_csv(infection_logs_file, index=False)
        print(f"Updated {infection_logs_file} with intervention parameters in {run_folder}.")