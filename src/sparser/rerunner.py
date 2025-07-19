"""
For generating logs for simulation with different parameters. 
Steps to use this script: 
1. Run the dmp (from the simulation repo by running `uvicorn api.dmp_api:app --reload`)
2. Run simulator by running `python app.py` in the simulation repo
3. Run this script, logs will be saved to data/raw/runX/ where X is the run number in the AI-counterfactual-analysis repo. 
"""
import os
import requests
import shutil
import time
from scenario_selector import load_parameter_space, latin_hypercube_scaled
import json

# Configurable variables
SIM_URL = "http://127.0.0.1:1880/simulation/"  
SIM_LOG_DIR = "../../../Simulation/simulation_logs/"  # where simulator writes CSVs
TARGET_DIR = "../../data/raw/"                     # where to store organized results

# List of intervention dictionaries to test

param_space = load_parameter_space('params.json')

# Latin Hypercube sampling

PARAMETER_SETS = latin_hypercube_scaled(param_space, n_samples = 200)


DEFAULTS = {
    "location": "barnsdall",  # optional, will use config default if omitted
    "length": 1440                # 12 hours * 60 minutes
}

def run_simulation(interventions, run_id):
    print(f"Running simulation {run_id} with interventions: {interventions}")

    # Combine defaults with current intervention parameters
    scaled = {k: v / 100 for k, v in interventions.items()}
    payload = {
        **DEFAULTS,
        **scaled
    }

    print("Sending payload to simulator:", payload)



    try:
        response = requests.post(SIM_URL, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Simulation {run_id} failed: {e}")
        return

    print(f"Simulation {run_id} succeeded. Waiting for logs...")

    # Give simulator time to write logs
    time.sleep(5)  

    run_folder = os.path.join(TARGET_DIR, f"run{run_id}")
    os.makedirs(run_folder, exist_ok=True)

    with open(os.path.join(run_folder, "interventions.json"), "w") as f:
        json.dump(interventions, f, indent=2)


    # Move CSVs from SIM_LOG_DIR to the new run folder
    for filename in os.listdir(SIM_LOG_DIR):
        if filename.endswith(".csv"):
            src = os.path.join(SIM_LOG_DIR, filename)
            dst = os.path.join(run_folder, filename)
            shutil.copy(src, dst)

    print(f"Logs for run {run_id} saved in {run_folder}\n")

    

if __name__ == "__main__":
    os.makedirs(TARGET_DIR, exist_ok=True)

    for idx, param_set in enumerate(PARAMETER_SETS, start=1):
        run_simulation(param_set, idx)
