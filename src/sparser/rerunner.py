import os
import requests
import shutil
import time

# Configurable variables
SIM_URL = "http://192.168.29.175:1880/simulation/"  # adjust port if needed
SIM_LOG_DIR = "../../../Simulation/simulation_logs/"            # where simulator writes CSVs
TARGET_DIR = "../../data/raw/"                     # where to store organized results

# List of intervention dictionaries to test
PARAMETER_SETS = [
    {'mask': 65.48696, 'vaccine': 80.45091, 'capacity': 58.07661, 'lockdown': 56.87906, 'selfiso': 30.24854},
    {'mask': 20.1, 'vaccine': 50.5, 'capacity': 40.0, 'lockdown': 60.0, 'selfiso': 70.0},
    # Add more if needed
]

# You can set optional fixed values if you want
DEFAULTS = {
    "location": "barnsdall",  # optional, will use config default if omitted
    "length": 720                # 12 hours * 60 minutes
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
        print(f"❌ Simulation {run_id} failed: {e}")
        return

    print(f"✅ Simulation {run_id} succeeded. Waiting for logs...")

    # Give simulator time to write logs
    time.sleep(5)  # adjust based on how long your sim takes

    run_folder = os.path.join(TARGET_DIR, f"run{run_id}")
    os.makedirs(run_folder, exist_ok=True)

    # Move CSVs from SIM_LOG_DIR to the new run folder
    for filename in os.listdir(SIM_LOG_DIR):
        if filename.endswith(".csv"):
            src = os.path.join(SIM_LOG_DIR, filename)
            dst = os.path.join(run_folder, filename)
            shutil.copy(src, dst)

    print(f"📁 Logs for run {run_id} saved in {run_folder}\n")

if __name__ == "__main__":
    os.makedirs(TARGET_DIR, exist_ok=True)

    for idx, param_set in enumerate(PARAMETER_SETS, start=1):
        run_simulation(param_set, idx)
