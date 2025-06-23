import sys
from pathlib import Path
import json

# STEP 1: Add Simulation repo to sys.path
sim_path = Path("../../../../Simulation/simulator").resolve()
sys.path.insert(0, str(sim_path))

# STEP 2: Import simulation function
from simulate import run_simulator

# STEP 3: Generate interventions
from scenario_selector import load_parameter_space, latin_hypercube_scaled
param_space = load_parameter_space('params.json')
samples = latin_hypercube_scaled(param_space, n_samples=50)

# STEP 4: Run simulations directly
from pathlib import Path

def save_log(run_id, data, base_dir="../../../data/raw"):
    run_folder = Path(base_dir) / f"run_{run_id:03}"
    run_folder.mkdir(parents=True, exist_ok=True)
    with open(run_folder / "log.json", "w") as f:
        json.dump(data, f, indent=2)

def batch_run(samples):
    for i, sample in enumerate(samples, start=1):
        config = {
            "length": 100,
            "location": "barnsdall",  # Default location
            **sample
        }
        print(f"[+] Running run_{i:03}")
        result = run_simulator(
            location=config["location"],
            max_length=config["length"],
            interventions={k: v for k, v in config.items() if k not in ["length", "location"]},
            save_file=False,
            enable_logging=True,
            log_dir=f"../../../data/raw/run_{i:03}"
        )
        save_log(i, result)

if __name__ == "__main__":
    batch_run(samples)
