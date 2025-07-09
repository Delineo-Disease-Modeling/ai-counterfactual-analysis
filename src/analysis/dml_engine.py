# Using EconML's Double Machine Learning (DML) engine for counterfactual analysis

''' S (shock): Infection
T (treatement / attribute): Treatment (e.g., masking)
Y (outcome):  Total infections
Covariates (X): Age (I might work with only this one to begin with and then add more)
Sample question to answer using this: What is the average treatment effect of increasing masking by 50% on infections, controlling for age?
'''

import pandas as pd 
import os 

# arranging data into this format 
# num_infections (Y) | masking_rate (T) | age (X)

base_path = "../../data/raw"
all_runs = sorted(os.listdir(base_path))

records = []

for run in all_runs: 
    run_path = os.path.join(base_path, run)
    file_path = os.path.join(run_path, "infection_logs.csv")

    try: 
        df = pd.read_csv(file_path)
        df = df[df['infected_person_id'].notna()]
        num_infections = len(df) - 1
        avg_age = df['infected_age'].dropna().astype(float).mean()
        mask_rate = df['mask'].dropna().astype(float).iloc[-1]
        mask_rate = min(mask_rate, 1)
        records.append({
            "run_id": run,
            "num_infections": num_infections,
            "mask_rate": mask_rate,
            "avg_age": avg_age
        })
        print(records)
    except Exception as e: 
        print(f"Skipping {run}: {e}")

# extracting variables from main dataFrame 
'''y = df['num_infections (Y)'].values
T = df['masking_rate (T)'].values
x = df['age (X)'].values'''



