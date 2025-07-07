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

base_path = "../data/raw"
all_runs = sorted(os.listdir(base_path))

records = []

for run in all_runs: 
    run_path = os.path.join(base_path, run)
    file_path = os.path.join(run_path, "infection_logs.csv")

    try: 
        df = pd.read_csv(file_path)
        
    except Exception as e: 
        print(f"Skipping {run}: {e}")

# extracting variables from main dataFrame 
y = df['num_infections (Y)'].values
T = df['masking_rate (T)'].values
x = df['age (X)'].values



