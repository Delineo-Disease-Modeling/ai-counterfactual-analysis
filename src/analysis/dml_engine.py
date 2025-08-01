# Using EconML's Double Machine Learning (DML) engine for counterfactual analysis

''' 
S (shock): Infection  
T (treatment / attribute): Treatment (e.g., masking)  
Y (outcome):  Total infections  
Covariates (X): Age (starting with this one)  
Question: What is the average treatment effect of increasing masking by 50% on infections, controlling for age?


Context about data: 
This analysis is using simulator reruns that are saved in 'AI-counterfactual-analysis/data/raw' directory.
Each run has a subdirectory named 'runXXX' where XXX is the run number.
The data file 'infection_chains.csv' contains the infection data for each run.
runs1 to run200: all parameters can change according to Latin Hypercube Sampling. 
runs201 to run250: all parameters are fixed, except for the mask rate.
runs251 to run300: all parameters are fixed, except for the vaccination rate. 
runs301 to run350: all parameters are fixed, except for the percentage of locations under lockdown.
'''

import pandas as pd 
import os 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from econml.dml import LinearDML 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import random 

# Step 1: Gather the data
base_path = "../../data/raw"
all_runs = sorted(os.listdir(base_path))

records = []

for run in all_runs:
    try:
        # Only include runs from run201 to run250
        run_num = int(run.replace("run", ""))
        if 201 <= run_num <= 250:
            run_path = os.path.join(base_path, run)
            file_path = os.path.join(run_path, "infection_chains.csv")

            df = pd.read_csv(file_path)
            df = df[df['infected_person_id'].notna()]
            num_infections = len(df)
            avg_age = random.randint(10, 90)
            mask_rate = float(df['mask'].dropna().iloc[0])
            mask_rate = min(mask_rate, 1)

            records.append({
                "run_id": run,
                "num_infections": num_infections,
                "mask_rate": mask_rate,
                "avg_age": avg_age
            })
    except Exception as e:
        print(f"Skipping {run}: {e}")

# Step 2: Create DataFrame and apply DML
df_summary = pd.DataFrame(records)
y = df_summary["num_infections"].values.ravel()
T = df_summary["mask_rate"].values
X = df_summary[["avg_age"]].values

model_y = GradientBoostingRegressor(random_state=0)
model_t = RandomForestRegressor(random_state=0)

dml = LinearDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=False,
    random_state=0
)

dml.fit(Y=y, T=T, X=X)

T0 = T
T1 = np.clip(T * 1.5, 0, 1)

effect = dml.effect(X=X, T0=T0, T1=T1)
ate = np.mean(effect)

print(f"Estimated ATE of +50% masking: {ate:.3f}")
print(df_summary["mask_rate"].value_counts())
print("Unique mask rates:", df_summary["mask_rate"].unique())
print("Unique num_infections:", df_summary["num_infections"].unique())
print("df_summary shape:", df_summary.shape)

# Store mask data for plotting
mask_data = df_summary.copy()

# ATE for Vaccination Rates

records2 = []

for run in all_runs:
    try:
        # Only include runs from run251 to run300
        run_num = int(run.replace("run", ""))
        if 251 <= run_num <= 300:
            run_path = os.path.join(base_path, run)
            file_path = os.path.join(run_path, "infection_chains.csv")

            df = pd.read_csv(file_path)
            df = df[df['infected_person_id'].notna()]
            num_infections = len(df)
            avg_age = random.randint(10, 90)
            vaccine_rate = float(df['vaccine'].dropna().iloc[0])
            vaccine_rate = min(vaccine_rate, 1)

            records2.append({
                "run_id": run,
                "num_infections": num_infections,
                "vaccine": vaccine_rate,
                "avg_age": avg_age
            })
    except Exception as e:
        print(f"Skipping {run}: {e}")

# Step 2: Create DataFrame and apply DML
df_summary2 = pd.DataFrame(records2)
y = df_summary2["num_infections"].values.ravel()
T = df_summary2["vaccine"].values
X = df_summary2[["avg_age"]].values

model_y = GradientBoostingRegressor(random_state=0)
model_t = RandomForestRegressor(random_state=0)

dml = LinearDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=False,
    random_state=0
)

dml.fit(Y=y, T=T, X=X)

T0 = T
T1 = np.clip(T * 1.5, 0, 1)

effect = dml.effect(X=X, T0=T0, T1=T1)
ate = np.mean(effect)

print("----------------------------------")
print(f"Estimated ATE of +50% vaccine: {ate:.3f}")
print(df_summary2["vaccine"].value_counts())
print("Unique vaccine rates:", df_summary2["vaccine"].unique())
print("Unique num_infections:", df_summary2["num_infections"].unique())
print("df_summary2 shape:", df_summary2.shape)

# Create a grid of scatterplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Mask Rate vs Infections
sns.scatterplot(data=mask_data, x="mask_rate", y="num_infections", ax=axes[0])
axes[0].set_title("Mask Rate vs Infections (Runs 201–250)")
axes[0].set_xlabel("Mask Rate")
axes[0].set_ylabel("Number of Infections")

# Plot 2: Vaccine Rate vs Infections
sns.scatterplot(data=df_summary2, x="vaccine", y="num_infections", ax=axes[1])
axes[1].set_title("Vaccine Rate vs Infections (Runs 251–300)")
axes[1].set_xlabel("Vaccine Rate")
axes[1].set_ylabel("Number of Infections")

plt.tight_layout()
plt.show()