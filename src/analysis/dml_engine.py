# Using EconML's Double Machine Learning (DML) engine for counterfactual analysis

''' S (shock): Infection
T (treatement / attribute): Treatment (e.g., masking)
Y (outcome):  Total infections
Covariates (X): Age (I might work with only this one to begin with and then add more)
Sample question to answer using this: What is the average treatment effect of increasing masking by 50% on infections, controlling for age?
'''

import pandas as pd 
import os 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from econml.dml import LinearDML 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

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
        num_infections = len(df) 
        avg_age = df['infected_age'].dropna().astype(float).mean()
        mask_rate = df['mask'].dropna().astype(float).mean()
        mask_rate = min(mask_rate, 1)
        records.append({
            "run_id": run,
            "num_infections": num_infections,
            "mask_rate": mask_rate,
            "avg_age": avg_age
        })
    except Exception as e: 
        print(f"Skipping {run}: {e}")

# extracting variables from main dataFrame 
df_summary = pd.DataFrame(records)
y = df_summary["num_infections"].values.ravel()
T = df_summary["mask_rate"].values
X = df_summary[["avg_age"]].values

model_y = GradientBoostingRegressor(random_state=0) #predicts infections
model_t = RandomForestRegressor(random_state=0) #predicts masking 

dml = LinearDML(
    model_y = model_y, 
    model_t = model_t, 
    discrete_treatment = False, # becase masking is a percentage 
    random_state = 0
)

dml.fit(Y = y, T = T, X = X)

T0 = T 
T1 = np.clip(T * 1.5, 0, 1)

effect = dml.effect(X = X, T0 = T0, T1 = T1)

ate = np.mean(effect)

print(f"Estimated ATE of +50% masking: {ate:.3f}")
print(df_summary["mask_rate"].value_counts())
print("Unique mask rates:", df_summary["mask_rate"].unique())
print("Unique num_infections:", df_summary["num_infections"].unique())
print("df_summary shape:", df_summary.shape)



sns.scatterplot(data=df_summary, x="mask_rate", y="num_infections")
plt.title("Mask Rate vs Infections")
plt.show()




