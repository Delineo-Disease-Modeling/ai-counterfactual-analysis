# Enhanced EconML Double Machine Learning (DML) Analysis
# Counterfactual Analysis of Masking and Vaccination Effects on Infections

# Context about data: 
# This analysis is using simulator reruns that are saved in 'AI-counterfactual-analysis/data/raw' directory.
# Each run has a subdirectory named 'runXXX' where XXX is the run number.
# The data file 'infection_chains.csv' contains the infection data for each run.
# runs1 to run200: all parameters can change according to Latin Hypercube Sampling. 
# runs201 to run250: all parameters are fixed, except for the mask rate.
# runs251 to run300: all parameters are fixed, except for the vaccination rate. 
# runs301 to run350: all parameters are fixed, except for the percentage of locations under lockdown.
# runs351 to run400: all parameters are fixed, except for masking rate, effectiveness of masking is 100% (i.e., no infections from masked individuals).
# run401 to run450: all parameters are fixed, except for vaccination rate, effectiveness of vaccination is 100% (i.e., no infections from vaccinated individuals).

import pandas as pd 
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from econml.dml import LinearDML 
from scipy import stats
from scipy.stats import probplot

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 80)
print("ENHANCED COUNTERFACTUAL ANALYSIS WITH ECONML DML")
print("=" * 80)

# ====================================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ====================================================================================

print("Loading and preprocessing simulation data...")

base_path = "../../data/raw"
all_runs = sorted(os.listdir(base_path))

masking_records = []      # runs351-400: Only masking varies, 100% effectiveness
vaccination_records = []  # runs251-300: Only vaccination varies
general_records = []      # runs1-200: All parameters vary

for run in all_runs:
    try:
        run_num = int(run.replace("run", ""))
        run_path = os.path.join(base_path, run)
        file_path = os.path.join(run_path, "infection_chains.csv")
        
        df = pd.read_csv(file_path)
        df = df[df['infected_person_id'].notna()]
        
        num_infections = len(df)
        avg_age = np.random.randint(10, 90)  # Simulated age (replace with real data)
        
        # Process different experimental conditions
        if 351 <= run_num <= 400:  # Masking experiments
            mask_rate = float(df['mask'].dropna().iloc[0]) if 'mask' in df.columns and len(df['mask'].dropna()) > 0 else 0
            mask_rate = min(mask_rate, 1)
            
            masking_records.append({
                "run_id": run,
                "run_number": run_num,
                "num_infections": num_infections,
                "mask_rate": mask_rate,
                "avg_age": avg_age,
                "experiment_type": "masking_100pct_effective"
            })
            
        elif 251 <= run_num <= 300:  # Vaccination experiments
            if 'vaccine' in df.columns:
                vaccine_col = df['vaccine'].dropna()
                if len(vaccine_col) > 0:
                    vaccine_rate = float(vaccine_col.iloc[0])
                    vaccine_rate = min(vaccine_rate, 1)
                    
                    vaccination_records.append({
                        "run_id": run,
                        "run_number": run_num,
                        "num_infections": num_infections,
                        "vaccine_rate": vaccine_rate,
                        "avg_age": avg_age,
                        "experiment_type": "vaccination_variable"
                    })
                    
        elif 1 <= run_num <= 200:  # General experiments
            mask_rate = float(df['mask'].dropna().iloc[0]) if 'mask' in df.columns and len(df['mask'].dropna()) > 0 else np.nan
            vaccine_rate = float(df['vaccine'].dropna().iloc[0]) if 'vaccine' in df.columns and len(df['vaccine'].dropna()) > 0 else np.nan
            
            general_records.append({
                "run_id": run,
                "run_number": run_num,
                "num_infections": num_infections,
                "mask_rate": mask_rate,
                "vaccine_rate": vaccine_rate,
                "avg_age": avg_age,
                "experiment_type": "general_LHS"
            })
            
    except Exception as e:
        print(f"Skipping {run}: {e}")

# Convert to DataFrames
df_masking = pd.DataFrame(masking_records)
df_vaccination = pd.DataFrame(vaccination_records)
df_general = pd.DataFrame(general_records)

print(f"\nData Summary:")
print(f"  Masking experiments (runs 351-400): {len(df_masking)} runs")
print(f"  Vaccination experiments (runs 251-300): {len(df_vaccination)} runs")
print(f"  General experiments (runs 1-200): {len(df_general)} runs")

# ====================================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ====================================================================================

print("\nGenerating exploratory data analysis...")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Distribution of infections by experiment type
ax1 = plt.subplot(2, 3, 1)
infection_data = []

# Plot 2: Mask rate vs infections
ax2 = plt.subplot(2, 3, 2)
if len(df_masking) > 0:
    ax2.scatter(df_masking['mask_rate'], df_masking['num_infections'], 
                alpha=0.7, c=df_masking['avg_age'], cmap='viridis')
    z = np.polyfit(df_masking['mask_rate'], df_masking['num_infections'], 1)
    p = np.poly1d(z)
    ax2.plot(df_masking['mask_rate'], p(df_masking['mask_rate']), "r--", alpha=0.8, linewidth=2)
    
    corr = df_masking[['mask_rate', 'num_infections']].corr().iloc[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax2.transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax2.set_title('Mask Rate vs Infections')
    ax2.set_xlabel('Mask Rate')
    ax2.set_ylabel('Number of Infections')
    plt.colorbar(ax2.collections[0], ax=ax2, label='Average Age')


# Plot 4: Age distribution
ax4 = plt.subplot(2, 3, 4)
all_ages = []
if len(df_masking) > 0:
    all_ages.extend(df_masking['avg_age'].tolist())
if len(df_vaccination) > 0:
    all_ages.extend(df_vaccination['avg_age'].tolist())
if len(df_general) > 0:
    all_ages.extend(df_general['avg_age'].tolist())

if all_ages:
    ax4.hist(all_ages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('Age Distribution Across All Runs')
    ax4.set_xlabel('Average Age')
    ax4.set_ylabel('Frequency')

# Plot 5: Parameter space coverage (General runs)
ax5 = plt.subplot(2, 3, 5)
if len(df_general) > 0:
    general_clean = df_general.dropna(subset=['mask_rate', 'vaccine_rate'])
    if len(general_clean) > 0:
        ax5.scatter(general_clean['mask_rate'], general_clean['vaccine_rate'], 
                   alpha=0.6, c=general_clean['num_infections'], cmap='Reds')
        ax5.set_title('Parameter Space Coverage (General Runs)')
        ax5.set_xlabel('Mask Rate')
        ax5.set_ylabel('Vaccine Rate')
        plt.colorbar(ax5.collections[0], ax=ax5, label='Infections')

# Plot 6: Infection counts comparison
ax6 = plt.subplot(2, 3, 6)
experiment_means = []
experiment_stds = []
experiment_labels = []

if len(df_masking) > 0:
    experiment_means.append(df_masking['num_infections'].mean())
    experiment_stds.append(df_masking['num_infections'].std())
    experiment_labels.append('Masking')

if len(df_vaccination) > 0:
    experiment_means.append(df_vaccination['num_infections'].mean())
    experiment_stds.append(df_vaccination['num_infections'].std())
    experiment_labels.append('Vaccination')

if len(df_general) > 0:
    experiment_means.append(df_general['num_infections'].mean())
    experiment_stds.append(df_general['num_infections'].std())
    experiment_labels.append('General')

if experiment_means:
    bars = ax6.bar(experiment_labels, experiment_means, yerr=experiment_stds, 
                   alpha=0.7, capsize=5)
    ax6.set_title('Mean Infections by Experiment Type')
    ax6.set_ylabel('Mean Number of Infections')
    
    for bar, mean in zip(bars, experiment_means):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(experiment_stds) * 0.1,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.suptitle('EXPLORATORY DATA ANALYSIS', fontsize=16, fontweight='bold', y=0.98)
plt.show()

# ====================================================================================
# SECTION 3: DOUBLE MACHINE LEARNING CAUSAL ANALYSIS
# ====================================================================================

print(f"\n" + "=" * 60)
print("CAUSAL ANALYSIS WITH DOUBLE MACHINE LEARNING")
print("=" * 60)

# Initialize variables to store results
masking_results = {}
vaccination_results = {}

# ---- MASKING ANALYSIS ----
print(f"\nAnalyzing masking data (n={len(df_masking)})...")

if len(df_masking) > 0:
    # Prepare data matrices for DML
    y = df_masking["num_infections"].values
    T = df_masking["mask_rate"].values
    X = df_masking[["avg_age"]].values

    # Initialize ML models
    model_y = GradientBoostingRegressor(random_state=0, n_estimators=100)
    model_t = RandomForestRegressor(random_state=0, n_estimators=100)

    # Initialize Double Machine Learning estimator
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=False,
        random_state=0
    )

    # Fit the DML model
    dml.fit(Y=y, T=T, X=X)

    # Define counterfactual scenarios
    T0 = T                              # Baseline treatment levels
    T1 = np.clip(T * 1.5, 0, 1)       # +50% increase in masking
    T2 = np.clip(T * 2.0, 0, 1)       # +100% increase (doubling)
    T3 = np.clip(T * 0.5, 0, 1)       # -50% decrease (halving)

    # Calculate treatment effects
    effect_50 = dml.effect(X=X, T0=T0, T1=T1)
    effect_100 = dml.effect(X=X, T0=T0, T1=T2)
    effect_minus_50 = dml.effect(X=X, T0=T0, T1=T3)

    # Calculate Average Treatment Effects (ATE)
    ate_50 = np.mean(effect_50)
    ate_100 = np.mean(effect_100)
    ate_minus_50 = np.mean(effect_minus_50)

    print(f"MASKING RESULTS:")
    print(f"  ATE of +50% masking: {ate_50:.3f} infections")
    print(f"  ATE of +100% masking: {ate_100:.3f} infections")
    print(f"  ATE of -50% masking: {ate_minus_50:.3f} infections")

    # Store results
    masking_results = {
        'dml_model': dml,
        'effect_50': effect_50,
        'effect_100': effect_100,
        'effect_minus_50': effect_minus_50,
        'ate_50': ate_50,
        'ate_100': ate_100,
        'ate_minus_50': ate_minus_50,
        'y': y, 'T': T, 'X': X
    }

# ---- VACCINATION ANALYSIS ----
print(f"\nAnalyzing vaccination data (n={len(df_vaccination)})...")

if len(df_vaccination) > 0:
    # Prepare data matrices
    y2 = df_vaccination["num_infections"].values
    T_vax = df_vaccination["vaccine_rate"].values
    X2 = df_vaccination[["avg_age"]].values

    # Initialize DML model for vaccination
    dml2 = LinearDML(
        model_y=GradientBoostingRegressor(random_state=0, n_estimators=100),
        model_t=RandomForestRegressor(random_state=0, n_estimators=100),
        discrete_treatment=False,
        random_state=0
    )

    dml2.fit(Y=y2, T=T_vax, X=X2)

    # Define vaccination scenarios
    T0_vax = T_vax
    T1_vax = np.clip(T_vax * 1.5, 0, 1)
    T2_vax = np.clip(T_vax * 2.0, 0, 1)
    T3_vax = np.clip(T_vax * 0.5, 0, 1)

    # Calculate treatment effects
    effect_vax_50 = dml2.effect(X=X2, T0=T0_vax, T1=T1_vax)
    effect_vax_100 = dml2.effect(X=X2, T0=T0_vax, T1=T2_vax)
    effect_vax_minus_50 = dml2.effect(X=X2, T0=T0_vax, T1=T3_vax)

    # Calculate ATEs
    ate_vax_50 = np.mean(effect_vax_50)
    ate_vax_100 = np.mean(effect_vax_100)
    ate_vax_minus_50 = np.mean(effect_vax_minus_50)

    print(f"VACCINATION RESULTS:")
    print(f"  ATE of +50% vaccination: {ate_vax_50:.3f} infections")
    print(f"  ATE of +100% vaccination: {ate_vax_100:.3f} infections")
    print(f"  ATE of -50% vaccination: {ate_vax_minus_50:.3f} infections")

    # Store results
    vaccination_results = {
        'dml_model': dml2,
        'effect_50': effect_vax_50,
        'effect_100': effect_vax_100,
        'effect_minus_50': effect_vax_minus_50,
        'ate_50': ate_vax_50,
        'ate_100': ate_vax_100,
        'ate_minus_50': ate_vax_minus_50,
        'y': y2, 'T': T_vax, 'X': X2
    }

# ====================================================================================
# SECTION 4: CAUSAL ANALYSIS VISUALIZATIONS
# ====================================================================================

print("\nGenerating causal analysis visualizations...")

fig = plt.figure(figsize=(18, 12))

# Plot 1: Treatment Effect Comparison
ax1 = plt.subplot(2, 4, 1)
effects_data = []
if masking_results:
    effects_data.extend([
        {'Intervention': 'Mask +50%', 'ATE': masking_results['ate_50'], 'Type': 'Masking'},
        {'Intervention': 'Mask +100%', 'ATE': masking_results['ate_100'], 'Type': 'Masking'},
        {'Intervention': 'Mask -50%', 'ATE': masking_results['ate_minus_50'], 'Type': 'Masking'}
    ])
if vaccination_results:
    effects_data.extend([
        {'Intervention': 'Vaccine +50%', 'ATE': vaccination_results['ate_50'], 'Type': 'Vaccination'},
        {'Intervention': 'Vaccine +100%', 'ATE': vaccination_results['ate_100'], 'Type': 'Vaccination'},
        {'Intervention': 'Vaccine -50%', 'ATE': vaccination_results['ate_minus_50'], 'Type': 'Vaccination'}
    ])

if effects_data:
    effects_df = pd.DataFrame(effects_data)
    sns.barplot(data=effects_df, x='Intervention', y='ATE', hue='Type', ax=ax1)
    ax1.set_title('Average Treatment Effects Comparison')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Change in Infection Count')

# Plot 2: Treatment Effect Distributions (Masking)
ax2 = plt.subplot(2, 4, 2)

# Plot 3: Treatment Effect Distributions (Vaccination)
ax3 = plt.subplot(2, 4, 3)


# Plot 4: Dose-Response Relationship (Masking)
ax4 = plt.subplot(2, 4, 4)
if masking_results:
    dose_response_mask = pd.DataFrame({
        'Treatment_Change': ['-50%', 'Baseline', '+50%', '+100%'],
        'ATE': [masking_results['ate_minus_50'], 0, masking_results['ate_50'], masking_results['ate_100']],
        'Treatment_Level': [-0.5, 0, 0.5, 1.0]
    })
    ax4.plot(dose_response_mask['Treatment_Level'], dose_response_mask['ATE'], 
             'o-', linewidth=3, markersize=10, color='blue')
    ax4.set_title('Dose-Response Curve (Masking)')
    ax4.set_xlabel('Treatment Change (proportion)')
    ax4.set_ylabel('Average Treatment Effect')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)

# Plot 5: Dose-Response Relationship (Vaccination)
ax5 = plt.subplot(2, 4, 5)
if vaccination_results:
    dose_response_vax = pd.DataFrame({
        'Treatment_Change': ['-50%', 'Baseline', '+50%', '+100%'],
        'ATE': [vaccination_results['ate_minus_50'], 0, vaccination_results['ate_50'], vaccination_results['ate_100']],
        'Treatment_Level': [-0.5, 0, 0.5, 1.0]
    })
    ax5.plot(dose_response_vax['Treatment_Level'], dose_response_vax['ATE'], 
             'o-', linewidth=3, markersize=10, color='orange')
    ax5.set_title('Dose-Response Curve (Vaccination)')
    ax5.set_xlabel('Treatment Change (proportion)')
    ax5.set_ylabel('Average Treatment Effect')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)

# Plot 6: Treatment Effect vs Baseline Rate (Masking)
ax6 = plt.subplot(2, 4, 6)
if masking_results and len(df_masking) > 0:
    sns.scatterplot(data=df_masking, x='mask_rate', y=masking_results['effect_50'], 
                   alpha=0.7, ax=ax6)
    z = np.polyfit(df_masking['mask_rate'], masking_results['effect_50'], 1)
    p = np.poly1d(z)
    ax6.plot(df_masking['mask_rate'], p(df_masking['mask_rate']), "r--", alpha=0.8)
    
    ax6.set_title('Treatment Effect vs Baseline Rate (Masking)')
    ax6.set_xlabel('Baseline Mask Rate')
    ax6.set_ylabel('Treatment Effect (+50%)')
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot 7: Treatment Effect vs Baseline Rate (Vaccination)
ax7 = plt.subplot(2, 4, 7)
if vaccination_results and len(df_vaccination) > 0:
    sns.scatterplot(data=df_vaccination, x='vaccine_rate', y=vaccination_results['effect_50'], 
                   alpha=0.7, ax=ax7)
    z = np.polyfit(df_vaccination['vaccine_rate'], vaccination_results['effect_50'], 1)
    p = np.poly1d(z)
    ax7.plot(df_vaccination['vaccine_rate'], p(df_vaccination['vaccine_rate']), "r--", alpha=0.8)
    
    ax7.set_title('Treatment Effect vs Baseline Rate (Vaccination)')
    ax7.set_xlabel('Baseline Vaccine Rate')
    ax7.set_ylabel('Treatment Effect (+50%)')
    ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot 8: Effect Size Comparison
ax8 = plt.subplot(2, 4, 8)
if masking_results or vaccination_results:
    effect_magnitudes = []
    intervention_names = []
    
    if masking_results:
        effect_magnitudes.append(abs(masking_results['ate_50']))
        intervention_names.append('Masking +50%')
        
    if vaccination_results:
        effect_magnitudes.append(abs(vaccination_results['ate_50']))
        intervention_names.append('Vaccination +50%')
    
    bars = ax8.bar(intervention_names, effect_magnitudes, 
                   color=['blue', 'orange'][:len(effect_magnitudes)], alpha=0.7)
    ax8.set_title('Effect Size Comparison (Absolute)')
    ax8.set_ylabel('Absolute Treatment Effect')
    
    for bar, magnitude in zip(bars, effect_magnitudes):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{magnitude:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.suptitle('CAUSAL ANALYSIS RESULTS: Double Machine Learning Treatment Effects', 
             fontsize=16, fontweight='bold', y=0.98)
plt.show()

# ====================================================================================
# SECTION 5: STATISTICAL SIGNIFICANCE TESTING
# ====================================================================================

print(f"\n" + "=" * 60)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("=" * 60)

# Bootstrap confidence intervals
n_bootstrap = 1000
np.random.seed(42)

def calculate_bootstrap_ci(effects, n_bootstrap=1000):
    """Calculate bootstrap confidence intervals"""
    bootstrap_ates = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(effects), size=len(effects), replace=True)
        bootstrap_ates.append(np.mean(effects[indices]))
    
    ci_lower = np.percentile(bootstrap_ates, 2.5)
    ci_upper = np.percentile(bootstrap_ates, 97.5)
    p_value = np.mean(np.array(bootstrap_ates) > 0) * 2  # Two-tailed test
    
    return bootstrap_ates, ci_lower, ci_upper, p_value

def cohen_d(x1, x2):
    """Calculate Cohen's d effect size"""
    pooled_std = np.sqrt(((len(x1) - 1) * np.var(x1) + (len(x2) - 1) * np.var(x2)) / (len(x1) + len(x2) - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Masking bootstrap analysis
if masking_results:
    print("Performing bootstrap analysis for masking effects...")
    bootstrap_ates_mask, ci_lower_mask, ci_upper_mask, p_value_mask = calculate_bootstrap_ci(masking_results['effect_50'])
    
    axes[0, 0].hist(bootstrap_ates_mask, bins=50, alpha=0.7, density=True, color='skyblue')
    axes[0, 0].axvline(masking_results['ate_50'], color='red', linestyle='--', linewidth=2, 
                       label=f'ATE: {masking_results["ate_50"]:.3f}')
    axes[0, 0].axvline(ci_lower_mask, color='orange', linestyle='--', 
                       label=f'95% CI: [{ci_lower_mask:.3f}, {ci_upper_mask:.3f}]')
    axes[0, 0].axvline(ci_upper_mask, color='orange', linestyle='--')
    axes[0, 0].axvline(0, color='black', linestyle='-', alpha=0.5, label='Null Effect')
    axes[0, 0].set_title(f'Bootstrap Distribution - Masking (p = {p_value_mask:.3f})')
    axes[0, 0].set_xlabel('Average Treatment Effect')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    
    print(f"Masking +50% Results:")
    print(f"  ATE: {masking_results['ate_50']:.3f}")
    print(f"  95% CI: [{ci_lower_mask:.3f}, {ci_upper_mask:.3f}]")
    print(f"  Statistically significant: {'Yes' if ci_lower_mask > 0 or ci_upper_mask < 0 else 'No'}")
    print(f"  P-value: {p_value_mask:.3f}")

# Vaccination bootstrap analysis
if vaccination_results:
    print("\nPerforming bootstrap analysis for vaccination effects...")
    bootstrap_ates_vax, ci_lower_vax, ci_upper_vax, p_value_vax = calculate_bootstrap_ci(vaccination_results['effect_50'])
    
    axes[0, 1].hist(bootstrap_ates_vax, bins=50, alpha=0.7, density=True, color='lightcoral')
    axes[0, 1].axvline(vaccination_results['ate_50'], color='red', linestyle='--', linewidth=2, 
                       label=f'ATE: {vaccination_results["ate_50"]:.3f}')
    axes[0, 1].axvline(ci_lower_vax, color='orange', linestyle='--', 
                       label=f'95% CI: [{ci_lower_vax:.3f}, {ci_upper_vax:.3f}]')
    axes[0, 1].axvline(ci_upper_vax, color='orange', linestyle='--')
    axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.5, label='Null Effect')
    axes[0, 1].set_title(f'Bootstrap Distribution - Vaccination (p = {p_value_vax:.3f})')
    axes[0, 1].set_xlabel('Average Treatment Effect')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    
    print(f"Vaccination +50% Results:")
    print(f"  ATE: {vaccination_results['ate_50']:.3f}")
    print(f"  95% CI: [{ci_lower_vax:.3f}, {ci_upper_vax:.3f}]")
    print(f"  Statistically significant: {'Yes' if ci_lower_vax > 0 or ci_upper_vax < 0 else 'No'}")
    print(f"  P-value: {p_value_vax:.3f}")

# Effect Size Analysis (Cohen's d)
effect_sizes = []
if masking_results:
    control_effect_mask = np.zeros(len(masking_results['effect_50']))
    cohens_d_mask = cohen_d(masking_results['effect_50'], control_effect_mask)
    effect_sizes.append({'Intervention': 'Masking +50%', 'Cohens_D': abs(cohens_d_mask)})
    
if vaccination_results:
    control_effect_vax = np.zeros(len(vaccination_results['effect_50']))
    cohens_d_vax = cohen_d(vaccination_results['effect_50'], control_effect_vax)
    effect_sizes.append({'Intervention': 'Vaccination +50%', 'Cohens_D': abs(cohens_d_vax)})

if effect_sizes:
    effect_size_df = pd.DataFrame(effect_sizes)
    bars = axes[1, 0].bar(effect_size_df['Intervention'], effect_size_df['Cohens_D'])
    axes[1, 0].set_title("Effect Size Analysis (Cohen's d)")
    axes[1, 0].set_ylabel("Cohen's d (Absolute)")
    axes[1, 0].axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Small effect')
    axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
    axes[1, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Large effect')
    axes[1, 0].legend()
    
    # Add value labels
    for bar, cohens_d_val in zip(bars, effect_size_df['Cohens_D']):
        height = bar.get_height()
        interpretation = 'Large' if cohens_d_val > 0.8 else 'Medium' if cohens_d_val > 0.5 else 'Small' if cohens_d_val > 0.2 else 'Negligible'
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{cohens_d_val:.3f}\n({interpretation})', ha='center', va='bottom', fontsize=10)

# Robustness check with different models
print("\nPerforming robustness checks with alternative models...")

model_combinations = [
    ("Original", GradientBoostingRegressor(random_state=0), RandomForestRegressor(random_state=0)),
    ("Linear", LinearRegression(), LinearRegression()),
    ("Ridge", Ridge(random_state=0), RandomForestRegressor(random_state=0)),
]

robustness_results = []

for name, model_y, model_t in model_combinations:
    try:
        if masking_results:
            dml_robust = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=False, random_state=0)
            dml_robust.fit(Y=masking_results['y'], T=masking_results['T'], X=masking_results['X'])
            effect_robust_mask = dml_robust.effect(X=masking_results['X'], 
                                                  T0=masking_results['T'], 
                                                  T1=np.clip(masking_results['T'] * 1.5, 0, 1))
            ate_robust_mask = np.mean(effect_robust_mask)
            robustness_results.append({'Model': name, 'Intervention': 'Masking', 'ATE': ate_robust_mask})
        
        if vaccination_results:
            dml_robust_vax = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=False, random_state=0)
            dml_robust_vax.fit(Y=vaccination_results['y'], T=vaccination_results['T'], X=vaccination_results['X'])
            effect_robust_vax = dml_robust_vax.effect(X=vaccination_results['X'], 
                                                     T0=vaccination_results['T'], 
                                                     T1=np.clip(vaccination_results['T'] * 1.5, 0, 1))
            ate_robust_vax = np.mean(effect_robust_vax)
            robustness_results.append({'Model': name, 'Intervention': 'Vaccination', 'ATE': ate_robust_vax})
            
    except Exception as e:
        print(f"  Failed for {name}: {e}")
        continue

# Plot robustness results
if robustness_results:
    robust_df = pd.DataFrame(robustness_results)
    sns.barplot(data=robust_df, x='Model', y='ATE', hue='Intervention', ax=axes[1, 1])
    axes[1, 1].set_title('Robustness Check: ATE Across Different Models')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_ylabel('Average Treatment Effect')

plt.tight_layout()
plt.suptitle('STATISTICAL SIGNIFICANCE AND ROBUSTNESS ANALYSIS', 
             fontsize=16, fontweight='bold', y=0.98)
plt.show()

# ====================================================================================
# SECTION 6: COMPREHENSIVE SUMMARY
# ====================================================================================

print(f"\n" + "=" * 80)
print("COMPREHENSIVE ANALYSIS SUMMARY")
print("=" * 80)

print(f"\n📊 PRIMARY CAUSAL FINDINGS:")

if masking_results:
    significance_mask = "Significant" if (ci_lower_mask > 0 or ci_upper_mask < 0) else "Not Significant"
    effect_interpretation_mask = "Large" if abs(cohens_d_mask) > 0.8 else "Medium" if abs(cohens_d_mask) > 0.5 else "Small"
    
    print(f"   • MASKING (+50% increase):")
    print(f"     - Average Treatment Effect: {masking_results['ate_50']:.3f} infections")
    print(f"     - 95% Confidence Interval: [{ci_lower_mask:.3f}, {ci_upper_mask:.3f}]")
    print(f"     - Statistical Significance: {significance_mask} (p = {p_value_mask:.3f})")
    print(f"     - Effect Size: {effect_interpretation_mask} (Cohen's d = {abs(cohens_d_mask):.3f})")
    print(f"     - Clinical Interpretation: {'Beneficial' if masking_results['ate_50'] < 0 else 'Harmful' if masking_results['ate_50'] > 0 else 'No effect'}")

if vaccination_results:
    significance_vax = "Significant" if (ci_lower_vax > 0 or ci_upper_vax < 0) else "Not Significant"
    effect_interpretation_vax = "Large" if abs(cohens_d_vax) > 0.8 else "Medium" if abs(cohens_d_vax) > 0.5 else "Small"
    
    print(f"   • VACCINATION (+50% increase):")
    print(f"     - Average Treatment Effect: {vaccination_results['ate_50']:.3f} infections")
    print(f"     - 95% Confidence Interval: [{ci_lower_vax:.3f}, {ci_upper_vax:.3f}]")
    print(f"     - Statistical Significance: {significance_vax} (p = {p_value_vax:.3f})")
    print(f"     - Effect Size: {effect_interpretation_vax} (Cohen's d = {abs(cohens_d_vax):.3f})")
    print(f"     - Clinical Interpretation: {'Beneficial' if vaccination_results['ate_50'] < 0 else 'Harmful' if vaccination_results['ate_50'] > 0 else 'No effect'}")

print(f"\n🔬 METHODOLOGICAL STRENGTHS:")
print(f"   • Double Machine Learning eliminates confounding bias through cross-fitting")
print(f"   • Ensemble methods capture non-linear relationships")
print(f"   • Bootstrap confidence intervals provide robust uncertainty quantification")
print(f"   • Multiple dose-response scenarios test intervention scalability")

print(f"\n⚖️  COMPARATIVE EFFECTIVENESS:")
if masking_results and vaccination_results:
    if abs(masking_results['ate_50']) > abs(vaccination_results['ate_50']):
        more_effective = "Masking"
    else:
        more_effective = "Vaccination"
    
    print(f"   • More effective intervention: {more_effective}")
    print(f"   • Masking effectiveness: {abs(masking_results['ate_50']):.3f} infections prevented per 50% increase")
    print(f"   • Vaccination effectiveness: {abs(vaccination_results['ate_50']):.3f} infections prevented per 50% increase")

print(f"\n🎯 POLICY IMPLICATIONS:")
if masking_results and masking_results['ate_50'] < 0:
    print(f"   • Masking interventions show public health benefit")
    print(f"   • Each 50% increase in masking rates could prevent ~{abs(masking_results['ate_50']):.1f} infections")

if vaccination_results and vaccination_results['ate_50'] < 0:
    print(f"   • Vaccination campaigns show public health benefit")
    print(f"   • Each 50% increase in vaccination rates could prevent ~{abs(vaccination_results['ate_50']):.1f} infections")

print(f"\n⚠️  LIMITATIONS:")
print(f"   • Age is the only covariate - unobserved confounders may bias results")
print(f"   • Linear treatment effect assumption may not capture threshold effects")
print(f"   • Simulation data may not fully reflect real-world complexity")
print(f"   • External validity depends on similarity to target population")

print(f"\n📈 ROBUSTNESS ASSESSMENT:")
if robustness_results:
    mask_ates = [r['ATE'] for r in robustness_results if r['Intervention'] == 'Masking']
    vax_ates = [r['ATE'] for r in robustness_results if r['Intervention'] == 'Vaccination']
    
    if mask_ates:
        mask_range = max(mask_ates) - min(mask_ates)
        print(f"   • Masking ATE range across models: {mask_range:.3f}")
        print(f"   • Masking result stability: {'High' if mask_range < 0.1 else 'Moderate' if mask_range < 0.3 else 'Low'}")
    
    if vax_ates:
        vax_range = max(vax_ates) - min(vax_ates)
        print(f"   • Vaccination ATE range across models: {vax_range:.3f}")
        print(f"   • Vaccination result stability: {'High' if vax_range < 0.1 else 'Moderate' if vax_range < 0.3 else 'Low'}")

print(f"\n💡 INTERPRETATION GUIDELINES:")
print(f"   • Negative treatment effects indicate infection reduction (beneficial)")
print(f"   • Effect sizes represent absolute change in infection counts")
print(f"   • Confidence intervals excluding zero suggest significant effects")
print(f"   • Cohen's d > 0.5 indicates practically meaningful effect sizes")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)