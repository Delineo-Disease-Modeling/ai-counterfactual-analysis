

# Delineo Data Analysis: AI-Augmented Counterfactual Analysis of Infection Chains

## Overview

This project develops an AI-augmented framework to analyze infection simulations by identifying pivotal moments where small behavioral or policy changes could significantly alter outcomes. Leveraging causal inference, clustering, and counterfactual modeling, the goal is to build tools that enable effective decision-making based on simulation data.

## Objectives

- **Optimize Simulation Pipeline**: Improve modularity and efficiency of the simulation engine to support large-scale scenario testing.
- **Structure and Store Data**: Design and implement a robust schema for capturing infection chains, agent behaviors, and policy states.
- **Extract Actionable Insights**: Apply advanced machine learning and causal inference techniques to uncover moments of critical impact.

## Methodology

1. **Simulation Optimization**
   - Refactor simulation code for modular execution and efficient logging.
   - Capture key metadata such as infection timestamps, infector IDs, and policy conditions.

2. **Data Structuring and Extraction**
   - Convert simulation outputs to structured formats (e.g., CSV, JSON) for analysis.
   - Validate and organize data into analyzable schemas for infection chains and agent states.

3. **Scenario Generation and Re-execution**
   - Implement the `sparser` module to run multiple parameterized scenarios.
   - Enable re-running simulations with controlled tweaks to assess causal impact.

4. **Clustering and Visualization**
   - Use clustering algorithms (KMeans, DBSCAN, Agglomerative) to identify patterns across simulations.
   - Apply dimensionality reduction (PCA, t-SNE) for interpretability.

5. **Causal Inference**
   - Employ Double Machine Learning (DML) and Causal Forests (via EconML) to quantify treatment effects.
   - Enable policy evaluation using interpretable models such as LinearDML.

6. **Counterfactual Flagging**
   - Build a rule-based and ML-powered system to flag high-impact simulations.
   - Use classification models (e.g., Random Forest, Logistic Regression) to predict impactful interventions.

## Deliverables

- **Sparser Module**: A parameter exploration tool to identify and re-run scenarios with the highest potential impact.
- **Counterfactual Analysis Framework**: An end-to-end system for evaluating policy alternatives in silico.
- **Clustering and Flagging Tools**: AI-driven techniques for grouping similar scenarios and identifying impactful deviations.

## Role of AI

Artificial intelligence is used to:
- Uncover hidden structure in complex simulation outputs.
- Detect and quantify the causal effect of individual decisions or policies.
- Prioritize scenarios for deeper investigation based on counterfactual impact.

## Technologies

- **Languages**: Python, Jupyter Notebooks
- **Libraries**: `scikit-learn`, `econml`, `matplotlib`, `seaborn`, `pandas`, `numpy`
- **Data Formats**: CSV, JSON
- **Models**: KMeans, DBSCAN, Agglomerative Clustering, DML, Random Forest, Logistic Regression

## Extractor Usage 

#Usage:
`python src/data_pipeline/extractor.py --raw_dir data/raw --output_dir all_extracted_logs`
`python src/data_pipeline/validate_extracted.py --base_dir all_extracted_logs`

## Future Work

- Integrating trade-off analysis for policy interventions (e.g., public health vs. economic cost).
- Improving simulation scalability and performance.
- Extending causal estimation methods to more complex treatments and temporal interventions.

