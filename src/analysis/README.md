### Analysis of Simulation Results 

Our analysis of simulation results is done in the following forms: 
1. Clustering 
2. Double Machine Learning (to calculate Average Treatment Effect of Interventions)
3. Flagging "High-Impact" counterfactuals

## Flagging "High Impact" Counterfactuals
The system flags parameter combinations where small changes could dramatically alter total infection outcomes.


# Process

1. Input: Simulation data with parameters and infection outcomes
2. AI Model (random forest): Learns which parameters are most sensitive
3. Output: Ranked list of high-impact parameter changes

Data needed: 
`{
    "run_id": "sim_001",
    "parameters": {
        "transmission_rate": 0.3,
        "intervention_day": 30,
        "compliance": 0.8
    },
    "total_infections": 150000
}`

# How the random forest model will work: 
1. Building many decision trees, each tree will learn a set of rules like "If mask rate > 0.27 and vaccination < 0.75, then infections might be around 180,000."

2. Random forest makes prediction about total infections (by averaging many cases, helps smooth noise)

3. Try the prediction again with a slight change in parameters

4. If the absolute value of the old prediction - the new prediction is large, it is a high impact counterfactual (maybe instead of a threshold, we consider the greates 10% of impact values to be high impact counterfactuals)
