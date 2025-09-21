##Tracking Software

agent_tracker.py contains a variety of functions for auditing runs which have been completed by the rerunner
using the csv files output by the simulation's logging feature.

Currently implemented: 
- track a specific agent's movements and infection history through the simulation.
- track the history of a specific location (who entered, left, and got infected) through the simulation.

To be implemented: 
- calculate the agent/s directly or indirectly responsible for the most infections.
- calculate the location/s where the most infections occurred.
- output the "family tree" of infectivity in a simulation in a visually appealing format.

To run the tracker, your current directory in the terminal MUST be (wherever you have your repos saved)/ai-counterfactual-analysis.
This is so the tracker can access the csv files in ai-counterfactual-analysis/data/raw.