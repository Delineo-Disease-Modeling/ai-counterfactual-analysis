## Sparser Modules

The sparser has the following files: 
1. scenario_selector.py: uses Latin Hypercube Sampling to select spread-out combinations of infection parameters. 
2. params.json: 6 main parameters
3. params_fixed.json: Only masking included, all other parameters become fixed and default. 
4. rerunner.py: Contacts the simulator, and saves logs for different simulator runs (with different parameters)
5. add_parameters.py: adds data about parameters used to the logs. 

