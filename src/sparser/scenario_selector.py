# program to generate a sample of parameter combinations

import json
import numpy as np 
import itertools # to generate combinations
import random 
from scipy.stats import qmc

# converts json file to python dictionary 
def load_parameter_space(json_path): 
    with open(json_path, 'r') as file:
        return json.load(file)
    

# generating all combinations of parameters
# to be used with random sampling 
def generate_param_combinations(param_space): 
    param_values = {}
    for param, cfg in param_space.items(): 
        values = list(np.arange(cfg['min'], cfg['max'] + cfg['step'], cfg['step']))
        param_values[param] = values 

    keys = list(param_values.keys())
    product = list(itertools.product(*(param_values[key] for key in keys)))

    param_grid = [dict(zip(keys, vals)) for vals in product]
    return param_grid 


# random sampling from parameter grid 
def sample_random(grid, k = 20, seed = 40): 
    # random.seed(seed)
    return random.sample(grid, min(k, len(grid)))

# Latin Hypercube sampling is a sampling technique that ensures we explore parameter space evenly and efficiently. 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html
def latin_hypercube_scaled(param_space, n_samples = 50, seed = 30): 
    '''
    Generates Latin Hypercube samples from the parameter space.
    Args:
        param_space (dict): Dictionary containing parameter configurations with 'min', 'max', and 'step' keys. 
        n_samples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
    Returns:
        List of dictionaries, each representing a sample configuration with parameter values.
    '''
    keys = list(param_space.keys())
    d = len(keys)
    l_bounds = [param_space[k]['min'] for k in keys]
    u_bounds = [param_space[k]['max'] for k in keys]

    sampler = qmc.LatinHypercube(d = d, seed = seed)
    sample = sampler.random(n = n_samples)
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    samples = []
    for row in sample_scaled: 
        config = {k: float(round(v, 5)) for k, v in zip(keys, row)}
        samples.append(config)

    return samples # samples is a list of dictionaries, each representing a sample configuration with parameter values

if __name__ == "__main__":
    param_space = load_parameter_space('params.json')

    # Latin Hypercube sampling
    lhs_samples = latin_hypercube_scaled(param_space, n_samples = 50)
    for i, s in enumerate(lhs_samples):
        print(f"Sample {i + 1}: {s}")


    '''
    # Random sampling 
    full_grid = generate_param_combinations(param_space)
    print(f"Total combinations: {len(full_grid)}")

    sampled_runs = sample_random(full_grid, k=20)

    for i, run in enumerate(sampled_runs):
        print(f"Run {i + 1}: {run}")

    '''
