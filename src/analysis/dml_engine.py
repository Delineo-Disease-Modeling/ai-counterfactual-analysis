'''Using EconML's Double Machine Learning (DML) engine for counterfactual analysis

Objective: To identify and quantify how much a change in a treatment variable (intervention parameter) will affect the outcome. 

To answer questions like: 
    # What is the effect of making 50% of people masked? 

S (shock) = infection 
A (attribute / treatment) = treatment parameters
Y (outcome) = total number of infections
Covariates = Age (for now, will add more later)


Outputs: 
- Average Treatment Effect (ATE) of the intervention on the outcome variable.
- 

'''

from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# making LinearDML model 

# making CausalForestDML model 

# calculating the Average Treatment Effect (ATE) using the DML model


# calculating the Conditional Average Treatment Effect (CATE) using the DML model

# plotting CATE using the DML model



