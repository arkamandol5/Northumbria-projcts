### Detailed Equations for Variable Elimination and Maximum Likelihood Estimator in Bayesian Networks

#### Bayesian Networks Overview

A Bayesian Network (BN) is a directed acyclic graph (DAG) that represents a set of random variables and their conditional dependencies via edges. Each node in the graph corresponds to a random variable, and the edges represent conditional dependencies.

#### Maximum Likelihood Estimator (MLE)

Given a dataset \( X = \{x_1, x_2, \ldots, x_n\} \) consisting of \( n \) independent and identically distributed (i.i.d) observations, we aim to estimate the parameter(s) \( \theta \) of the underlying probability distribution \( P(X|\theta) \).

1. **Likelihood Function**:
   \[
   L(\theta|X) = P(X|\theta) = \prod_{i=1}^{n} P(x_i|\theta)
   \]

2. **Log-Likelihood Function**:
   \[
   \ell(\theta|X) = \log L(\theta|X) = \log \left( \prod_{i=1}^{n} P(x_i|\theta) \right) = \sum_{i=1}^{n} \log P(x_i|\theta)
   \]

3. **Maximum Likelihood Estimator**:
   \[
   \hat{\theta} = \arg \max_{\theta} \ell(\theta|X)
   \]
   This involves solving:
   \[
   \frac{\partial \ell(\theta|X)}{\partial \theta} = 0
   \]

#### Example: MLE for Normal Distribution

For a normal distribution with unknown mean \( \mu \) and variance \( \sigma^2 \):
1. **Probability Density Function**:
   \[
   f(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
   \]

2. **Log-Likelihood Function**:
   \[
   \ell(\mu, \sigma^2|X) = \sum_{i=1}^{n} \log f(x_i|\mu, \sigma^2) = -\frac{n}{2} \log (2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2
   \]

3. **Estimates**:
   \[
   \hat{\mu} = \frac{1}{n} \sum_{i=1}^{n} x_i
   \]
   \[
   \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{\mu})^2
   \]

#### Variable Elimination Method

Variable Elimination (VE) is an algorithm used for exact inference in Bayesian Networks. It systematically eliminates variables from the network by summing them out, reducing the complexity of inference.

##### Steps of Variable Elimination

1. **Factor Representation**:
    - Represent each CPD \( P(X_i | \text{parents}(X_i)) \) as a factor \( \phi \).

2. **Multiplication of Factors**:
    - Combine factors involving the variable to be eliminated:
      \[
      \psi(X) = \prod_{j} \phi_j
      \]
      where \( \phi_j \) are factors involving \( X \).

3. **Summing Out Variables**:
    - Sum out variables not of interest:
      \[
      \sum_{X} \psi(X) = \sum_{X} \prod_{j} \phi_j
      \]
      This results in a new factor that no longer includes \( X \).

4. **Repeat**:
    - Repeat the multiplication and summing out for each variable to be eliminated.

5. **Normalization**:
    - Normalize the resulting factor to get a proper probability distribution.

##### Example

Consider a Bayesian Network with three variables \( A \), \( B \), and \( C \), where \( A \rightarrow B \rightarrow C \). We aim to compute \( P(A|C) \).

1. **Initial Factors**:
   \[
   \phi_1(A) = P(A)
   \]
   \[
   \phi_2(B|A) = P(B|A)
   \]
   \[
   \phi_3(C|B) = P(C|B)
   \]

2. **Multiply Factors Involving \( B \)**:
   \[
   \psi(B) = \phi_2(B|A) \cdot \phi_3(C|B)
   \]

3. **Sum Out \( B \)**:
   \[
   \sum_{B} \psi(B) = \sum_{B} P(B|A) \cdot P(C|B)
   \]

4. **Resulting Factor**:
   The resulting factor after summing out \( B \):
   \[
   \phi'(A) = P(A) \cdot \sum_{B} P(B|A) \cdot P(C|B)
   \]

5. **Normalize**:
   Normalize to obtain \( P(A|C) \).

#### Implementing Variable Elimination in Python

Below is a sample code for Bayesian Network inference using Variable Elimination:

```python
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import numpy as np

# Define the model structure
model_structure = [
    ('Month', 'crime_count_binned'),
    ('economically_inactive_percent_binned', 'crime_count_binned'),
    ('house_price_to_earning_ratio_binned', 'crime_count_binned'),
    ('unemployed_percent_binned', 'crime_count_binned'),
    ('qualification_index_score_binned', 'crime_count_binned'),
    ('ethnic_percent_in_ethnic_population_binned', 'crime_count_binned'),
    ('no_qualifications_binned', 'crime_count_binned')
]

# Initialize and fit the Bayesian Network
model = BayesianModel(model_structure)
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Initialize Variable Elimination
inference = VariableElimination(model)

# Perform inference
results = {}
variables = [
    'house_price_to_earning_ratio_binned',
    'economically_inactive_percent_binned',
    'unemployed_percent_binned',
    'qualification_index_score_binned',
    'ethnic_percent_in_ethnic_population_binned',
    'no_qualifications_binned'
]

# Perform the queries and store only the highest probability
for variable in variables:
    variable_influence = {}
    cpd = model.get_cpds(variable)
    bins = cpd.state_names[variable]

    for bin in bins:
        result = inference.query(variables=['crime_count_binned'], evidence={variable: bin})
        max_prob = np.max(result.values)
        variable_influence[bin] = max_prob
    results[variable] = variable_influence
```

### Summary

The Variable Elimination method systematically reduces the complexity of inference in Bayesian Networks by focusing on relevant variables and eliminating the rest through summation and factor multiplication. The Maximum Likelihood Estimator (MLE) provides a robust way to estimate the parameters of the model, ensuring that the observed data is most probable given the model parameters. These techniques are foundational in probabilistic graphical models, enabling effective and efficient inference and learning from data.