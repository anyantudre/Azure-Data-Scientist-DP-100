# Perform Hyperparameter Tuning with Azure Machine Learning

## Introduction
In machine learning, hyperparameters are values used to configure training behavior but are not derived from training data. For example:
- **Logistic regression** uses a regularization rate hyperparameter.
- **Deep learning** models use hyperparameters like learning rate and batch size.

The choice of hyperparameter values significantly impacts model performance.

### Hyperparameter Tuning
Hyperparameter tuning involves training multiple models with varying hyperparameters. The best-performing model is selected based on a performance metric (e.g., accuracy).

In Azure Machine Learning, hyperparameter tuning is done via **sweep jobs**, which train models using combinations of hyperparameter values and log the resulting performance metrics.

## Learning Objectives
Learn how to:
1. Define a hyperparameter search space.
2. Configure hyperparameter sampling.
3. Select an early-termination policy.
4. Run a sweep job.

## Define a Search Space
The search space is the range of hyperparameter values to test. 

### Discrete Hyperparameters
Discrete hyperparameters take values from a finite set. Define using:
- `Choice(values=[...])`
- `Choice(values=range(...))`
- Discrete distributions like `QUniform`, `QLogUniform`, `QNormal`, or `QLogNormal`.

### Continuous Hyperparameters
Continuous hyperparameters take any value within a range. Define using distributions like:
- `Uniform(min, max)`
- `LogUniform(min, max)`
- `Normal(mu, sigma)`
- `LogNormal(mu, sigma)`

### Example
```python
from azure.ai.ml.sweep import Choice, Normal

command_job_for_sweep = job(
    batch_size=Choice(values=[16, 32, 64]),    
    learning_rate=Normal(mu=10, sigma=3),
)
```

## Configure Sampling Methods
Sampling methods determine the hyperparameter values to test:
- **Grid Sampling**: Tries every combination (for discrete hyperparameters).
- **Random Sampling**: Selects random values (supports mixed types).
    - **Sobol Sampling**: A seeded random sampling variant for reproducibility.
- **Bayesian Sampling**: Chooses values based on prior results for optimization.

### Grid Sampling Example
```python
from azure.ai.ml.sweep import Choice

command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),
    learning_rate=Choice(values=[0.01, 0.1, 1.0]),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm="grid",
    ...
)
```

### Random Sampling Example
```python
from azure.ai.ml.sweep import Normal, Uniform

command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),   
    learning_rate=Normal(mu=10, sigma=3),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm="random",
    ...
)
```

### Sobol Sampling Example
```python
from azure.ai.ml.sweep import RandomSamplingAlgorithm

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm=RandomSamplingAlgorithm(seed=123, rule="sobol"),
    ...
)
```

### Bayesian Sampling Example
```python
from azure.ai.ml.sweep import Uniform, Choice

command_job_for_sweep = job(
    batch_size=Choice(values=[16, 32, 64]),    
    learning_rate=Uniform(min_value=0.05, max_value=0.1),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm="bayesian",
    ...
)
```

## Configure Early Termination
Early termination stops trials that do not show significant improvement, saving time and resources.

### Parameters
- **evaluation_interval**: Frequency of policy evaluation.
- **delay_evaluation**: Minimum number of trials to run before applying policy.

### Termination Policies
1. **Bandit Policy**: Stops trials underperforming the best by a margin.
    ```python
    from azure.ai.ml.sweep import BanditPolicy

    sweep_job.early_termination = BanditPolicy(
        slack_amount=0.2, 
        delay_evaluation=5, 
        evaluation_interval=1
    )
    ```
2. **Median Stopping Policy**: Stops trials below the median of averages.
    ```python
    from azure.ai.ml.sweep import MedianStoppingPolicy

    sweep_job.early_termination = MedianStoppingPolicy(
        delay_evaluation=5, 
        evaluation_interval=1
    )
    ```
3. **Truncation Selection Policy**: Stops a percentage of worst-performing trials.
    ```python
    from azure.ai.ml.sweep import TruncationSelectionPolicy

    sweep_job.early_termination = TruncationSelectionPolicy(
        evaluation_interval=1, 
        truncation_percentage=20, 
        delay_evaluation=4
    )
    ```

## Use a Sweep Job
### Create a Training Script
Include arguments for each hyperparameter and log the performance metric using **MLflow**.
