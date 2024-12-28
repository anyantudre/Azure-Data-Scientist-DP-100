# Perform Hyperparameter Tuning with Azure Machine Learning

## Introduction
In machine learning, models are trained to predict unknown labels for new data based on correlations between known labels and features found in the training data. Depending on the algorithm used, you may need to specify hyperparameters to configure how the model is trained.

For example, the logistic regression algorithm uses a regularization rate hyperparameter to counteract overfitting; and deep learning techniques for convolutional neural networks (CNNs) use hyperparameters like learning rate to control how weights are adjusted during training, and batch size to determine how many data items are included in each training batch.

 Note

Machine Learning is an academic field with its own particular terminology. Data scientists refer to the values determined from the training features as parameters, so a different term is required for values that are used to configure training behavior but which are not derived from the training data - hence the term hyperparameter.

The choice of hyperparameter values can significantly affect the resulting model, making it important to select the best possible values for your particular data and predictive performance goals.

Tuning hyperparameters
Diagram of different hyperparameter values resulting in different models by performing hyperparameter tuning.

Hyperparameter tuning is accomplished by training the multiple models, using the same algorithm and training data but different hyperparameter values. The resulting model from each training run is then evaluated to determine the performance metric for which you want to optimize (for example, accuracy), and the best-performing model is selected.

In Azure Machine Learning, you can tune hyperparameters by submitting a script as a sweep job. A sweep job will run a trial for each hyperparameter combination to be tested. Each trial uses a training script with parameterized hyperparameter values to train a model, and logs the target performance metric achieved by the trained model.

Learning objectives
In this module, you'll learn how to:

Define a hyperparameter search space.
Configure hyperparameter sampling.
Select an early-termination policy.
Run a sweep job.


## Define a search space
The set of hyperparameter values tried during hyperparameter tuning is known as the search space. The definition of the range of possible values that can be chosen depends on the type of hyperparameter.

### Discrete hyperparameters
Some hyperparameters require discrete values - in other words, you must select the value from a particular finite set of possibilities. You can define a search space for a discrete parameter using a Choice from a list of explicit values, which you can define as a Python list (Choice(values=[10,20,30])), a range (Choice(values=range(1,10))), or an arbitrary set of comma-separated values (Choice(values=(30,50,100)))

You can also select discrete values from any of the following discrete distributions:

QUniform(min_value, max_value, q): Returns a value like round(Uniform(min_value, max_value) / q) * q
QLogUniform(min_value, max_value, q): Returns a value like round(exp(Uniform(min_value, max_value)) / q) * q
QNormal(mu, sigma, q): Returns a value like round(Normal(mu, sigma) / q) * q
QLogNormal(mu, sigma, q): Returns a value like round(exp(Normal(mu, sigma)) / q) * q

### Continuous hyperparameters
Some hyperparameters are continuous - in other words you can use any value along a scale, resulting in an infinite number of possibilities. To define a search space for these kinds of value, you can use any of the following distribution types:

Uniform(min_value, max_value): Returns a value uniformly distributed between min_value and max_value
LogUniform(min_value, max_value): Returns a value drawn according to exp(Uniform(min_value, max_value)) so that the logarithm of the return value is uniformly distributed
Normal(mu, sigma): Returns a real value that's normally distributed with mean mu and standard deviation sigma
LogNormal(mu, sigma): Returns a value drawn according to exp(Normal(mu, sigma)) so that the logarithm of the return value is normally distributed

### Defining a search space
To define a search space for hyperparameter tuning, create a dictionary with the appropriate parameter expression for each named hyperparameter.

For example, the following search space indicates that the batch_size hyperparameter can have the value 16, 32, or 64, and the learning_rate hyperparameter can have any value from a normal distribution with a mean of 10 and a standard deviation of 3.

```python
from azure.ai.ml.sweep import Choice, Normal

command_job_for_sweep = job(
    batch_size=Choice(values=[16, 32, 64]),    
    learning_rate=Normal(mu=10, sigma=3),
)
```

## Configure a sampling method
The specific values used in a hyperparameter tuning run, or sweep job, depend on the type of sampling used.

There are three main sampling methods available in Azure Machine Learning:

Grid sampling: Tries every possible combination.
Random sampling: Randomly chooses values from the search space.
Sobol: Adds a seed to random sampling to make the results reproducible.
Bayesian sampling: Chooses new values based on previous results.

Note: Sobol is a variation of random sampling.


### Grid sampling
Grid sampling can only be applied when all hyperparameters are discrete, and is used to try every possible combination of parameters in the search space.

For example, in the following code example, grid sampling is used to try every possible combination of discrete batch_size and learning_rate value:

```python
from azure.ai.ml.sweep import Choice

command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),
    learning_rate=Choice(values=[0.01, 0.1, 1.0]),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = "grid",
    ...
)
```

### Random sampling
Random sampling is used to randomly select a value for each hyperparameter, which can be a mix of discrete and continuous values as shown in the following code example:

```python
from azure.ai.ml.sweep import Normal, Uniform

command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),   
    learning_rate=Normal(mu=10, sigma=3),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = "random",
    ...
)
```

#### Sobol
You may want to be able to reproduce a random sampling sweep job. If you expect that you do, you can use Sobol instead. Sobol is a type of random sampling that allows you to use a seed. When you add a seed, the sweep job can be reproduced, and the search space distribution is spread more evenly.

The following code example shows how to use Sobol by adding a seed and a rule, and using the RandomSamplingAlgorithm class:

```python
from azure.ai.ml.sweep import RandomSamplingAlgorithm

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = RandomSamplingAlgorithm(seed=123, rule="sobol"),
    ...
)
```

### Bayesian sampling
Bayesian sampling chooses hyperparameter values based on the Bayesian optimization algorithm, which tries to select parameter combinations that will result in improved performance from the previous selection. The following code example shows how to configure Bayesian sampling:

```python
from azure.ai.ml.sweep import Uniform, Choice

command_job_for_sweep = job(
    batch_size=Choice(values=[16, 32, 64]),    
    learning_rate=Uniform(min_value=0.05, max_value=0.1),
)

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = "bayesian",
    ...
)
```
You can only use Bayesian sampling with choice, uniform, and quniform parameter expressions.


## Configure early termination
Hyperparameter tuning helps you fine-tune your model and select the hyperparameter values that will make your model perform best.

For you to find the best model, however, can be a never-ending conquest. You always have to consider whether it's worth the time and expense of testing new hyperparameter values to find a model that may perform better.

Each trial in a sweep job, a new model is trained with a new combination of hyperparameter values. If training a new model doesn't result in a significantly better model, you may want to stop the sweep job and use the model that performed best so far.

When you configure a sweep job in Azure Machine Learning, you can also set a maximum number of trials. A more sophisticated approach may be to stop a sweep job when newer models don't produce significantly better results. To stop a sweep job based on the performance of the models, you can use an early termination policy.

### When to use an early termination policy
Whether you want to use an early termination policy may depend on the search space and sampling method you're working with.

For example, you may choose to use a grid sampling method over a discrete search space that results in a maximum of six trials. With six trials, a maximum of six models will be trained and an early termination policy may be unnecessary.

An early termination policy can be especially beneficial when working with continuous hyperparameters in your search space. Continuous hyperparameters present an unlimited number of possible values to choose from. You'll most likely want to use an early termination policy when working with continuous hyperparameters and a random or Bayesian sampling method.

### Configure an early termination policy
There are two main parameters when you choose to use an early termination policy:

evaluation_interval: Specifies at which interval you want the policy to be evaluated. Every time the primary metric is logged for a trial counts as an interval.
delay_evaluation: Specifies when to start evaluating the policy. This parameter allows for at least a minimum of trials to complete without an early termination policy affecting them.
New models may continue to perform only slightly better than previous models. To determine the extent to which a model should perform better than previous trials, there are three options for early termination:

Bandit policy: Uses a slack_factor (relative) or slack_amount(absolute). Any new model must perform within the slack range of the best performing model.
Median stopping policy: Uses the median of the averages of the primary metric. Any new model must perform better than the median.
Truncation selection policy: Uses a truncation_percentage, which is the percentage of lowest performing trials. Any new model must perform better than the lowest performing trials.


### Bandit policy
You can use a bandit policy to stop a trial if the target performance metric underperforms the best trial so far by a specified margin.

For example, the following code applies a bandit policy with a delay of five trials, evaluates the policy at every interval, and allows an absolute slack amount of 0.2.

```python
from azure.ai.ml.sweep import BanditPolicy

sweep_job.early_termination = BanditPolicy(
    slack_amount = 0.2, 
    delay_evaluation = 5, 
    evaluation_interval = 1
)
```

### Bandit policy
You can use a bandit policy to stop a trial if the target performance metric underperforms the best trial so far by a specified margin.

For example, the following code applies a bandit policy with a delay of five trials, evaluates the policy at every interval, and allows an absolute slack amount of 0.2.

```python
from azure.ai.ml.sweep import BanditPolicy

sweep_job.early_termination = BanditPolicy(
    slack_amount = 0.2, 
    delay_evaluation = 5, 
    evaluation_interval = 1
)
```
Imagine the primary metric is the accuracy of the model. When after the first five trials, the best performing model has an accuracy of 0.9, any new model needs to perform better than (0.9-0.2) or 0.7. If the new model's accuracy is higher than 0.7, the sweep job will continue. If the new model has an accuracy score lower than 0.7, the policy will terminate the sweep job.

You can also apply a bandit policy using a slack factor, which compares the performance metric as a ratio rather than an absolute value.

### Median stopping policy
A median stopping policy abandons trials where the target performance metric is worse than the median of the running averages for all trials.

For example, the following code applies a median stopping policy with a delay of five trials and evaluates the policy at every interval.

```python
from azure.ai.ml.sweep import MedianStoppingPolicy

sweep_job.early_termination = MedianStoppingPolicy(
    delay_evaluation = 5, 
    evaluation_interval = 1
)
```
Imagine the primary metric is the accuracy of the model. When the accuracy is logged for the sixth trial, the metric needs to be higher than the median of the accuracy scores so far. Suppose the median of the accuracy scores so far is 0.82. If the new model's accuracy is higher than 0.82, the sweep job will continue. If the new model has an accuracy score lower than 0.82, the policy will stop the sweep job, and no new models will be trained.

### Truncation selection policy
A truncation selection policy cancels the lowest performing X% of trials at each evaluation interval based on the truncation_percentage value you specify for X.

For example, the following code applies a truncation selection policy with a delay of four trials, evaluates the policy at every interval, and uses a truncation percentage of 20%.

```python
from azure.ai.ml.sweep import TruncationSelectionPolicy

sweep_job.early_termination = TruncationSelectionPolicy(
    evaluation_interval=1, 
    truncation_percentage=20, 
    delay_evaluation=4 
)
```
Imagine the primary metric is the accuracy of the model. When the accuracy is logged for the fifth trial, the metric should not be in the worst 20% of the trials so far. In this case, 20% translates to one trial. In other words, if the fifth trial is not the worst performing model so far, the sweep job will continue. If the fifth trial has the lowest accuracy score of all trials so far, the sweep job will stop.


## Use a sweep job for hyperparameter tuning
In Azure Machine Learning, you can tune hyperparameters by running a sweep job.

### Create a training script for hyperparameter tuning
To run a sweep job, you need to create a training script just the way you would do for any other training job, except that your script must:

Include an argument for each hyperparameter you want to vary.
Log the target performance metric with MLflow. A logged metric enables the sweep job to evaluate the performance of the trials it initiates, and identify the one that produces the best performing model.

For example, the following example script trains a logistic regression model using a --regularization argument to set the regularization rate hyperparameter, and logs the accuracy metric with the name Accuracy:

```python
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow

# get regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01)
args = parser.parse_args()
reg = args.reg_rate

# load the training dataset
data = pd.read_csv("data.csv")

# separate features and labels, and split for training/validatiom
X = data[['feature1','feature2','feature3','feature4']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# train a logistic regression model with the reg hyperparameter
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate and log accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
mlflow.log_metric("Accuracy", acc)
```

### Configure and run a sweep job
To prepare the sweep job, you must first create a base command job that specifies which script to run and defines the parameters used by the script:

```python
from azure.ai.ml import command

# configure command job as base
job = command(
    code="./src",
    command="python train.py --regularization ${{inputs.reg_rate}}",
    inputs={
        "reg_rate": 0.01,
    },
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    )
```
You can then override your input parameters with your search space:

```python
from azure.ai.ml.sweep import Choice

command_job_for_sweep = job(
    reg_rate=Choice(values=[0.01, 0.1, 1]),
)
```
Finally, call sweep() on your command job to sweep over your search space:

```python
from azure.ai.ml import MLClient

# apply the sweep parameter to obtain the sweep_job
sweep_job = command_job_for_sweep.sweep(
    compute="aml-cluster",
    sampling_algorithm="grid",
    primary_metric="Accuracy",
    goal="Maximize",
)

# set the name of the sweep job experiment
sweep_job.experiment_name="sweep-example"

# define the limits for this sweep
sweep_job.set_limits(max_total_trials=4, max_concurrent_trials=2, timeout=7200)

# submit the sweep
returned_sweep_job = ml_client.create_or_update(sweep_job)
```

### Monitor and review sweep jobs
You can monitor sweep jobs in Azure Machine Learning studio. The sweep job will initiate trials for each hyperparameter combination to be tried. For each trial, you can review all logged metrics.

Additionally, you can evaluate and compare models by visualizing the trials in the studio. You can adjust each chart to show and compare the hyperparameter values and metrics for each trial.

## Knowledge check

1. You plan to use hyperparameter tuning to find optimal discrete values for a set of hyperparameters. You want to try every possible combination of a set of specified discrete values. Which kind of sampling should you use? 
    - [ ] Random sampling
    - [x] Grid sampling. Correct. You should use grid sampling to try every combination of discrete hyperparameter values.
    - [ ] Bayesian sampling

2. You're using hyperparameter tuning to train an optimal model based on a target metric named "AUC". What should you do in your training script? 
    - [ ] Import the logging package and use a logging.info() statement to log the AUC.
    - [ ] Include a print() statement to write the AUC value to the standard output stream.
    - [x] Use a mlflow.log_metric() statement to log the AUC value. Correct. Your script needs to use MLflow to log the primary metric to the run using the same name as specified in the sweep job.