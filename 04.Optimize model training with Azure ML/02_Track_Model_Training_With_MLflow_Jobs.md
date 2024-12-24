## Track Model Training with MLflow in Jobs

### Introduction

Scripts are useful for running machine learning workloads in production. For example, you might use a script to retrain a diabetes prediction model every month with new data.

**MLflow** helps monitor and track models by recording metrics, parameters, and artifacts. It integrates seamlessly with **Azure Machine Learning**, allowing you to:

- Run training scripts locally or in the cloud.
- Review metrics and artifacts in the Azure ML workspace.

#### Learning Objectives
- Use MLflow in scripts for tracking.
- Review metrics, parameters, artifacts, and models from a run.

---

### I. Track Metrics with MLflow

#### Understand MLflow
MLflow is an open-source platform for managing the machine learning lifecycle. You can track jobs with MLflow in two ways:

1. Enable **autologging** using `mlflow.autolog()`.
2. Use logging functions like `mlflow.log_*` to track custom metrics.

#### Include MLflow in the Environment
Install the `mlflow` and `azureml-mlflow` packages. Use a YAML file to define the environment:

```yaml
name: mlflow-env
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pip
  - pip:
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - mlflow
    - azureml-mlflow
```

#### Enable Autologging
MLflow can autolog metrics, parameters, and artifacts for supported libraries:

- **Scikit-learn**
- **TensorFlow/Keras**
- **XGBoost**
- **LightGBM**
- **PyTorch**

Add the following to your script:

```python
import mlflow

mlflow.autolog()
```

#### Log Custom Metrics
Add custom logging to your script:

```python
import mlflow

reg_rate = 0.1
mlflow.log_param("Regularization rate", reg_rate)
```

Use the following MLflow functions:
- `mlflow.log_param()` for parameters.
- `mlflow.log_metric()` for numerical metrics.
- `mlflow.log_artifact()` for files (e.g., images).

#### Submit the Job
Submit your script as an Azure ML job. Ensure your job configuration references the environment with MLflow installed.

---

### II. View Metrics and Evaluate Models

After training, use Azure ML Studio or MLflow to explore metrics and evaluate models.

#### View Metrics in Azure ML Studio
1. Navigate to [Azure ML Studio](https://ml.azure.com).
2. Locate and open the experiment run.
3. Explore:
   - **Params**: Logged parameters.
   - **Metrics**: Select metrics to view trends.
   - **Images**: Logged plots under artifacts.
   - **Models**: Registered models under Outputs + logs.

#### Retrieve Metrics with MLflow
Use MLflow in a notebook to query runs for comparison.

##### Search Experiments
Retrieve experiments:

```python
experiments = mlflow.search_experiments(max_results=2)
for exp in experiments:
    print(exp.name)
```

Include archived experiments:

```python
from mlflow.entities import ViewType

experiments = mlflow.search_experiments(view_type=ViewType.ALL)
for exp in experiments:
    print(exp.name)
```

Retrieve a specific experiment:

```python
exp = mlflow.get_experiment_by_name(experiment_name)
print(exp)
```

##### Retrieve Runs
Search for runs in an experiment:

```python
mlflow.search_runs(exp.experiment_id)
```

Sort results by start time:

```python
mlflow.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=2)
```

Filter runs by hyperparameters:

```python
mlflow.search_runs(
    exp.experiment_id, filter_string="params.num_boost_round='100'", max_results=2
)
```

---

By leveraging MLflow with Azure Machine Learning, you can efficiently track, compare, and evaluate machine learning models.

