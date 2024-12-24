## Track Model Training in Jupyter Notebooks with MLflow

### Introduction
Imagine you're a data scientist for a company developing an application for a cancer research lab. The application is to be used by researchers who upload an image of tissue to determine whether it's healthy. You're tasked with training a model to detect breast cancer using a large image database.

You'll use **Jupyter notebooks** to develop the model and **MLflow** to track training experiments in Azure Machine Learning.

### Learning Objectives
- **Configure MLflow** to use in notebooks.
- Use **MLflow** for model tracking in notebooks.

---

### I. Configure MLflow for Model Tracking in Notebooks

#### Configure MLflow in Notebooks

##### Using Azure Machine Learning Notebooks
When running a notebook on an Azure Machine Learning compute instance, MLflow is pre-configured. Verify the installation:

```python
pip show mlflow
pip show azureml-mlflow
```

##### Using MLflow on a Local Device

1. Install required packages:
   ```python
   pip install mlflow
   pip install azureml-mlflow
   ```
2. Navigate to Azure Machine Learning studio.
3. Copy the **MLflow tracking URI** from the Azure portal.
4. Configure MLflow in your local notebook:
   ```python
   mlflow.set_tracking_uri = "MLFLOW-TRACKING-URI"
   ```

---

### II. Train and Track Models in Notebooks

#### Create an MLflow Experiment
Group training results by creating an experiment:

```python
import mlflow
mlflow.set_experiment(experiment_name="heart-condition-classifier")
```

#### Log Results with MLflow

##### Enable Autologging
MLflow supports autologging for popular libraries. For example, with **XGBoost**:

```python
from xgboost import XGBClassifier

with mlflow.start_run():
    mlflow.xgboost.autolog()

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```

##### Use Custom Logging
Manually log additional or custom information using:

- `mlflow.log_param()` - Logs a key-value parameter.
- `mlflow.log_metric()` - Logs a key-value metric (numerical value).
- `mlflow.log_artifact()` - Logs a file (e.g., image or plot).
- `mlflow.log_model()` - Logs a model.

Example:

```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

with mlflow.start_run():
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
```

Custom logging provides flexibility but requires defining all parameters, metrics, or artifacts manually.

---

### Resources
- [Working with tables in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-mltable?view=azureml-api-2&tabs=cli)
