# Find the Best Classification Model with Automated Machine Learning

## Introduction

Automated Machine Learning (AutoML) simplifies the process of finding the best-performing model by automating preprocessing transformations and algorithm selection.

> AutoML supports tasks like regression, forecasting, image classification, and natural language processing.

## Learning Objectives
- Prepare data for AutoML classification.
- Configure and run an AutoML experiment.
- Evaluate and compare models.

---

## I. Preprocess Data and Configure Featurization

### Create a Data Asset
To enable AutoML to read the data:
1. Create a data asset (MLTable) with a schema.
2. Specify it as input:

```python
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml:input-data-automl:1")
```

### Scaling and Normalization
AutoML applies scaling and normalization automatically to numeric data.

### Optional Featurization
Default featurization includes:
- Missing value imputation
- Categorical encoding
- Dropping high-cardinality features
- Feature engineering (e.g., date part extraction)

Customization is available, such as specifying imputation methods for specific features.

### Review Featurization Results
After training:
- Check scaling and normalization methods.
- Address issues like missing values or class imbalance.

---

## II. Run an Automated Machine Learning Experiment

### Supported Algorithms for Classification
- Logistic Regression
- Light GBM
- Decision Tree
- Random Forest
- Naive Bayes
- Linear SVM
- XGBoost

### Restrict Algorithm Selection
To block unsuitable algorithms:

```python
classification_job.set_training_properties(allowed_training_algorithms=["Logistic Regression", "XGBoost"])
```

### Configure an AutoML Experiment
Use the `automl.classification` function:

```python
from azure.ai.ml import automl

classification_job = automl.classification(
    compute="aml-cluster",
    experiment_name="auto-ml-class-dev",
    training_data=my_training_data_input,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True
)
```

### Specify the Primary Metric
Retrieve available metrics:

```python
from azure.ai.ml.automl import ClassificationPrimaryMetrics
 
list(ClassificationPrimaryMetrics)
```

### Set Experiment Limits
Control costs and runtime:

```python
classification_job.set_limits(
    timeout_minutes=60, 
    trial_timeout_minutes=20, 
    max_trials=5,
    enable_early_termination=True,
)
```

### Submit an Experiment
Submit and monitor the job:

```python
returned_job = ml_client.jobs.create_or_update(classification_job)

aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)
```

---

## III. Evaluate and Compare Models

### Review Models in Azure ML Studio
Explore details:
- Overview: Input data and summary of the best model.
- Models tab: List of trained models.

### Data Guardrails
Three states:
- **Passed**: No issues detected.
- **Done**: Data modified; review changes.
- **Alerted**: Issue detected; manual review required.

### Retrieve the Best Model
To identify and explore the best model:
- Models are sorted by the primary metric.
- Generate explanations for interpretability.

---
