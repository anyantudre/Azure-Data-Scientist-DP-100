# Deploy a Model to a Batch Endpoint

## Overview
Batch inferencing applies a predictive model to multiple cases asynchronously and writes results to a file or database. Azure Machine Learning enables this by deploying models to batch endpoints.

### Key Objectives:
- Create a batch endpoint.
- Deploy an MLflow model to a batch endpoint.
- Deploy a custom model to a batch endpoint.
- Invoke and troubleshoot batch endpoints.

---

## 1. Understand and Create Batch Endpoints

### Batch Predictions
A batch endpoint is an HTTPS endpoint used to trigger batch scoring jobs. It integrates with data pipelines like Azure Synapse Analytics or Azure Databricks. Scoring jobs use a compute cluster, and results are stored in a datastore.

#### Create a Batch Endpoint
```python
from azure.ai.ml.entities import BatchEndpoint

endpoint = BatchEndpoint(
    name="endpoint-example",
    description="A batch endpoint",
)

ml_client.batch_endpoints.begin_create_or_update(endpoint)
```

---

## 2. Deploy Your MLflow Model to a Batch Endpoint

### Register an MLflow Model
Register the model in the Azure Machine Learning workspace.
```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model = ml_client.models.create_or_update(
    Model(name="mlflow-model", path="./model", type=AssetTypes.MLFLOW_MODEL)
)
```

### Deploy an MLflow Model
Configure the deployment using the `BatchDeployment` class.
```python
from azure.ai.ml.entities import BatchDeployment, BatchRetrySettings
from azure.ai.ml.constants import BatchDeploymentOutputAction

deployment = BatchDeployment(
    name="forecast-mlflow",
    endpoint_name=endpoint.name,
    model=model,
    compute="aml-cluster",
    instance_count=2,
    max_concurrency_per_instance=2,
    mini_batch_size=2,
    output_action=BatchDeploymentOutputAction.APPEND_ROW,
    output_file_name="predictions.csv",
    retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
    logging_level="info",
)
ml_client.batch_deployments.begin_create_or_update(deployment)
```

---

## 3. Deploy a Custom Model to a Batch Endpoint

### Create the Scoring Script
Include `init()` and `run()` functions in the script.
```python
import os
import mlflow
import pandas as pd

def init():
    global model
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")
    model = mlflow.pyfunc.load(model_path)

def run(mini_batch):
    results = []
    for file_path in mini_batch:
        data = pd.read_csv(file_path)
        preds = model.predict(data)
        results.append(preds)
    return results
```

### Create an Environment
Define dependencies in a `conda.yaml` file.
```yaml
name: basic-env-cpu
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pandas
  - pip
  - pip:
      - azureml-core
      - mlflow
```

Create the environment in Azure ML:
```python
from azure.ai.ml.entities import Environment

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./src/conda-env.yml",
    name="deployment-environment",
)
ml_client.environments.create_or_update(env)
```

### Deploy the Model
```python
deployment = BatchDeployment(
    name="custom-model",
    endpoint_name=endpoint.name,
    model=model,
    compute="aml-cluster",
    code_path="./code",
    scoring_script="score.py",
    environment=env,
    instance_count=2,
    max_concurrency_per_instance=2,
    mini_batch_size=2,
    output_action=BatchDeploymentOutputAction.APPEND_ROW,
    output_file_name="predictions.csv",
    retry_settings=BatchRetrySettings(max_retries=3, timeout=300),
    logging_level="info",
)
ml_client.batch_deployments.begin_create_or_update(deployment)
```

---

## 4. Invoke and Troubleshoot Batch Endpoints

### Trigger the Batch Scoring Job
Register a folder as a data asset and invoke the endpoint:
```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

input = Input(type=AssetTypes.URI_FOLDER, path="azureml:new-data:1")
job = ml_client.batch_endpoints.invoke(endpoint_name=endpoint.name, input=input)
```

### Troubleshoot
Logs can be found under `Outputs + logs`:
- **job_error.txt**: Summarizes errors in the script.
- **job_progress_overview.txt**: Provides mini-batch processing progress.
- **job_result.txt**: Shows errors in `init()` or `run()` functions.

---

## Knowledge check
1. You are creating a batch endpoint that you want to use to predict new values for a large volume of data files. You want the pipeline to run the scoring script on multiple nodes and collate the results. What output action should you choose for the deployment? 
    - [ ] `summary_only`
    - [x] `append_row`. Correct. You should use append_row to append each prediction to one output file.
    - [ ] `concurrency`

2. You have multiple models deployed to a batch endpoint. You invoke the endpoint without indicating which model you want to use. Which deployed model will do the actual batch scoring? 
    - [ ] The latest version of the deployed model.
    - [ ] The latest deployed model.
    - [x] The default deployed model. Correct. The default deployment will be used to do the actual batch scoring when the endpoint is invoked.