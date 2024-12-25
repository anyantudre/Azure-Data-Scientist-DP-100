# Deploy a Model to a Managed Online Endpoint

Learn how to deploy models to a managed online endpoint for real-time inferencing.

## Learning Objectives
- Use managed online endpoints.
- Deploy your MLflow model to a managed online endpoint.
- Deploy a custom model to a managed online endpoint.
- Test online endpoints.

---

## I. Explore Managed Online Endpoints

### 1. Real-Time Predictions
Deploy a model to an HTTPS endpoint for instant predictions. The endpoint:
- Receives input data.
- Processes it using a scoring script.
- Returns predictions in real time.

### 2. Managed Online Endpoint
Two types of online endpoints:
- **Managed Online Endpoints**: Azure manages the infrastructure.
- **Kubernetes Online Endpoints**: Users manage Kubernetes infrastructure.

Managed endpoints handle:
- VM type and scaling settings.
- Automatic provisioning and OS updates.

### 3. Deploy Your Model
Requirements:
- **Model assets**: Files like pickle or a registered model.
- **Scoring script**: Loads the model.
- **Environment**: Lists dependencies.
- **Compute configuration**: VM size and scaling.

> **Note**: MLflow models automatically generate scoring scripts and environments.

### 4. Blue/Green Deployment
Deploy multiple versions of a model to the same endpoint. For example:
- **Blue deployment**: Original version.
- **Green deployment**: New version for testing.

Traffic can be split (e.g., 90% blue, 10% green). Easily switch traffic between versions to test or roll back.

### 5. Create an Endpoint
```python
from azure.ai.ml.entities import ManagedOnlineEndpoint

# Create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name="endpoint-example",
    description="Online endpoint",
    auth_mode="key",
)
ml_client.begin_create_or_update(endpoint).result()
```

---

## II. Deploy Your MLflow Model to a Managed Online Endpoint

### 1. Deploy an MLflow Model
Use MLflow models without custom scoring scripts or environments. Ensure model files (with `MLmodel` descriptor) are available locally or registered.

```python
from azure.ai.ml.entities import Model, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes

# Create a blue deployment
model = Model(
    path="./model",
    type=AssetTypes.MLFLOW_MODEL,
    description="my sample mlflow model",
)
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="endpoint-example",
    model=model,
    instance_type="Standard_F4s_v2",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(blue_deployment).result()
```

Route traffic:
```python
# Route 100% traffic to blue deployment
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()
```

Delete the endpoint:
```python
ml_client.online_endpoints.begin_delete(name="endpoint-example")
```

---

## III. Deploy a Custom Model to a Managed Online Endpoint

### 1. Deploy a Model
Requirements:
- **Model files**: Local or registered.
- **Scoring script**.
- **Execution environment**.

### 2. Create the Scoring Script
Example scoring script:
```python
import json
import joblib
import numpy as np
import os

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    predictions = model.predict(data)
    return predictions.tolist()
```

### 3. Create an Environment
Define dependencies in a `conda.yml` file:
```yaml
name: basic-env-cpu
channels:
  - conda-forge
dependencies:
  - python=3.7
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
```
Create the environment:
```python
from azure.ai.ml.entities import Environment

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./src/conda.yml",
    name="deployment-environment",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env)
```

### 4. Create the Deployment
Deploy with custom scoring script and environment:
```python
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration

model = Model(path="./model")

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="endpoint-example",
    model=model,
    environment="deployment-environment",
    code_configuration=CodeConfiguration(
        code="./src", scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(blue_deployment).result()
```

Route traffic:
```python
# Route 100% traffic to blue deployment
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()
```

Delete the endpoint:
```python
ml_client.online_endpoints.begin_delete(name="endpoint-example")
```

---

## IV. Test Managed Online Endpoints

### 1. Use Azure Machine Learning Studio
- Navigate to **Endpoints** in Azure Machine Learning Studio.
- Select an endpoint to review details or test it.

### 2. Use Azure Machine Learning Python SDK
Send JSON data for predictions:
```json
{
  "data": [
      [0.1,2.3,4.1,2.0],
      [0.2,1.8,3.9,2.1]
  ]
}
```
Invoke the endpoint:
```python
# Test the blue deployment with sample data
response = ml_client.online_endpoints.invoke(
    endpoint_name="endpoint-example",
    deployment_name="blue",
    request_file="sample-data.json",
)

if response[1] == '1':
    print("Yes")
else:
    print("No")
```

---

## Resources
- [Practice Steps](https://microsoftlearning.github.io/mslearn-azure-ml/Instructions/11-Deploy-online-endpoint.html)
- [YouTube: Managed Endpoints](https://www.youtube.com/watch?v=SxFGw_OBxNM)
