# Explore Developer Tools for Workspace Interaction
Azure Machine Learning provides data scientists with tools to build and manage machine learning models. You can interact with the Azure Machine Learning workspace using:

- **Azure Machine Learning Studio**
- **Python SDK**
- **Azure Command Line Interface (CLI)**

## Learning Objectives
- Use **Azure Machine Learning studio**.
- Use the **Python SDK**.
- Use the **Azure CLI**.

---

## I. Explore the Studio
The **Azure Machine Learning studio** is a web portal for managing resources and assets in the workspace.

### Access the Studio
1. **From Azure Portal**: Launch from the Overview page of the Azure Machine Learning workspace.
2. **Direct Sign-in**: Navigate to [https://ml.azure.com](https://ml.azure.com) and sign in with Azure credentials.

### Menu Overview
- **Author**: Create jobs for training models.
- **Assets**: Review assets used in model training.
- **Manage**: Create and manage resources for training.

Use the studio for quick experimentation, reviewing past jobs, and troubleshooting pipelines. For repetitive or automated tasks, prefer the **Azure CLI** or **Python SDK**.

---

## II. Explore the Python SDK
The **Python SDK** enables data scientists to interact with Azure Machine Learning in Python environments.

### Install the Python SDK
Ensure Python 3.7 or later is installed, then run:

```bash
pip install azure-ai-ml
```

### Connect to the Workspace
To authenticate and connect to the workspace, provide the following parameters:
- **subscription_id**: Your subscription ID.
- **resource_group**: Resource group name.
- **workspace_name**: Workspace name.

Example:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)
```

Use `MLClient` to create or update assets in the workspace. Example for submitting a training job:

```python
from azure.ai.ml import command

# Configure job
job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    experiment_name="train-model"
)

# Submit job
returned_job = ml_client.create_or_update(job)
```

### Reference Documentation
Find all classes, methods, and parameters in the [Python SDK reference documentation](https://learn.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-python).

---

## III. Explore the CLI
The **Azure CLI** is a code-based tool for automating tasks in Azure.

### Install the Azure CLI
Install on Linux, Mac, or Windows, or use the browser-based Azure Cloud Shell.

### Install the Azure Machine Learning Extension
Add the extension with:

```bash
az extension add -n ml -y
```

Verify installation:

```bash
az ml -h
```

### Work with the Azure CLI
Commands are prefixed with `az ml`. Example to create a compute target:

```bash
az ml compute create --name aml-cluster \
  --size STANDARD_DS3_v2 --min-instances 0 --max-instances 5 \
  --type AmlCompute --resource-group my-resource-group \
  --workspace-name my-workspace
```

### YAML Configuration
Define configuration in YAML for easier automation. Example:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/amlCompute.schema.json 
name: aml-cluster
type: amlcompute
size: STANDARD_DS3_v2
min_instances: 0
max_instances: 5
```

Create compute target using the YAML file:

```bash
az ml compute create --file compute.yml \
  --resource-group my-resource-group \
  --workspace-name my-workspace
```

Find YAML schemas and parameter details in the [reference documentation](https://learn.microsoft.com/en-us/azure/machine-learning/).
