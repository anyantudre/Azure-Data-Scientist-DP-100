## Register an MLflow Model in Azure Machine Learning

### Introduction
Deploying a machine learning model allows you to integrate it with an application. In Azure Machine Learning, you can deploy a model to batch or online endpoints by registering it with MLflow. This ensures practitioners have access to the latest model after every retraining.

### Learning Objectives
- Log models with MLflow.
- Understand the MLModel format.
- Register an MLflow model in Azure Machine Learning.

---

### I. Log Models with MLflow

MLflow is an open-source platform that simplifies machine learning deployment, irrespective of the model type or framework used.

#### Why Use MLflow?
- Standardized model packaging facilitates easy import/export across workflows.
- MLflow stores artifacts in a directory and creates an MLmodel file containing the model’s metadata, enhancing traceability.

#### Autologging
Enable autologging with `mlflow.autolog()` to automatically log parameters, metrics, artifacts, and models. Example:

```python
import mlflow.sklearn
mlflow.sklearn.autolog()
```

Common flavors for autologging:
- Keras: `mlflow.keras.autolog()`
- Scikit-learn: `mlflow.sklearn.autolog()`
- LightGBM: `mlflow.lightgbm.autolog()`
- XGBoost: `mlflow.xgboost.autolog()`
- TensorFlow: `mlflow.tensorflow.autolog()`
- PyTorch: `mlflow.pytorch.autolog()`
- ONNX: `mlflow.onnx.autolog()`

#### Manually Log a Model
For more control, disable autologging for models (`log_models=False`) and log manually. Customize the model’s signature:

```python
from mlflow.models.signature import infer_signature

# Infer signature
signature = infer_signature(training_data, model.predict(training_data))

# Log the model with the custom signature
mlflow.sklearn.log_model(model, "model_name", signature=signature)
```

Manually define a signature:

```python
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Define input and output schemas
input_schema = Schema([ColSpec("double", "feature1"), ColSpec("double", "feature2")])
output_schema = Schema([ColSpec("long")])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

---

### II. Understand the MLflow Model Format

MLflow stores model assets in a directory containing an `MLmodel` file. Key components of the file include:
- **artifact_path**: Path to logged artifacts.
- **flavor**: Framework used (e.g., Scikit-learn, TensorFlow).
- **model_uuid**: Unique identifier for the model.
- **signature**: Schema of inputs and outputs.

Example `MLmodel` file:

```yaml
artifact_path: model_path
flavors:
  sklearn:
    pickled_model: model.pkl
    sklearn_version: 1.0
  python_function:
    loader_module: mlflow.sklearn
    python_version: 3.8
signature:
  inputs: '[{"name": "age", "type": "long"}]'
  outputs: '[{"name": "target", "type": "long"}]'
```

#### Flavors
A flavor defines the framework used to create the model. Examples:
- **Scikit-learn Flavor**: Allows integration with Scikit-learn workflows.
- **Python Function Flavor**: Default for models, enabling deployment in diverse environments.

---

### III. Register an MLflow Model

Models trained in Azure Machine Learning jobs can be stored in the model registry, enabling:
- Versioning.
- Metadata tagging.
- Seamless integration for deployment.

#### Registering Methods
Use Azure ML Studio, Azure CLI, or the Python SDK. Example with Python SDK:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(DefaultAzureCredential(), "subscription_id", "resource_group", "workspace")

model = ml_client.models.create_or_update(
    name="model_name",
    version="1",
    path="./model_path"
)
```

Registered models can be deployed directly to endpoints, benefiting from Azure ML’s no-code deployment features.
