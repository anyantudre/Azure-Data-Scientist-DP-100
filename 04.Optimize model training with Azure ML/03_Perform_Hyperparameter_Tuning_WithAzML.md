# Perform Hyperparameter Tuning with Azure Machine Learning

## Run Pipelines in Azure Machine Learning

### Introduction
In Azure Machine Learning, you can:
- Experiment in notebooks.
- Train machine learning models by running scripts as jobs.

To manage the overall process, separate it into individual tasks and group them as pipelines. Pipelines are essential for implementing an effective Machine Learning Operations (MLOps) solution.

### Objectives
- Create components.
- Build an Azure Machine Learning pipeline.
- Run an Azure Machine Learning pipeline.

---

## Create Components

Components allow reusable scripts that can be shared across users within the same Azure Machine Learning workspace. They are useful for:
- Building pipelines.
- Sharing ready-to-use code.

### Steps to Create a Component

1. **Define a Component**
   - **Metadata:** Name, version, etc.
   - **Interface:** Input parameters (e.g., dataset or hyperparameters) and outputs (e.g., metrics or artifacts).
   - **Command, Code, and Environment:** Execution details.

2. **Files Needed:**
   - A script (e.g., `prep.py`) for workflow execution.
   - A YAML file (e.g., `prep.yml`) for metadata, interface, and execution configuration.

### Example: Prepare Data Component

**Python Script (prep.py):**
```python
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", dest='input_data', type=str)
parser.add_argument("--output_data", dest='output_data', type=str)
args = parser.parse_args()

df = pd.read_csv(args.input_data)
df = df.dropna()
scaler = MinMaxScaler()
df[['feature1', 'feature2', 'feature3', 'feature4']] = scaler.fit_transform(df[['feature1', 'feature2', 'feature3', 'feature4']])
df.to_csv(Path(args.output_data) / "prepped-data.csv", index=False)
```

**YAML File (prep.yml):**
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: prep_data
display_name: Prepare training data
version: 1
type: command
inputs:
  input_data:
    type: uri_file
outputs:
  output_data:
    type: uri_file
code: ./src
environment: azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest
command: >-
  python prep.py
  --input_data ${{inputs.input_data}}
  --output_data ${{outputs.output_data}}
```

**Load Component:**
```python
from azure.ai.ml import load_component

loaded_component_prep = load_component(source="./prep.yml")
```

### Register a Component
To share a component, register it:
```python
prep = ml_client.components.create_or_update(prepare_data_component)
```

---

## Create a Pipeline

A pipeline is a workflow of machine learning tasks (components), arranged sequentially or in parallel, to orchestrate operations.

### Build a Pipeline

**Example Code:**
```python
from azure.ai.ml.dsl import pipeline

@pipeline()
def pipeline_function_name(pipeline_job_input):
    prep_data = loaded_component_prep(input_data=pipeline_job_input)
    train_model = loaded_component_train(training_data=prep_data.outputs.output_data)

    return {
        "pipeline_job_transformed_data": prep_data.outputs.output_data,
        "pipeline_job_trained_model": train_model.outputs.model_output,
    }

pipeline_job = pipeline_function_name(
    Input(type=AssetTypes.URI_FILE, path="azureml:data:1")
)

print(pipeline_job)
```

### Example Output (YAML File):
```yaml
display_name: pipeline_function_name
type: pipeline
inputs:
  pipeline_job_input:
    type: uri_file
    path: azureml:data:1
outputs:
  pipeline_job_transformed_data: null
  pipeline_job_trained_model: null
jobs:
  prep_data:
    type: command
    inputs:
      input_data:
        path: ${{parent.inputs.pipeline_job_input}}
    outputs:
      output_data: ${{parent.outputs.pipeline_job_transformed_data}}
  train_model:
    type: command
    inputs:
      input_data:
        path: ${{parent.outputs.pipeline_job_transformed_data}}
    outputs:
      output_model: ${{parent.outputs.pipeline_job_trained_model}}
```

---

## Run a Pipeline Job

### Configure a Pipeline Job
Modify pipeline configurations as needed:
```python
pipeline_job.outputs.pipeline_job_transformed_data.mode = "upload"
pipeline_job.outputs.pipeline_job_trained_model.mode = "upload"
pipeline_job.settings.default_compute = "aml-cluster"
pipeline_job.settings.default_datastore = "workspaceblobstore"
```

### Submit a Pipeline Job
Run the workflow:
```python
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_job"
)
```

### Troubleshooting
Check logs and outputs of the pipeline and child jobs in Azure Machine Learning studio to identify issues.

---

## Schedule a Pipeline Job

### Time-Based Schedule
Use `JobSchedule` with `RecurrenceTrigger`:
```python
from azure.ai.ml.entities import RecurrenceTrigger, JobSchedule

schedule_name = "run_every_minute"
recurrence_trigger = RecurrenceTrigger(frequency="minute", interval=1)

job_schedule = JobSchedule(
    name=schedule_name, trigger=recurrence_trigger, create_job=pipeline_job
)
ml_client.schedules.begin_create_or_update(schedule=job_schedule).result()
```

### Delete a Schedule
Disable and delete:
```python
ml_client.schedules.begin_disable(name=schedule_name).result()
ml_client.schedules.begin_delete(name=schedule_name).result()
```

