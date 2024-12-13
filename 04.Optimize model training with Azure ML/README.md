# Optimize model training with Azure Machine Learning

## Run a training script as a command job in Azure Machine Learning

### Introduction
A common challenge when developing machine learning models is to prepare for production scenarios. When you write code to process data and train models, you want the code to be scalable, repeatable, and ready for automation.

Though notebooks are ideal for experimentation and development, scripts are a better fit for production workloads. In Azure Machine Learning, you can run a script as a command job. When you submit a command job, you can configure various parameters like the input data and the compute environment. Azure Machine Learning also helps you track your work when working with command jobs to make it easier to compare workloads.

You'll learn how to run a script as a command job using the Python software development kit (SDK) v2 for Azure Machine Learning.

Learning objectives
In this module, you'll learn how to:

Convert a notebook to a script.
Test scripts in a terminal.
Run a script as a command job.
Use parameters in a command job.


### I. Convert a notebook to a script
When you've used notebooks for experimentation and development, you'll first need to convert a notebook to a script. Alternatively, you might choose to skip using notebooks and work only with scripts. Either way, there are some recommendations when creating scripts to have production-ready code.

Scripts are ideal for testing and automation in your production environment. To create a production-ready script, you'll need to:

Remove nonessential code.
Refactor your code into functions.
Test your script in the terminal.

##### Remove all nonessential code
The main benefit of using notebooks is being able to quickly explore your data. For example, you can use print() and df.describe() statements to explore your data and variables. When you create a script that will be used for automation, you want to avoid including code written for exploratory purposes.

The first thing you therefore need to do to convert your code to production code is to remove the nonessential code. Especially when you'll run the code regularly, you want to avoid executing anything nonessential to reduce cost and compute time.

##### Refactor your code into functions
When using code in business processes, you want the code to be easy to read so that anyone can maintain it. One common approach to make code easier to read and test is to use functions.

For example, you might have used the following example code in a notebook to read and split the data:

```python
# read and visualize the data
print("Reading data...")
df = pd.read_csv('diabetes.csv')
df.head()

# split data
print("Splitting data...")
X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
```

As functions also allow you to test parts of your code, you might prefer to create multiple smaller functions, rather than one large function. If you want to test a part of your code, you can choose to only test a small part and avoid running more code than necessary.

You can refactor the code shown in the example into two functions:

Read the data
Split the data
An example of refactored code might be the following:

```python
def main(csv_file):
    # read data
    df = get_data(csv_file)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)
    
    return df

# function that splits the data
def split_data(df):
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test
```

##### Test your script
Before using scripts in production environments, for example by integrating them with automation pipelines, you'll want to test whether the scripts work as expected.

One simple way to test your script, is to run the script in a terminal. Within the Azure Machine Learning workspace, you can quickly run a script in the terminal of the compute instance.

When you open a script in the Notebooks page of the Azure Machine Learning studio, you can choose to save and run the script in the terminal.

Alternatively, you can navigate directly to the terminal of the compute instance. Navigate to the Compute page and select the Terminal of the compute instance you want to use. You can use the following command to run a Python script named train.py:

```python
python train.py
```
Outputs of print statements will show in the terminal. Any possible errors will also appear in the terminal.



### II. Run a script as a command job
When you have a script that train a machine learning model, you can run it as a command job in Azure Machine Learning.

##### Configure and submit a command job
To run a script as a command job, you'll need to configure and submit the job.

To configure a command job with the Python SDK (v2), you'll use the command function. To run a script, you'll need to specify values for the following parameters:

code: The folder that includes the script to run.
command: Specifies which file to run.
environment: The necessary packages to be installed on the compute before running the command.
compute: The compute to use to run the command.
display_name: The name of the individual job.
experiment_name: The name of the experiment the job belongs to.

You can configure a command job to run a file named train.py, on the compute cluster named aml-cluster with the following code:

```python
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-model",
    experiment_name="train-classification-model"
    )
```

When your job is configured, you can submit it, which will initiate the job and run the script:

```python
# submit job
returned_job = ml_client.create_or_update(job)
```
You can monitor and review the job in the Azure Machine Learning studio. All jobs with the same experiment name will be grouped under the same experiment. You can find an individual job using the specified display name.

All inputs and outputs of a command job are tracked. You can review which command you specified, which compute was used, and which environment was used to run the script on the specified compute.


### III. Use parameters in a command job
You can increase the flexibility of your scripts by using parameters. For example, you might have created a script that trains a machine learning model. You can use the same script to train a model on different datasets, or using various hyperparameter values.

##### Working with script arguments
To use parameters in a script, you must use a library such as argparse to read arguments passed to the script and assign them to variables.

For example, the following script reads an arguments named training_data, which specifies the path to the training data.

```python
# import libraries
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main(args):
    # read data
    df = get_data(args.training_data)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)
    
    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":

    # parse args
    args = parse_args()

    # run main function
    main(args)
```
Any parameters you expect should be defined in the script. In the script, you can specify what type of value you expect for each parameter and whether you want to set a default value.

##### Passing arguments to a script
To pass parameter values to a script, you need to provide the argument value in the command.

For example, if you would pass a parameter value when running a script in a terminal, you would use the following command:

```python
python train.py --training_data diabetes.csv
```
In the example, diabetes.csv is a local file. Alternatively, you could specify the path to a data asset created in the Azure Machine Learning workspace.

Similarly, when you want to pass a parameter value to a script you want to run as a command job, you'll specify the values in the command:

```python
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python train.py --training_data diabetes.csv",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-model",
    experiment_name="train-classification-model"
    )
```

## Track model training with MLflow in jobs


### Introduction

Scripts are ideal when you want to run machine learning workloads in production environments. Imagine you're a data scientist who has developed a machine learning model to predict diabetes. The model is performing as expected and you created a training script. The script is used to retrain the model every month when new data has been collected.

You'll want to monitor the model's performance over time. You want to understand whether the new data every month benefits the model. Next to tracking models that are trained in notebooks, you can also use MLflow to track models in scripts.

MLflow is an open-source platform that helps you to track model metrics and artifacts across platforms and is integrated with Azure Machine Learning.

When you use MLflow together with Azure Machine Learning, you can run training scripts locally or in the cloud. You can review model metrics and artifacts in the Azure Machine Learning workspace to compare runs and decide on next steps.

Learning objectives
In this module, you learn how to:
Use MLflow when you run a script as a job.
Review metrics, parameters, artifacts, and models from a run.


### I. Track metrics with MLflow

When you train a model with a script, you can include MLflow in the scripts to track any parameters, metrics, and artifacts. When you run the script as a job in Azure Machine Learning, you're able to review all input parameters and outputs for each run.

##### Understand MLflow
MLflow is an open-source platform, designed to manage the complete machine learning lifecycle. As it's open source, it can be used when training models on different platforms. Here, we explore how we can integrate MLflow with Azure Machine Learning jobs.

There are two options to track machine learning jobs with MLflow:

Enable autologging using mlflow.autolog()
Use logging functions to track custom metrics using mlflow.log_*
Before you can use either of these options, you need to set up the environment to use MLflow.

##### Include MLflow in the environment
To use MLflow during training job, the mlflow and azureml-mlflow pip packages need to be installed on the compute executing the script. Therefore, you need to include these two packages in the environment. You can create an environment by referring to a YAML file that describes the Conda environment. As part of the Conda environment, you can include these two packages.

For example, in this custom environment mlflow and azureml-mlflow are installed using pip:

```python
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
Once the environment is defined and registered, make sure to refer to it when submitting a job.

##### Enable autologging
When working with one of the common libraries for machine learning, you can enable autologging in MLflow. Autologging logs parameters, metrics, and model artifacts without anyone needing to specify what needs to be logged.

Autologging is supported for the following libraries:

Scikit-learn
TensorFlow and Keras
XGBoost
LightGBM
Spark
Fastai
Pytorch
To enable autologging, add the following code to your training script:

```python
import mlflow

mlflow.autolog()
```

##### Log metrics with MLflow
In your training script, you can decide whatever custom metric you want to log with MLflow.

Depending on the type of value you want to log, use the MLflow command to store the metric with the experiment run:

mlflow.log_param(): Log single key-value parameter. Use this function for an input parameter you want to log.
mlflow.log_metric(): Log single key-value metric. Value must be a number. Use this function for any output you want to store with the run.
mlflow.log_artifact(): Log a file. Use this function for any plot you want to log, save as image file first.
To add MLflow to an existing training script, you can add the following code:

```python
import mlflow

reg_rate = 0.1
mlflow.log_param("Regularization rate", reg_rate)
```

##### Submit the job
Finally, you need to submit the training script as a job in Azure Machine Learning. When you use MLflow in a training script and run it as a job, all tracked parameters, metrics, and artifacts are stored with the job run.

You configure the job as usual. You only need to make sure that the environment you refer to in the job includes the necessary packages, and the script describes which metrics you want to log.


### II. View metrics and evaluate models
After you've trained and tracked models with MLflow in Azure Machine Learning, you can explore the metrics and evaluate your models.

Review metrics in the Azure Machine Learning studio.
Retrieve runs and metrics with MLflow.

##### View the metrics in the Azure Machine Learning studio
When your job is completed, you can review the logged parameters, metrics, and artifacts in the Azure Machine Learning studio.

When you review job runs in the Azure Machine Learning studio, you'll explore a job run's metrics, which is part of an experiment.

To view the metrics through an intuitive user interface, you can:

Open the Studio by navigating to https://ml.azure.com.
Find your experiment run and open it to view its details.
In the Details tab, all logged parameters are shown under Params.
Select the Metrics tab and select the metric you want to explore.
Any plots that are logged as artifacts can be found under Images.
The model assets that can be used to register and deploy the model are stored in the models folder under Outputs + logs.


##### Retrieve metrics with MLflow in a notebook
When you run a training script as a job in Azure Machine Learning, and track your model training with MLflow, you can query the runs in a notebook by using MLflow. Using MLflow in a notebook gives you more control over which runs you want to retrieve to compare.

When using MLflow to query your runs, you'll refer to experiments and runs.

###### Search all the experiments
You can get all the active experiments in the workspace using MLFlow:
```python
experiments = mlflow.search_experiments(max_results=2)
for exp in experiments:
    print(exp.name)
```
If you want to retrieve archived experiments too, then include the option ViewType.ALL:
```python
from mlflow.entities import ViewType

experiments = mlflow.search_experiments(view_type=ViewType.ALL)
for exp in experiments:
    print(exp.name)
```
To retrieve a specific experiment, you can run:
```python
exp = mlflow.get_experiment_by_name(experiment_name)
print(exp)
```

###### Retrieve runs
MLflow allows you to search for runs inside of any experiment. You need either the experiment ID or the experiment name.

For example, when you want to retrieve the metrics of a specific run:
```python
mlflow.search_runs(exp.experiment_id)
```

You can search runs across multiple experiments if necessary. Searching across experiments may be useful in case you want to compare runs of the same model when it's being logged in different experiments (by different people or different project iterations).

You can use search_all_experiments=True if you want to search across all the experiments in the workspace.

By default, experiments are ordered descending by start_time, which is the time the experiment was queued in Azure Machine Learning. However, you can change this default by using the parameter order_by.

For example, if you want to sort by start time and only show the last two results:
```python
mlflow.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=2)
```

You can also look for a run with a specific combination in the hyperparameters:

```python
mlflow.search_runs(
    exp.experiment_id, filter_string="params.num_boost_round='100'", max_results=2
)
```

## Perform hyperparameter tuning with Azure Machine Learning


## Run pipelines in Azure Machine Learning

### Introduction
In Azure Machine Learning, you can experiment in notebooks and train (and retrain) machine learning models by running scripts as jobs.

In an enterprise data science process, you'll want to separate the overall process into individual tasks. You can group tasks together as pipelines. Pipelines are key to implementing an effective Machine Learning Operations (MLOps) solution in Azure.

You'll learn how to create components of individual tasks, making it easier to reuse and share code. You'll then combine components into an Azure Machine Learning pipeline, which you'll run as a pipeline job.

Learning objectives
In this module, you'll learn how to:

Create components.
Build an Azure Machine Learning pipeline.
Run an Azure Machine Learning pipeline.


### II. Create components
Components allow you to create reusable scripts that can easily be shared across users within the same Azure Machine Learning workspace. You can also use components to build an Azure Machine Learning pipeline.


##### Use a component
There are two main reasons why you'd use components:
To build a pipeline.
To share ready-to-go code.

You'll want to create components when you're preparing your code for scale. When you're done with experimenting and developing, and ready to move your model to production.

Within Azure Machine Learning, you can create a component to store code (in your preferred language) within the workspace. Ideally, you design a component to perform a specific action that is relevant to your machine learning workflow.

For example, a component may consist of a Python script that normalizes your data, trains a machine learning model, or evaluates a model.

Components can be easily shared to other Azure Machine Learning users, who can reuse components in their own Azure Machine Learning pipelines.

##### Create a component
A component consists of three parts:

Metadata: Includes the component's name, version, etc.
Interface: Includes the expected input parameters (like a dataset or hyperparameter) and expected output (like metrics and artifacts).
Command, code and environment: Specifies how to run the code.
To create a component, you need two files:

A script that contains the workflow you want to execute.
A YAML file to define the metadata, interface, and command, code, and environment of the component.
You can create the YAML file, or use the command_component() function as a decorator to create the YAML file.

For example, you may have a Python script prep.py that prepares the data by removing missing values and normalizing the data:

```python
# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# setup arg parser
parser = argparse.ArgumentParser()

# add arguments
parser.add_argument("--input_data", dest='input_data',
                    type=str)
parser.add_argument("--output_data", dest='output_data',
                    type=str)

# parse args
args = parser.parse_args()

# read the data
df = pd.read_csv(args.input_data)

# remove missing values
df = df.dropna()

# normalize the data    
scaler = MinMaxScaler()
num_cols = ['feature1','feature2','feature3','feature4']
df[num_cols] = scaler.fit_transform(df[num_cols])

# save the data as a csv
output_df = df.to_csv(
    (Path(args.output_data) / "prepped-data.csv"), 
    index = False
)
```
To create a component for the prep.py script, you'll need a YAML file prep.yml:

```
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
Notice that the YAML file refers to the prep.py script, which is stored in the src folder. You can load the component with the following code:

```python
from azure.ai.ml import load_component
parent_dir = ""

loaded_component_prep = load_component(source=parent_dir + "./prep.yml")
```
When you've loaded the component, you can use it in a pipeline or register the component.

##### Register a component
To use components in a pipeline, you'll need the script and the YAML file. To make the components accessible to other users in the workspace, you can also register components to the Azure Machine Learning workspace.

You can register a component with the following code:

```python
prep = ml_client.components.create_or_update(prepare_data_component)
```


### II. Create a pipeline

In Azure Machine Learning, a pipeline is a workflow of machine learning tasks in which each task is defined as a component.

Components can be arranged sequentially or in parallel, enabling you to build sophisticated flow logic to orchestrate machine learning operations. Each component can be run on a specific compute target, making it possible to combine different types of processing as required to achieve an overall goal.

A pipeline can be executed as a process by running the pipeline as a pipeline job. Each component is executed as a child job as part of the overall pipeline job.


##### Build a pipeline
An Azure Machine Learning pipeline is defined in a YAML file. The YAML file includes the pipeline job name, inputs, outputs, and settings.

You can create the YAML file, or use the @pipeline() function to create the YAML file.

For example, if you want to build a pipeline that first prepares the data, and then trains the model, you can use the following code:

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
```
To pass a registered data asset as the pipeline job input, you can call the function you created with the data asset as input:

```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

pipeline_job = pipeline_function_name(
    Input(type=AssetTypes.URI_FILE, 
    path="azureml:data:1"
))
```

![]()

The result of running the @pipeline() function is a YAML file that you can review by printing the pipeline_job object you created when calling the function:
```python
print(pipeline_job)
```
The output will be formatted as a YAML file, which includes the configuration of the pipeline and its components. Some parameters included in the YAML file are shown in the following example.
```python
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
tags: {}
properties: {}
settings: {}
```

### III. Run a pipeline job

When you've built a component-based pipeline in Azure Machine Learning, you can run the workflow as a pipeline job.

##### Configure a pipeline job
A pipeline is defined in a YAML file, which you can also create using the @pipeline() function. After you've used the function, you can edit the pipeline configurations by specifying which parameters you want to change and the new value.

For example, you may want to change the output mode for the pipeline job outputs:

```python
# change the output mode
pipeline_job.outputs.pipeline_job_transformed_data.mode = "upload"
pipeline_job.outputs.pipeline_job_trained_model.mode = "upload"
```
Or, you may want to set the default pipeline compute. When a compute isn't specified for a component, it will use the default compute instead:

```python
# set pipeline level compute
pipeline_job.settings.default_compute = "aml-cluster"
```
You may also want to change the default datastore to where all outputs will be stored:
```python
# set pipeline level datastore
pipeline_job.settings.default_datastore = "workspaceblobstore"
```
To review your pipeline configuration, you can print the pipeline job object:
```python
print(pipeline_job)
```

##### Run a pipeline job
When you've configured the pipeline, you're ready to run the workflow as a pipeline job.

To submit the pipeline job, run the following code:
```python
# submit job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_job"
)
```
After you submit a pipeline job, a new job will be created in the Azure Machine Learning workspace. 
A pipeline job also contains child jobs, which represent the execution of the individual components. The Azure Machine Learning studio creates a graphical representation of your pipeline. 
You can expand the Job overview to explore the pipeline parameters, outputs, and child jobs:

![]()

To troubleshoot a failed pipeline, you can check the outputs and logs of the pipeline job and its child jobs.

If there's an issue with the configuration of the pipeline itself, you'll find more information in the outputs and logs of the pipeline job.
If there's an issue with the configuration of a component, you'll find more information in the outputs and logs of the child job of the failed component.

##### Schedule a pipeline job
A pipeline is ideal if you want to get your model ready for production. Pipelines are especially useful for automating the retraining of a machine learning model. To automate the retraining of a model, you can schedule a pipeline.

To schedule a pipeline job, you'll use the JobSchedule class to associate a schedule to a pipeline job.

There are various ways to create a schedule. A simple approach is to create a time-based schedule using the RecurrenceTrigger class with the following parameters:

frequency: Unit of time to describe how often the schedule fires. Value can be either minute, hour, day, week, or month.
interval: Number of frequency units to describe how often the schedule fires. Value needs to be an integer.
To create a schedule that fires every minute, run the following code:
```python
from azure.ai.ml.entities import RecurrenceTrigger

schedule_name = "run_every_minute"

recurrence_trigger = RecurrenceTrigger(
    frequency="minute",
    interval=1,
)
```
To schedule a pipeline, you'll need pipeline_job to represent the pipeline you've built:

```python
from azure.ai.ml.entities import JobSchedule

job_schedule = JobSchedule(
    name=schedule_name, trigger=recurrence_trigger, create_job=pipeline_job
)

job_schedule = ml_client.schedules.begin_create_or_update(
    schedule=job_schedule
).result()
```
The display names of the jobs triggered by the schedule will be prefixed with the name of your schedule. You can review the jobs in the Azure Machine Learning studio:

![]()

To delete a schedule, you first need to disable it:
```python
ml_client.schedules.begin_disable(name=schedule_name).result()
ml_client.schedules.begin_delete(name=schedule_name).result()
```








