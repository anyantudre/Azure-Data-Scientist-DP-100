# Deploy and consume models with Azure Machine Learning

Learn how to deploy a model to an endpoint. When you deploy a model, you can get real-time or batch predictions by calling the endpoint.

## Deploy a model to a managed online endpoint
Learn how to deploy models to a managed online endpoint for real-time inferencing.
In this module, you'll learn how to:
- Use managed online endpoints.
- Deploy your MLflow model to a managed online endpoint.
- Deploy a custom model to a managed online endpoint.
- Test online endpoints.

### Introduction
Whenever you train a model, you ultimately will want to consume the model. You want to use the trained model to predict labels for new data on which the model hasn't been trained.  
To consume the model, you need to deploy it. One way to deploy a model is to integrate it with a service that allows applications to request instant, or real-time, predictions for individual or small sets of data points.


In Azure Machine Learning, you can use online endpoints to deploy and consume your model.

### I. Explore managed online endpoints

##### 1. Real-time predictions
To get real-time predictions, you can deploy a model to an endpoint. An endpoint is an HTTPS endpoint to which you can send data, and which will return a response (almost) immediately.

Any data you send to the endpoint will serve as the input for the scoring script hosted on the endpoint. The scoring script loads the trained model to predict the label for the new input data, which is also called inferencing. The label is then part of the output that's returned.

##### 2. Managed online endpoint
Within Azure Machine Learning, there are two types of online endpoints:
- Managed online endpoints: Azure Machine Learning manages all the underlying infrastructure.
- Kubernetes online endpoints: Users manage the Kubernetes cluster which provides the necessary infrastructure.
As a data scientist, you may prefer to work with managed online endpoints to test whether your model works as expected when deployed. If a Kubernetes online endpoint is required for control and scalability, it'll likely be managed by other teams.  
If you're using a managed online endpoint, you only need to specify the virtual machine (VM) type and scaling settings. Everything else, such as provisioning compute power and updating the host operating system (OS) is done for you automatically.

##### 3. Deploy your model
After you create an endpoint in the Azure Machine Learning workspace, you can deploy a model to that endpoint. To deploy your model to a managed online endpoint, you need to specify four things:
- Model assets like the model pickle file, or a registered model in the Azure Machine Learning workspace.
- Scoring script that loads the model.
- Environment which lists all necessary packages that need to be installed on the compute of the endpoint.
- Compute configuration including the needed compute size and scale settings to ensure you can handle the amount of requests the endpoint will receive.

Note: When you deploy MLFlow models to an online endpoint, you don't need to provide a scoring script and environment, as both are automatically generated.

##### 4. Blue/green deployment
One endpoint can have multiple deployments. One approach is the blue/green deployment.

Let's take the example of the restaurant recommender model. After experimentation, you select the best performing model. You use the blue deployment for this first version of the model. As new data is collected, the model can be retrained, and a new version is registered in the Azure Machine Learning workspace. To test the new model, you can use the green deployment for the second version of the model.

Both versions of the model are deployed to the same endpoint, which is integrated with the application. Within the application, a user selects a restaurant, sending a request to the endpoint to get new real-time recommendations of other restaurants the user may like.

When a request is sent to the endpoint, 90% of the traffic can go to the blue deployment*, and 10% of the traffic can go to the green deployment. With two versions of the model deployed on the same endpoint, you can easily test the model.

After testing, you can also seamlessly transition to the new version of the model by redirecting 90% of the traffic to the green deployment. If it turns out that the new version doesn't perform better, you can easily roll back to the first version of the model by redirecting most of the traffic back to the blue deployment.

Blue/green deployment allows for multiple models to be deployed to an endpoint. You can decide how much traffic to forward to each deployed model. This way, you can switch to a new version of the model without interrupting service to the consumer.

##### 5. Create an endpoint
To create an online endpoint, you'll use the ManagedOnlineEndpoint class, which requires the following parameters:
- name: Name of the endpoint. Must be unique in the Azure region.
- auth_mode: Use key for key-based authentication. Use aml_token for Azure Machine Learning token-based authentication.
To create an endpoint, use the following command:

```python
from azure.ai.ml.entities import ManagedOnlineEndpoint

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name="endpoint-example",
    description="Online endpoint",
    auth_mode="key",
)

ml_client.begin_create_or_update(endpoint).result()
```

### II. Deploy your MLflow model to a managed online endpoint
The easiest way to deploy a model to an online endpoint is to use an MLflow model and deploy it to a managed online endpoint. Azure Machine Learning will automatically generate the scoring script and environment for MLflow models.  
To deploy an MLflow model, you need to have created an endpoint. Then you can deploy the model to the endpoint.

##### 1. Deploy an MLflow model to an endpoint
When you deploy an MLflow model to a managed online endpoint, you don´t need to have the scoring script and environment.

To deploy an MLflow model, you must have model files stored on a local path or with a registered model. You can log model files when training a model by using MLflow tracking.

In this example, we're taking the model files from a local path. The files are all stored in a local folder called model. The folder must include the MLmodel file, which describes how the model can be loaded and used.

Next to the model, you also need to specify the compute configuration for the deployment:
- instance_type: Virtual machine (VM) size to use. 
- instance_count: Number of instances to use.

To deploy (and automatically register) the model, run the following command:
```python
from azure.ai.ml.entities import Model, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes

# create a blue deployment
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

Since only one model is deployed to the endpoint, you want this model to take 100% of the traffic. When you deploy multiple models to the same endpoint, you can distribute the traffic among the deployed models.

To route traffic to a specific deployment, use the following code:
```python
# blue deployment takes 100 traffic
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()
```

To delete the endpoint and all associated deployments, run the command:
```python
ml_client.online_endpoints.begin_delete(name="endpoint-example")
```


### Deploy a model to a managed online endpoint
You can choose to deploy a model to a managed online endpoint without using the MLflow model format. To deploy a model, you'll need to create the scoring script and define the environment necessary during inferencing.

To deploy a model, you need to have created an endpoint. Then you can deploy the model to the endpoint.

##### 1. Deploy a model to an endpoint
To deploy a model, you must have:
- Model files stored on local path or registered model.
- A scoring script.
- An execution environment.
The model files can be logged and stored when you train a model.

##### 2. Create the scoring script

The scoring script needs to include two functions:

init(): Called when the service is initialized.
run(): Called when new data is submitted to the service.
The init function is called when the deployment is created or updated, to load and cache the model from the model registry. The run function is called for every time the endpoint is invoked, to generate predictions from the input data. The following example Python script shows this pattern:

```python
import json
import joblib
import numpy as np
import os

# called when the deployment is created or updated
def init():
    global model
    # get the path to the registered model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

# called when a request is received
def run(raw_data):
    # get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # get a prediction from the model
    predictions = model.predict(data)
    # return the predictions as any JSON serializable format
    return predictions.tolist()
```

##### 3. Create an environment
Your deployment requires an execution environment in which to run the scoring script.  
You can create an environment with a Docker image with Conda dependencies, or with a Dockerfile.  
To create an environment using a base Docker image, you can define the Conda dependencies in a conda.yml file:

```yml
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
Then, to create the environment, run the following code:

```yml
from azure.ai.ml.entities import Environment

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./src/conda.yml",
    name="deployment-environment",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env)
```

##### 4. Create the deployment
When you have your model files, scoring script, and environment, you can create the deployment.

To deploy a model to an endpoint, you can specify the compute configuration with two parameters:
- instance_type: Virtual machine (VM) size to use. 
- instance_count: Number of instances to use.
To deploy the model, use the ManagedOnlineDeployment class and run the following command:
```python
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration

model = Model(path="./model",

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

You can deploy multiple models to an endpoint. To route traffic to a specific deployment, use the following code:
```python
# blue deployment takes 100 traffic
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()
```

To delete the endpoint and all associated deployments, run the command:
```python
ml_client.online_endpoints.begin_delete(name="endpoint-example")
```


### IV. Test managed online endpoints
After deploying a real-time service, you can consume it from client applications to predict labels for new data cases.

##### 1. Use the Azure Machine Learning studio
You can list all endpoints in the Azure Machine Learning studio, by navigating to the Endpoints page. In the Real-time endpoints tab, all endpoints are shown.

You can select an endpoint to review its details and deployment logs.

Additionally, you can use the studio to test the endpoint.

![](test-studio.png)

##### 2. Use the Azure Machine Learning Python SDK
For testing, you can also use the Azure Machine Learning Python SDK to invoke an endpoint.

Typically, you send data to deployed model in JSON format with the following structure:
```json
{
  "data":[
      [0.1,2.3,4.1,2.0], // 1st case
      [0.2,1.8,3.9,2.1],  // 2nd case,
      ...
  ]
}
```
The response from the deployed model is a JSON collection with a prediction for each case that was submitted in the data. The following code sample invokes an endpoint and displays the response:
```python
# test the blue deployment with some sample data
response = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="blue",
    request_file="sample-data.json",
)

if response[1]=='1':
    print("Yes")
else:
    print ("No")
```

##### Ressources

- [Steps for practice](https://microsoftlearning.github.io/mslearn-azure-ml/Instructions/11-Deploy-online-endpoint.html)
- [YouTube - Managed Endpoints](https://www.youtube.com/watch?v=SxFGw_OBxNM)
- 



## Deploy a model to a batch endpoint



### Introduction
Imagine you trained a model to predict the product sales. The model has been trained and tracked in Azure Machine Learning. Every month, you want to use the model to forecast the sales for the upcoming month.

In many production scenarios, long-running tasks that deal with large amounts of data are performed as batch operations. In machine learning, batch inferencing is used to asynchronously apply a predictive model to multiple cases and write the results to a file or database.

![]()

In Azure Machine Learning, you can implement batch inferencing solutions by deploying a model to a batch endpoint.
In this module, you'll learn how to:

Create a batch endpoint.
Deploy your MLflow model to a batch endpoint.
Deploy a custom model to a batch endpoint.
Invoke batch endpoints.


### I. Understand and create batch endpoints
To get a model to generate batch predictions, you can deploy the model to a batch endpoint.

You'll learn how to use batch endpoints for asynchronous batch scoring.


### II. Understand and create batch endpoints

##### Batch predictions
To get batch predictions, you can deploy a model to an endpoint. An endpoint is an HTTPS endpoint that you can call to trigger a batch scoring job. The advantage of such an endpoint is that you can trigger the batch scoring job from another service, such as Azure Synapse Analytics or Azure Databricks. A batch endpoint allows you to integrate the batch scoring with an existing data ingestion and transformation pipeline.

Whenever the endpoint is invoked, a batch scoring job is submitted to the Azure Machine Learning workspace. The job typically uses a compute cluster to score multiple inputs. The results can be stored in a datastore, connected to the Azure Machine Learning workspace.

##### Create a batch endpoint
To deploy a model to a batch endpoint, you'll first have to create the batch endpoint.

To create a batch endpoint, you'll use the BatchEndpoint class. Batch endpoint names need to be unique within an Azure region.

To create an endpoint, use the following command:
```python
# create a batch endpoint
endpoint = BatchEndpoint(
    name="endpoint-example",
    description="A batch endpoint",
)

ml_client.batch_endpoints.begin_create_or_update(endpoint)
```

##### Deploy a model to a batch endpoint
You can deploy multiple models to a batch endpoint. Whenever you call the batch endpoint, which triggers a batch scoring job, the default deployment will be used unless specified otherwise.


##### Use compute clusters for batch deployments
The ideal compute to use for batch deployments is the Azure Machine Learning compute cluster. If you want the batch scoring job to process the new data in parallel batches, you need to provision a compute cluster with more than one maximum instances.

To create a compute cluster, you can use the AMLCompute class.
```python
from azure.ai.ml.entities import AmlCompute

cpu_cluster = AmlCompute(
    name="aml-cluster",
    type="amlcompute",
    size="STANDARD_DS11_V2",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=120,
    tier="Dedicated",
)

cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)
```

### II. Deploy your MLflow model to a batch endpoint
An easy way to deploy a model to a batch endpoint is to use an MLflow model. Azure Machine Learning will automatically generate the scoring script and environment for MLflow models.

##### Register an MLflow model
To avoid needed a scoring script and environment, an MLflow model needs to be registered in the Azure Machine Learning workspace before you can deploy it to a batch endpoint.

To register an MLflow model, you'll use the Model class, while specifying the model type to be MLFLOW_MODEL. To register the model with the Python SDK, you can use the following code:
```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model_name = 'mlflow-model'
model = ml_client.models.create_or_update(
    Model(name=model_name, path='./model', type=AssetTypes.MLFLOW_MODEL)
)
```


##### Deploy an MLflow model to an endpoint

To deploy an MLflow model to a batch endpoint, you'll use the BatchDeployment class.

When you deploy a model, you'll need to specify how you want the batch scoring job to behave. The advantage of using a compute cluster to run the scoring script (which is automatically generated by Azure Machine Learning), is that you can run the scoring script on separate instances in parallel.

When you configure the model deployment, you can specify:

instance_count: Count of compute nodes to use for generating predictions.
max_concurrency_per_instance: Maximum number of parallel scoring script runs per compute node.
mini_batch_size: Number of files passed per scoring script run.
output_action: What to do with the predictions: summary_only or append_row.
output_file_name: File to which predictions will be appended, if you choose append_row for output_action.

```python
from azure.ai.ml.entities import BatchDeployment, BatchRetrySettings
from azure.ai.ml.constants import BatchDeploymentOutputAction

deployment = BatchDeployment(
    name="forecast-mlflow",
    description="A sales forecaster",
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


### III. Deploy a custom model to a batch endpoint

If you want to deploy a model to a batch endpoint without using the MLflow model format, you need to create the scoring script and environment.

To deploy a model, you must have already created an endpoint. Then you can deploy the model to the endpoint.

##### Create the scoring script
The scoring script is a file that reads the new data, loads the model, and performs the scoring.

The scoring script must include two functions:

init(): Called once at the beginning of the process, so use for any costly or common preparation like loading the model.
run(): Called for each mini batch to perform the scoring.
The run() method should return a pandas DataFrame or an array/list.

A scoring script may look as follows:

```python
import os
import mlflow
import pandas as pd


def init():
    global model

    # get the path to the registered model file and load it
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")
    model = mlflow.pyfunc.load(model_path)


def run(mini_batch):
    print(f"run method start: {__file__}, run({len(mini_batch)} files)")
    resultList = []

    for file_path in mini_batch:
        data = pd.read_csv(file_path)
        pred = model.predict(data)

        df = pd.DataFrame(pred, columns=["predictions"])
        df["file"] = os.path.basename(file_path)
        resultList.extend(df.values)

    return resultList
```

There are some things to note from the example script:

AZUREML_MODEL_DIR is an environment variable that you can use to locate the files associated with the model.
Use global variable to make any assets available that are needed to score the new data, like the loaded model.
The size of the mini_batch is defined in the deployment configuration. If the files in the mini batch are too large to be processed, you need to split the files into smaller files.
By default, the predictions will be written to one single file.


##### Create an environment
Your deployment requires an execution environment in which to run the scoring script. Any dependency your code requires should be included in the environment.

You can create an environment with a Docker image with Conda dependencies, or with a Dockerfile.

You'll also need to add the library azureml-core as it is required for batch deployments to work.

To create an environment using a base Docker image, you can define the Conda dependencies in a conda.yaml file:

```
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


```python
from azure.ai.ml.entities import Environment

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./src/conda-env.yml",
    name="deployment-environment",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env)
```

##### Configure and create the deployment
Finally, you can configure and create the deployment with the BatchDeployment class.
```python
from azure.ai.ml.entities import BatchDeployment, BatchRetrySettings
from azure.ai.ml.constants import BatchDeploymentOutputAction

deployment = BatchDeployment(
    name="forecast-mlflow",
    description="A sales forecaster",
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

### IV. Invoke and troubleshoot batch endpoints

When you invoke a batch endpoint, you trigger an Azure Machine Learning pipeline job. The job will expect an input parameter pointing to the data set you want to score.


##### Trigger the batch scoring job

To prepare data for batch predictions, you can register a folder as a data asset in the Azure Machine Learning workspace.

You can then use the registered data asset as input when invoking the batch endpoint with the Python SDK:
```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

input = Input(type=AssetTypes.URI_FOLDER, path="azureml:new-data:1")

job = ml_client.batch_endpoints.invoke(
    endpoint_name=endpoint.name, 
    input=input)
```
You can monitor the run of the pipeline job in the Azure Machine Learning studio. All jobs that are triggered by invoking the batch endpoint will show in the Jobs tab of the batch endpoint.


The predictions will be stored in the default datastore.


##### Troubleshoot a batch scoring job
The batch scoring job runs as a pipeline job. If you want to troubleshoot the pipeline job, you can review its details and the outputs and logs of the pipeline job itself.

If you want to troubleshoot the scoring script, you can select the child job and review its outputs and logs.

Navigate to the Outputs + logs tab. The logs/user/ folder contains three files that will help you troubleshoot:

job_error.txt: Summarize the errors in your script.
job_progress_overview.txt: Provides high-level information about the number of mini-batches processed so far.
job_result.txt: Shows errors in calling the init() and run() function in the scoring script.





