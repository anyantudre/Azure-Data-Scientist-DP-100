## Work with environments in Azure Machine Learning

### Introduction
To ensure consistent execution of code across different compute environments, Azure Machine Learning (Azure ML) uses environments to list and store necessary packages, libraries, and dependencies. Environments can be reused across compute targets, whether local or in the cloud.

### I. Understand Environments

Azure ML environments allow you to define and manage runtime configurations for experiments. These environments can be:

- **Curated Environments**: Automatically available in the Azure ML workspace.
- **Custom Environments**: User-created and registered for consistent, reusable runtime contexts.

#### What is an Environment in Azure Machine Learning?

An environment is a virtual environment defining the Python runtime version and installed packages. For portability, environments are often encapsulated in Docker containers hosted on compute targets like local machines, virtual machines, or cloud clusters.

To list all environments using the Python SDK:

```python
envs = ml_client.environments.list()
for env in envs:
    print(env.name)
```

To retrieve details of a specific environment:

```python
env = ml_client.environments.get("<environment_name>", version="<version>")
print(env.description, env.tags)
```

### II. Explore and Use Curated Environments

Curated environments are prebuilt for common machine learning tasks and use the prefix `AzureML-`. These environments include popular ML frameworks like Scikit-Learn.

#### Example: Retrieve a Curated Environment

```python
env = ml_client.environments.get("AzureML-sklearn-0.24-ubuntu18.04-py37-cpu", version=44)
print(env.description, env.tags)
```

#### Use a Curated Environment for a Command Job

```python
from azure.ai.ml import command

# Configure job
job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-with-curated-environment",
    experiment_name="train-with-curated-environment"
)

# Submit job
returned_job = ml_client.create_or_update(job)
```

#### Troubleshooting
If a job fails due to missing packages (e.g., `ModuleNotFoundError`), review the error logs and update the environment by adding the necessary packages.

### III. Create and Use Custom Environments

Custom environments allow you to define specific dependencies and configurations. They can be created from:

- **Docker images**
- **Docker build contexts**
- **Conda specification files**

#### Create an Environment from a Docker Image

```python
from azure.ai.ml.entities import Environment

env_docker_image = Environment(
    image="pytorch/pytorch:latest",
    name="public-docker-image-example",
    description="Environment created from a public Docker image."
)
ml_client.environments.create_or_update(env_docker_image)
```

#### Create an Environment with a Conda Specification File

Example Conda specification file (`conda-env.yml`):

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

Code to create an environment:

```python
from azure.ai.ml.entities import Environment

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./conda-env.yml",
    name="docker-image-plus-conda-example",
    description="Environment created from a Docker image plus Conda environment."
)
ml_client.environments.create_or_update(env_docker_conda)
```

#### Use a Custom Environment

```python
from azure.ai.ml import command

# Configure job
job = command(
    code="./src",
    command="python train.py",
    environment="docker-image-plus-conda-example:1",
    compute="aml-cluster",
    display_name="train-custom-env",
    experiment_name="train-custom-env"
)

# Submit job
returned_job = ml_client.create_or_update(job)
```

### Key Notes

- The first build of an environment can take 10-15 minutes.
- Once built, environments are reusable, and their images are stored in the Azure Container Registry.
- Reuse reduces build time for subsequent jobs.
