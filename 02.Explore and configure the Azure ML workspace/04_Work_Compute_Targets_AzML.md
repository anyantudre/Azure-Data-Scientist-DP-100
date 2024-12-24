## Work with Compute Targets in Azure Machine Learning

In Azure Machine Learning, you can use various types of managed cloud computes to scale your work efficiently. This guide summarizes the types of compute targets and their usage for experimentation, training, and deployment.

---

### I. Choose the Appropriate Compute Target

Compute targets are physical or virtual machines on which jobs are run. Azure Machine Learning provides multiple types of compute:

- **Compute instance**: Virtual machine for running notebooks; ideal for experimentation.
- **Compute clusters**: Multi-node clusters that scale up or down; cost-effective for processing large datasets and parallel processing.
- **Kubernetes clusters**: Self-managed clusters based on Kubernetes technology for customizable compute.
- **Attached compute**: Attach existing compute resources like Azure VMs or Azure Databricks clusters.
- **Serverless compute**: Fully managed, on-demand compute for training jobs.

#### When to Use Which Compute?

- **Experimentation**: Use **compute instance** for Jupyter notebooks or **Spark serverless compute** for distributed Spark code.
- **Production**: Use **compute clusters** for scalable, automated training scripts or **serverless compute** for on-demand tasks.
- **Deployment**: Use **compute clusters** or **serverless compute** for batch predictions and **Kubernetes clusters** or **containers** for real-time predictions.

---

### II. Create and Use a Compute Instance

A compute instance is a managed virtual machine for running notebooks.

#### Create a Compute Instance with the Python SDK
```python
from azure.ai.ml.entities import ComputeInstance

ci_basic_name = "basic-ci-12345"
ci_basic = ComputeInstance(
    name=ci_basic_name, 
    size="STANDARD_DS3_v2"
)
ml_client.begin_create_or_update(ci_basic).result()
```

#### Minimize Compute Time
- Start/stop the compute instance manually or set schedules to save costs.
- Configure auto-shutdown after idle time.

#### Use a Compute Instance
- Work directly in Azure Machine Learning studio or attach the instance to Visual Studio Code for source control.

---

### III. Create and Use a Compute Cluster

Compute clusters are scalable and ideal for production workloads.

#### Create a Compute Cluster with the Python SDK
```python
from azure.ai.ml.entities import AmlCompute

cluster_basic = AmlCompute(
    name="cpu-cluster",
    type="amlcompute",
    size="STANDARD_DS3_v2",
    location="westus",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=120,
    tier="low_priority",
)
ml_client.begin_create_or_update(cluster_basic).result()
```

Key Parameters:
- **size**: Virtual machine type (CPU/GPU).
- **max_instances**: Maximum number of nodes for scaling.
- **tier**: Choose between low priority (cost-efficient) or dedicated.

#### Use a Compute Cluster
- **Scenarios**: Run pipeline jobs, Automated Machine Learning, or scripts.
- **Example: Run a Script as a Job**
```python
from azure.ai.ml import command

job = command(
    code="./src",
    command="python diabetes-training.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="cpu-cluster",
    display_name="train-with-cluster",
    experiment_name="diabetes-training"
)

returned_job = ml_client.create_or_update(job)
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)
```
After submission, the cluster scales out to handle the workload, automatically shutting down when the job completes.

---

This concise summary will help you quickly review the key points for working with compute targets in Azure Machine Learning. For more details, refer to the [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/).
