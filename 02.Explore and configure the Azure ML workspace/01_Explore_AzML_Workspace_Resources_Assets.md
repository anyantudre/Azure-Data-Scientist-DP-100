# Explore Azure Machine Learning Workspace Resources and Assets

Azure Machine Learning provides a platform to train, deploy, and manage machine learning models on the Microsoft Azure platform. Below is a concise summary of its features and usage.

## I. Create an Azure Machine Learning Workspace

### What is an Azure Machine Learning Workspace?
- Central resource to manage data, compute, models, and endpoints.
- Stores a history of all training jobs for reproducibility, including logs, metrics, outputs, and code snapshots.

### Steps to Create a Workspace
1. **Understand the Azure Machine Learning Service**:
   - Requires an Azure subscription and resource group.
   - Creates supporting resources:
     - **Azure Storage Account**: Stores files, notebooks, metadata.
     - **Azure Key Vault**: Manages secrets securely.
     - **Application Insights**: Monitors predictive services.
     - **Azure Container Registry**: Stores images for environments.

2. **Ways to Create a Workspace**:
   - **Azure Portal**: User interface.
   - **Azure Resource Manager (ARM) Template**.
   - **Azure CLI**.
   - **Azure Machine Learning Python SDK**:

```python
from azure.ai.ml.entities import Workspace

workspace_name = "mlw-example"

ws_basic = Workspace(
    name=workspace_name,
    location="eastus",
    display_name="Basic workspace-example",
    description="This example shows how to create a basic workspace",
)
ml_client.workspaces.begin_create(ws_basic)
```

3. **Explore the Workspace in Azure Portal**:
   - Use the **Azure Machine Learning Studio** for easy management.
   - Grant access using **role-based access control (RBAC)**.

### Access Control Roles
- **Owner**: Full access, can grant permissions.
- **Contributor**: Full access, cannot grant permissions.
- **Reader**: View only.
- **AzureML-Specific Roles**:
  - **AzureML Data Scientist**: Full actions except workspace settings.
  - **AzureML Compute Operator**: Manages compute resources.

## II. Identify Azure Machine Learning Resources

### Key Resources
1. **Workspace**: Top-level resource for managing assets like models and logs.
2. **Compute Resources**:
   - **Compute Instances**: Development environments.
   - **Compute Clusters**: Auto-scaling production workloads.
   - **Kubernetes Clusters**: Deploy models in production.
   - **Attached Computes**: External Azure compute resources.
   - **Serverless Compute**: Fully managed, on-demand compute.

   *Best Practice*: Restrict compute management to administrators.

3. **Datastores**: References to Azure data services, securely managed via Azure Key Vault. Examples include:
   - **workspaceartifactstore**: Stores logs.
   - **workspaceblobstore**: Default datastore for assets.
   - **workspacefilestore**: File storage for notebooks.

## III. Identify Azure Machine Learning Assets

### Key Assets
1. **Models**:
   - Stored as `.pkl` (pickle) or MLflow formats.
   - Versioned for traceability.

2. **Environments**:
   - Define software packages, variables, and settings.
   - Stored as images in Azure Container Registry.

3. **Data Assets**:
   - Reference specific files or folders.
   - Include name, version, and path.

4. **Components**:
   - Reusable code for pipeline steps (e.g., data normalization or model training).

## IV. Train Models in the Workspace

### Options for Training
1. **Automated Machine Learning**:
   - Tests various algorithms and hyperparameters to find the best model.

2. **Notebooks**:
   - Use built-in Jupyter notebooks or Visual Studio Code.
   - Files are stored in the file share of the Azure Storage account.

3. **Scripts as Jobs**:
   - **Command Jobs**: Execute a single script.
   - **Sweep Jobs**: Perform hyperparameter tuning.
   - **Pipeline Jobs**: Run multiple scripts or components.