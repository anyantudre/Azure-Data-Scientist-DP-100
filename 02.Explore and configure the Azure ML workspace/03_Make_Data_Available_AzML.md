# Azure Machine Learning: Making Data Available

## Overview
In Azure Machine Learning, managing and accessing data effectively is essential for training models and generating predictions. This guide explains URIs, datastores, and data assets to ensure secure and efficient data connections.

---

## 1. Understanding URIs
URIs identify data locations. Common protocols include:
- **http(s):** Public/private Azure Blob Storage or public URLs.
- **abfs(s):** Azure Data Lake Storage Gen 2.
- **azureml:** Data in a datastore.

### Example
For a private Blob Storage container `training-data/datastore-path/diabetes.csv`:
- Use **http(s)** with authentication (e.g., SAS token).
- Use **azureml** if the container is registered as a datastore.

**Tip:** Avoid embedding sensitive data in code; use datastores for secure connections.

---

## 2. Datastores
Datastores securely store connection info to cloud data sources.

### Benefits
- Simplify data access with secure URIs.
- Enable data discovery and sharing.

### Authentication Methods
1. **Credential-based:** Service principal, SAS token, or account key.
2. **Identity-based:** Microsoft Entra or managed identity.

### Supported Azure Sources
- Azure Blob Storage
- Azure File Share
- Azure Data Lake Gen 2

### Creating a Datastore
#### Using Account Key
```python
from azure.ai.ml.entities import AzureBlobDatastore, AccountKeyConfiguration

blob_datastore = AzureBlobDatastore(
    name="blob_example",
    account_name="mytestblobstore",
    container_name="data-container",
    credentials=AccountKeyConfiguration(account_key="XXXxxxXXX"),
)
ml_client.create_or_update(blob_datastore)
```

#### Using SAS Token
```python
from azure.ai.ml.entities import SasTokenConfiguration

blob_datastore = AzureBlobDatastore(
    name="blob_sas_example",
    account_name="mytestblobstore",
    container_name="data-container",
    credentials=SasTokenConfiguration(sas_token="?xx=XXXX..."),
)
ml_client.create_or_update(blob_datastore)
```

---

## 3. Data Assets
Data assets simplify access to storage locations and metadata.

### Types
1. **URI file:** Points to a specific file.
2. **URI folder:** Points to a folder.
3. **MLTable:** Tabular data with schema.

### Benefits
- Shareable and reusable.
- Versioned metadata.
- Streamlined integration in workflows.

### Creating Data Assets
#### URI File Asset
Supported paths:
- Local: `./<path>`
- Azure Blob Storage: `wasbs://<account>.blob.core.windows.net/<container>/<file>`
- Datastore: `azureml://datastores/<datastore_name>/paths/<file>`

```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

uri_file_data = Data(
    path="azureml://datastores/datastore_name/paths/data.csv",
    type=AssetTypes.URI_FILE,
    name="data_file",
    version="1.0",
)
ml_client.data.create_or_update(uri_file_data)
```

#### Usage in a Job
```python
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

df = pd.read_csv(args.input_data)
print(df.head())
```

#### URI Folder Asset
```python
uri_folder_data = Data(
    path="azureml://datastores/datastore_name/paths/folder/",
    type=AssetTypes.URI_FOLDER,
    name="data_folder",
    version="1.0",
)
ml_client.data.create_or_update(uri_folder_data)
```

#### Usage in a Job
```python
import argparse
import pandas as pd
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

all_files = glob.glob(args.input_data + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files))
print(df.head())
```

#### MLTable Asset
##### Define Schema (MLTable File)
```yaml
type: mltable
paths:
  - pattern: ./*.csv
transformations:
  - read_delimited:
      delimiter: ','
```

##### Create Asset
```python
mltable_data = Data(
    path="azureml://datastores/datastore_name/paths/table/",
    type=AssetTypes.MLTABLE,
    name="ml_table_data",
    version="1.0",
)
ml_client.data.create_or_update(mltable_data)
```

##### Usage in a Job
```python
import argparse
import mltable

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

tbl = mltable.load(args.input_data)
df = tbl.to_pandas_dataframe()
print(df.head())
```
