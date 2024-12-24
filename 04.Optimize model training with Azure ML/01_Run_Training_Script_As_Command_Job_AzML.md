# Run a Training Script as a Command Job in Azure Machine Learning

## Introduction
Running scripts as command jobs in Azure Machine Learning ensures scalability, repeatability, and automation for production scenarios. This module explains how to:

- Convert a notebook to a script.
- Test scripts in a terminal.
- Run a script as a command job.
- Use parameters in a command job.

---

## I. Convert a Notebook to a Script

### 1. Remove Nonessential Code
Exclude exploratory code like `print()` and `df.describe()` to minimize costs and compute time.

### 2. Refactor Code into Functions
Simplify code readability and testing by organizing it into smaller functions:

```python
# Main function
def main(csv_file):
    df = get_data(csv_file)
    X_train, X_test, y_train, y_test = split_data(df)

# Function to read data
def get_data(path):
    return pd.read_csv(path)

# Function to split data
def split_data(df):
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
               'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
    return train_test_split(X, y, test_size=0.30, random_state=0)
```

### 3. Test the Script
Run scripts in the terminal of a compute instance using:

```bash
python train.py
```
This displays `print` outputs and errors directly in the terminal.

---

## II. Run a Script as a Command Job

### Configure and Submit a Command Job
Use the Python SDK v2 to configure a job with parameters like `code`, `command`, `environment`, `compute`, `display_name`, and `experiment_name`.

Example:

```python
from azure.ai.ml import command

# Configure job
job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-model",
    experiment_name="train-classification-model"
)

# Submit job
returned_job = ml_client.create_or_update(job)
```

Monitor and review jobs in the Azure Machine Learning studio, grouped by `experiment_name`.

---

## III. Use Parameters in a Command Job

### Working with Script Arguments
Use the `argparse` library to define parameters in the script:

```python
import argparse

# Main function
def main(args):
    df = get_data(args.training_data)

# Function to read data
def get_data(path):
    return pd.read_csv(path)

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest="training_data", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
```

### Passing Arguments to a Script
Run the script with parameters in the terminal:

```bash
python train.py --training_data diabetes.csv
```

### Passing Arguments to a Command Job
Include parameters in the `command` field:

```python
job = command(
    code="./src",
    command="python train.py --training_data diabetes.csv",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    display_name="train-model",
    experiment_name="train-classification-model"
)
```

---

By following these steps, you can efficiently run scripts as command jobs in Azure Machine Learning, enabling flexibility and automation in machine learning workflows.
