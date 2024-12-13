# Experiment with Azure Machine Learning

## Find the best classification model with Automated Machine Learning

### Introduction

Going through trial and error to find the best performing model can be time-consuming. Instead of manually having to test and evaluate various configurations to train a machine learning model, you can automate it with automated machine learning or AutoML.

AutoML allows you to try multiple preprocessing transformations and algorithms with your data to find the best machine learning model.


Imagine you want to find the best performing classification model. You can create an AutoML experiment using the visual interface of Azure Machine Learning studio, the Azure command-line interface (CLI), or the Python software development kit (SDK).

 Note

You can use AutoML for other tasks such as regression, forecasting, image classification, and natural language processing. Learn more about when you can use AutoML.

As a data scientist, you may prefer to configure your AutoML experiment with the Python SDK.

Learning objectives
In this module, you'll learn how to:

Prepare your data to use AutoML for classification.
Configure and run an AutoML experiment.
Evaluate and compare models.

### I. Preprocess data and configure featurization

Before you can run an automated machine learning (AutoML) experiment, you need to prepare your data. When you want to train a classification model, you'll only need to provide the training data.

After you've collected the data, you need to create a data asset in Azure Machine Learning. In order for AutoML to understand how to read the data, you need to create a MLTable data asset that includes the schema of the data.

You can create a MLTable data asset when your data is stored in a folder together with a MLTable file. When you have created the data asset, you can specify it as input with the following code:

```python
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import Input

my_training_data_input = Input(type=AssetTypes.MLTABLE, path="azureml:input-data-automl:1")
```
Once you've created the data asset, you can configure the AutoML experiment. Before AutoML trains a classification model, preprocessing transformations can be applied to your data.

##### Understand scaling and normalization
AutoML applies scaling and normalization to numeric data automatically, helping prevent any large-scale features from dominating training. During an AutoML experiment, multiple scaling or normalization techniques will be applied.

##### Configure optional featurization
You can choose to have AutoML apply preprocessing transformations, such as:

Missing value imputation to eliminate nulls in the training dataset.
Categorical encoding to convert categorical features to numeric indicators.
Dropping high-cardinality features, such as record IDs.
Feature engineering (for example, deriving individual date parts from DateTime features)
By default, AutoML will perform featurization on your data. You can disable it if you don't want the data to be transformed.

If you do want to make use of the integrated featurization function, you can customize it. For example, you can specify which imputation method should be used for a specific feature.

After an AutoML experiment is completed, you'll be able to review which scaling and normalization methods have been applied. You'll also get notified if AutoML has detected any issues with the data, like whether there are missing values or class imbalance.

### II. Run an Automated Machine Learning experiment

To run an automated machine learning (AutoML) experiment, you can configure and submit the job with the Python SDK.

The algorithms AutoML uses will depend on the task you specify. When you want to train a classification model, AutoML will choose from a list of classification algorithms:

Logistic Regression
Light Gradient Boosting Machine (GBM)
Decision Tree
Random Forest
Naive Bayes
Linear Support Vector Machine (SVM)
XGBoost
And others...

##### Restrict algorithm selection
By default, AutoML will randomly select from the full range of algorithms for the specified task. You can choose to block individual algorithms from being selected; which can be useful if you know that your data isn't suited to a particular type of algorithm. You also may want to block certain algorithms if you have to comply with a policy that restricts the type of machine learning algorithms you can use in your organization.

##### Configure an AutoML experiment
When you use the Python SDK (v2) to configure an AutoML experiment or job, you configure the experiment using the automl class. For classification, you'll use the automl.classification function as shown in the following example:
```python
from azure.ai.ml import automl

# configure the classification job
classification_job = automl.classification(
    compute="aml-cluster",
    experiment_name="auto-ml-class-dev",
    training_data=my_training_data_input,
    target_column_name="Diabetic",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True
)
```

##### Specify the primary metric
One of the most important settings you must specify is the primary_metric. The primary metric is the target performance metric for which the optimal model will be determined. Azure Machine Learning supports a set of named metrics for each type of task.

To retrieve the list of metrics available when you want to train a classification model, you can use the ClassificationPrimaryMetrics function as shown here:
```python
from azure.ai.ml.automl import ClassificationPrimaryMetrics
 
list(ClassificationPrimaryMetrics)
```

##### Set the limits
Training machine learning models will cost compute. To minimize costs and time spent on training, you can set limits to an AutoML experiment or job by using set_limits().

There are several options to set limits to an AutoML experiment:

timeout_minutes: Number of minutes after which the complete AutoML experiment is terminated.
trial_timeout_minutes: Maximum number of minutes one trial can take.
max_trials: Maximum number of trials, or models that will be trained.
enable_early_termination: Whether to end the experiment if the score isn't improving in the short term.
```python
classification_job.set_limits(
    timeout_minutes=60, 
    trial_timeout_minutes=20, 
    max_trials=5,
    enable_early_termination=True,
)
```
To save time, you can also run multiple trials in parallel. When you use a compute cluster, you can have as many parallel trials as you have nodes. The maximum number of parallel trials is therefore related to the maximum number of nodes your compute cluster has. If you want to set the maximum number of parallel trials to be less than the maximum number of nodes, you can use max_concurrent_trials.


##### Set the training properties
AutoML will try various combinations of featurization and algorithms to train a machine learning model. If you already know that certain algorithms aren't well-suited for your data, you can exclude (or include) a subset of the available algorithms.

You can also choose whether you want to allow AutoML to use ensemble models.


##### Submit an AutoML experiment
You can submit an AutoML job with the following code:

```python
# submit the AutoML job
returned_job = ml_client.jobs.create_or_update(
    classification_job
)
```
You can monitor AutoML job runs in the Azure Machine Learning studio. To get a direct link to the AutoML job by running the following code:

```python
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)
```



### III. Evaluate and compare models
When an automated machine learning (AutoML) experiment has completed, you'll want to review the models that have been trained and decide which one performed best.

In the Azure Machine Learning studio, you can select an AutoML experiment to explore its details.

On the Overview page of the AutoML experiment run, you can review the input data asset and the summary of the best model. To explore all models that have been trained, you can select the Models tab:

![]()

##### Explore preprocessing steps
When you've enabled featurization for your AutoML experiment, data guardrails will automatically be applied too. The three data guardrails that are supported for classification models are:

Class balancing detection.
Missing feature values imputation.
High cardinality feature detection.
Each of these data guardrails will show one of three possible states:

Passed: No problems were detected and no action is required.
Done: Changes were applied to your data. You should review the changes AutoML has made to your data.
Alerted: An issue was detected but couldn't be fixed. You should review the data to fix the issue.
Next to data guardrails, AutoML can apply scaling and normalization techniques to each model that is trained. You can review the technique applied in the list of models under Algorithm name.

For example, the algorithm name of a model listed may be MaxAbsScaler, LightGBM. MaxAbsScaler refers to a scaling technique where each feature is scaled by its maximum absolute value. LightGBM refers to the classification algorithm used to train the model.

##### Retrieve the best run and its model
When you're reviewing the models in AutoML, you can easily identify the best run based on the primary metric you specified. In the Azure Machine Learning studio, the models are automatically sorted to show the best performing model at the top.

In the Models tab of the AutoML experiment, you can edit the columns if you want to show other metrics in the same overview. By creating a more comprehensive overview that includes various metrics, it may be easier to compare models.

To explore a model even further, you can generate explanations for each model that has been trained. When configuring an AutoML experiment, you can specify that explanations should be generated for the best performing model. If however, you're interested in the interpretability of another model, you can select the model in the overview and select Explain model.



## Track model training in Jupyter notebooks with MLflow

### Introduction
Imagine you're a data scientist for a company developing an application for a cancer research lab. The application is to be used by researchers who upload an image of tissue to determine whether or not it's healthy tissue. You're asked to train the model to detect breast cancer on a large image database that contains examples of healthy and unhealthy tissue

You're familiar with Jupyter notebooks, which you plan to use to develop the model. However, you want to periodically retrain the model to create a better performing model that must then be deployed so that researchers can use the model in the application they're using.

You'll learn how to track model training in notebooks with MLflow in Azure Machine Learning.

Learning objectives
In this module, you'll learn how to:

Configure MLflow to use in notebooks
Use MLflow for model tracking in notebooks


### I. Configure MLflow for model tracking in notebooks

As a data scientist, you'll want to develop a model in a notebook as it allows you to quickly test and run code.

Anytime you train a model, you want the results to be reproducible. By tracking and logging your work, you can review your work at any time and decide what the best approach is to train a model.

MLflow is an open-source library for tracking and managing your machine learning experiments. In particular, MLflow Tracking is a component of MLflow that logs everything about the model you're training, such as parameters, metrics, and artifacts.

To use MLflow in notebooks in the Azure Machine Learning workspace, you'll need to install the necessary libraries and set Azure Machine Learning as the tracking store. When you've configured MLflow, you can start to use MLflow when training models in notebooks.

##### Configure MLflow in notebooks

###### Use Azure Machine Learning notebooks
Within the Azure Machine Learning workspace, you can create notebooks and connect the notebooks to an Azure Machine Learning managed compute instance.

When you're running a notebook on a compute instance, MLflow is already configured, and ready to be used.

To verify that the necessary packages are installed, you can run the following code:

```python
pip show mlflow
pip show azureml-mlflow
```
The mlflow package is the open-source library. The azureml-mlflow package contains the integration code of Azure Machine Learning with MLflow.

###### Use MLflow on a local device
When you prefer working in notebooks on a local device, you can also make use of MLflow. You'll need to configure MLflow by completing the following steps:

1. Install the mlflow and azureml-mlflow package.
```python
pip install mlflow
pip install azureml-mlflow
```

2. Navigate to the Azure Machine Learning studio.

3. Select the name of the workspace you're working on in the top right corner of the studio.

4. Select View all properties in Azure portal. A new tab will open to take you to the Azure Machine Learning service in the Azure portal.

5. Copy the value of the MLflow tracking URI.

6. Use the following code in your local notebook to configure MLflow to point to the Azure Machine Learning workspace, and set it to the workspace tracking URI
```python
mlflow.set_tracking_uri = "MLFLOW-TRACKING-URI"
```

### II. Train and track models in notebooks
As a data scientist, you use notebooks to experiment and train models. To group model training results, you'll use experiments. To track model metrics with MLflow when training a model in a notebook, you can use MLflow's logging capabilities.

##### Create an MLflow experiment
You can create a MLflow experiment, which allows you to group runs. If you don't create an experiment, MLflow will assume the default experiment with name Default.

To create an experiment, run the following command in a notebook:

```python
import mlflow

mlflow.set_experiment(experiment_name="heart-condition-classifier")
```

##### Log results with MLflow
Now, you're ready to train your model. To start a run tracked by MLflow, you'll use start_run(). Next, to track the model, you can:

Enable autologging.
Use custom logging.

###### Enable autologging
MLflow supports automatic logging for popular machine learning libraries. If you're using a library that is supported by autolog, then MLflow tells the framework you're using to log all the metrics, parameters, artifacts, and models that the framework considers relevant.

You can turn on autologging by using the autolog method for the framework you're using. For example, to enable autologging for XGBoost models you can use mlflow.xgboost.autolog().

A notebook cell that trains and tracks a classification model using autologging may be similar to the following code example:

```python
from xgboost import XGBClassifier

with mlflow.start_run():
    mlflow.xgboost.autolog()

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```
As soon as mlflow.xgboost.autolog() is called, MLflow will start a run within an experiment in Azure Machine Learning to start tracking the experiment's run.

When the job has completed, you can review all logged metrics in the studio.


###### Use custom logging
Additionally, you can manually log your model with MLflow. Manually logging models is helpful when you want to log supplementary or custom information that isn't logged through autologging.

Common functions used with custom logging are:

mlflow.log_param(): Logs a single key-value parameter. Use this function for an input parameter you want to log.
mlflow.log_metric(): Logs a single key-value metric. Value must be a number. Use this function for any output you want to store with the run.
mlflow.log_artifact(): Logs a file. Use this function for any plot you want to log, save as image file first.
mlflow.log_model(): Logs a model. Use this function to create an MLflow model, which may include a custom signature, environment, and input examples.

To use custom logging in a notebook, start a run and log any metric you want:
```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

with mlflow.start_run():
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
```
Custom logging gives you more flexibility, but also creates more work as you'll have to define any parameter, metric, or artifact you want to log.

When the job has completed, you can review all logged metrics in the studio.



### Ressources
- [Working with tables in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-mltable?view=azureml-api-2&tabs=cli)





