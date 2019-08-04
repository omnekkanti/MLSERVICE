from azureml.core.runconfig import RunConfiguration
import azureml.core
import pandas as pd
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
import logging
import os
import azureml.dataprep as dprep
from sklearn.model_selection import train_test_split
import azureml.train.automl as aml
from azureml.core.experiment import Experiment
import numpy as np
from sklearn.metrics import mean_absolute_error
from azureml.train.automl.automlexplainer import retrieve_model_explanation
from azureml.core.model import Model
from azureml.core.image import ContainerImage
from azureml.core.image.image import Image
from azureml.core import Webservice
from azureml.core.webservice import AciWebservice

# try:
# setting the local env to hadnle missing packages
run_user_managed = RunConfiguration()
run_user_managed.environment.python.user_managed_dependencies = False

# Create workspace object for existing one and create an experiment
ws = Workspace.from_config('subscription.json')
print(ws.name, ws.location, ws.resource_group, ws.location, sep='\t')
experiment = Experiment(workspace=ws, name='experiment1')

# full path to training data,testing data
file_path1 = os.path.join(os.getcwd(), "cumodelwo2014.csv")
dflowtr = dprep.auto_read_file(path=file_path1)
file_path2 = os.path.join(os.getcwd(), "test2014.csv")
dflowte = dprep.auto_read_file(path=file_path2)

# Specifying x(causal) and y(response) attributes in training data
dflowtr_x = dflowtr.keep_columns([
    'cell-ID', 'Soil_Name', 'MEAN_Yld_V', 'COUNT_Yld', 'MEAN_Eleva',
    'RANGE_Elev', 'Crop-Type', 'V.A.T(F)', 'R.A.T(F)', 'M.A.T(F)',
    'V.PET(inch)', 'R.PET(inch)', 'M.PET(inch)', 'V.T.R(inch)', 'R.T.R(inch)',
    'M.T.R(inch)'
])
dflowtr_y = dflowtr.keep_columns('NormalizedYield')
# causal factors in training dataframe
trainingx_df = dflowtr_x.to_pandas_dataframe()
# response variable in training dataframe
trainingy_df = dflowtr_y.to_pandas_dataframe()

# Specifying x(causal) and y(response) attributes in testing data
dflowte_x = dflowte.keep_columns([
    'cell-ID', 'Soil_Name', 'MEAN_Yld_V', 'COUNT_Yld', 'MEAN_Eleva',
    'RANGE_Elev', 'Crop-Type', 'V.A.T(F)', 'R.A.T(F)', 'M.A.T(F)',
    'V.PET(inch)', 'R.PET(inch)', 'M.PET(inch)', 'V.T.R(inch)', 'R.T.R(inch)',
    'M.T.R(inch)'
])

dflowte_y = dflowte.keep_columns('NormalizedYield')
# causal factors in testing dataframe
testx_df = dflowte_x.to_pandas_dataframe()
# response variable in testing dataframe
testy_df = dflowte_y.to_pandas_dataframe()

# Experiment parameters in dictionary
automl_settings = {
    "iteration_timeout_minutes": 10,
    "iterations": 10,
    "primary_metric": 'normalized_mean_absolute_error',
    "preprocess": True,
    "verbosity": logging.INFO,
    "n_cross_validations": 10
}

# AutoML object for running experiment
automated_ml_config = aml.AutoMLConfig(task='regression',
                                       debug_log='automated_ml_errors.log',
                                       path='./automated-ml-regression',
                                       X=trainingx_df.values,
                                       y=trainingy_df.values.flatten(),
                                       model_explainability=True,
                                       **automl_settings)

# Submit experiment to get AutoMLRun object
local_run = experiment.submit(automated_ml_config, show_output=True)

# Best pipeline, Model from the best pipeline, in the bunch of runs(experiment)
best_run, fitted_model = local_run.get_output()
# this tells which algorithm was used in the model from best pipeline
print(best_run.get_details())

# Predicting vlaues in a list for test data
y_predict = fitted_model.predict(testx_df.values)

# Printing the predictions to csv
f = open('predict2014.csv', 'w')

# Mean Absolute Error
mae = mean_absolute_error(testy_df, y_predict)
# Range of y in training data
rod = 1-0.08
# Primary metric for model- Normalized mean absolute error
nmae = mae/rod
print('Normalized Mean Absolute Error: ', nmae)

# Explainable AI
shap_values, expected_values, overall_summary, overall_imp, per_class_summary, per_class_imp = retrieve_model_explanation(
    best_run)

# Overall feature importance (features name in descending order)
print(*overall_imp, sep="\n")

# Accuracy of the Model
mape_sum = count = 0
for actual_val, predict_val in zip(testy_df['NormalizedYield'], y_predict):
    abs_error = actual_val - predict_val
    if abs_error < 0:
        abs_error = abs_error * (-1)
    mape_sum += abs_error/actual_val
    count += 1


mean_abs_percent_error = mape_sum/count
print("Model MAPE:")
print(mean_abs_percent_error)
print()
print("Model Accuracy:")
print(1 - mean_abs_percent_error)
# % abs error(|ypredicted-yoriginal|/yoriginal) will have its own distribution as each record gets its own prediction from model
# mean of that abs percentage error(mape) distribution is 10%
# Lets say standard deviation in that distribution is x(I did not calculate as it is besides the point)
# meaning of accuracy:
# if you select a record at random and use model on it to predict, there will be error in prediction.
# This error value comes from interval [10+3x%,10-3x%](this proportion is based on original value of y which we dont know).
# we dont know if it is positive or negative error and also we dont know original value of y for the record to caliculate the error magnitude
# So on average there will be 10% error(10% of original value of y which we dont know) for a record
# Therefore 90% of the original value which we dont know is predicted by model
