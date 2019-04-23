from azureml.core.experiment import Experiment
from sklearn.model_selection import train_test_split
import azureml.dataprep as dprep
import azureml.core
import pandas as pd
from azureml.core.workspace import Workspace
import logging
import os
from azureml.core.runconfig import RunConfiguration
from azureml.train.automl import *

def main():
    
    # local compute
    run_user_managed = RunConfiguration()
    run_user_managed.environment.python.user_managed_dependencies = False

    # print to check azure sdk installation
    print(azureml.core.VERSION)

    # create workspace object to connect to omtest workspace in MLSERVICE
    ws = Workspace.from_config('./config.json')
    # default data store
    # ds = ws.get_default_datastore()
    # print(ds)

    # choose a name for the run history container in the workspace
    experiment_name = 'automated-ml-regression'
    # project folder
    project_folder = './automated-ml-regression'

    output = {}
    output['SDK version'] = azureml.core.VERSION
    output['Subscription ID'] = ws.subscription_id
    output['Workspace'] = ws.name
    output['Resource Group'] = ws.resource_group
    output['Location'] = ws.location
    output['Project Directory'] = project_folder
    pd.set_option('display.max_colwidth', -1)
    pd.DataFrame(data=output, index=['']).T

    # stats for all the columns
    dflow = dprep.auto_read_file(path='/Users/omprakashnekkanti/Desktop/Spring 2019/CS445-Capstone/automatedML/cuformodel.csv')
    print(type(dflow))
    dflow.get_profile()

    # filepath as a string
    file_path = os.path.join(os.getcwd(), 'cuformodel.csv')
    print(file_path)
    print(type(file_path))

    # dflow_prepared = dprep.Dataflow.open(file_path)
    # dflow_prepared.get_profile()

    dflow_X = dflow.keep_columns([
        'cell-ID', 'Soil_Name', 'MEAN_Yld_V', 'COUNT_Yld', 'MEAN_Eleva',
        'RANGE_Elev', 'Crop-Type', 'V.A.T(F)', 'R.A.T(F)', 'M.A.T(F)',
        'V.PET(inch)', 'R.PET(inch)', 'M.PET(inch)', 'V.T.R(inch)', 'R.T.R(inch)',
        'M.T.R(inch)'
    ])
    dflow_y = dflow.keep_columns('NormalizedYield')

    x_df = dflow_X.to_pandas_dataframe()
    y_df = dflow_y.to_pandas_dataframe()

    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y_df, test_size=0.2, random_state=223)
    # flatten y_train to 1d array
    y_train.values.flatten()

    automl_settings = {
        "iteration_timeout_minutes": 20,
        "iterations": 40,
        "primary_metric": 'mean_absolute_error',
        "preprocess": False,
        "verbosity": logging.INFO,
        "n_cross_validations": 10
    }

    # local compute
    automated_ml_config = AutoMLConfig(
        task='regression',
        debug_log='automated_ml_errors.log',
        path=project_folder,
        X=x_train.values,
        y=y_train.values.flatten(),
        **automl_settings)
    experiment = Experiment(ws, experiment_name)
    local_run = experiment.submit(automated_ml_config, show_output=True)

if __name__ == "__main__": main()