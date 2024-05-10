
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_model_path = os.path.join(config['prod_deployment_path'])

# Function to get the test data
def get_test_data():

    test_data = pd.DataFrame()
    # read data
    for file in os.listdir(os.path.join(os.getcwd(), test_data_path)):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, file))
            test_data = pd.concat([df, test_data])
    
    return test_data


# Function to get model predictions
def model_predictions(test_data: pd.DataFrame):
    # read the deployed model and a test dataset, calculate predictions
    # load model
    with open(os.path.join(os.getcwd(), prod_model_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    X = test_data[['lastmonth_activity',
                   'lastyear_activity', 'number_of_employees']].values

    y_pred = model.predict(X)

    return list(y_pred)


# Function to get summary statistics
def dataframe_summary():

    # means, medians, and standard deviations
    df = get_test_data()

    # calculate summary statistics here
    numeric_columns = ['lastmonth_activity',
                       'lastyear_activity', 'number_of_employees']

    summary_statistics = dict()
    for col in numeric_columns:
        summary_statistics[col] = [df[col].mean(),
                                   df[col].median(),
                                   df[col].std()]

    return [summary_statistics] # return value should be a list containing all summary statistics

def missing_data():
    test_data = pd.DataFrame()
    # read data
    for file in os.listdir(os.path.join(os.getcwd(), test_data_path)):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, file))
            test_data = pd.concat([df, test_data])

    # return percentage of NA for each column in the data frame
    return  (test_data.isna().sum() / len(test_data)) *  100 

# Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    os.system('python3 training.py')
    training_time = timeit.default_timer() - start_time

    return [ingestion_time, training_time] # return a list of 2 timing values in seconds

# Function to check dependencies
def outdated_packages_list():
    # get a list of outdated packages
    outdated = subprocess.check_output(['pip', 'list','--outdated'])

    return str(outdated)


if __name__ == '__main__':

    model_predictions(get_test_data())
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
