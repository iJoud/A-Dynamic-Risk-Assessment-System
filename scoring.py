from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])


# Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    test_data = pd.DataFrame()
    # read data
    for file in os.listdir(os.path.join(os.getcwd(), test_data_path)):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, file))
            test_data = pd.concat([df, test_data])

    # load model
    with open(os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl'), 'rb') as f:
        model = pickle.load(f)

    # make predictions
    y_pred = model.predict(test_data[['lastmonth_activity', 'lastyear_activity',
                                      'number_of_employees']].values)
    y_true = test_data['exited'].values

    # calculate the F1 score 
    f1_score = metrics.f1_score(y_true, y_pred)

    # save score 
    with open(os.path.join(os.getcwd(), model_path, 'latestscore.txt'), 'a+') as f:
        f.write(f"{f1_score} \n")
    
    return f1_score

