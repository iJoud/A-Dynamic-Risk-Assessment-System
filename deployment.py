from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])

# create the directory if it doesn't exist
os.makedirs(prod_deployment_path, exist_ok=True)

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    src_model_path = os.path.join(os.getcwd(), model_path, 'trainedmodel.pkl')
    dst_model_path = os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl')
    shutil.copy(src_model_path, dst_model_path)

    src_scores_path = os.path.join(os.getcwd(), model_path, 'latestscore.txt')
    dst_scores_path = os.path.join(os.getcwd(), prod_deployment_path, 'latestscore.txt')
    shutil.copy(src_scores_path, dst_scores_path)

    src_records_path = os.path.join(os.getcwd(), dataset_csv_path, 'ingestedfiles.txt')
    dst_records_path = os.path.join(os.getcwd(), prod_deployment_path, 'ingestedfiles.txt')
    shutil.copy(src_records_path, dst_records_path)


if __name__ == "__main__":
    store_model_into_pickle()
