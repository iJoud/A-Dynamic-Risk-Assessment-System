

import training
import scoring
import deployment
import diagnostics
import reporting
import os
import json
import sys

import subprocess
import pickle
import pandas as pd
from sklearn import metrics

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


##################Check and read new data
#first, read ingestedfiles.txt
ingested_files=[]
with open(os.path.join(os.getcwd(), prod_deployment_path, "ingestedfiles.txt"), "r") as file:
    ingested_files = file.read().splitlines()

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_data_files = os.listdir( os.path.join(os.getcwd(), input_folder_path))

new_files = []
for file in source_data_files:
    if file.endswith('.csv') and file not in ingested_files:
        new_files.append(file)


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if new_files: # not empty
    subprocess.run(['python', 'ingestion.py'])
else:
    sys.exit()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
model_score = None
with open(os.path.join(os.getcwd(), prod_deployment_path, "latestscore.txt"), "r") as file:
    model_score = file.read().splitlines()

latest_score = float(model_score[-1])

# load model
with open(os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl'), 'rb') as f:
    model = pickle.load(f)


new_data = pd.DataFrame()
# read data
for file in os.listdir(os.path.join(os.getcwd(), output_folder_path)):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(os.getcwd(), output_folder_path, file))
        new_data = pd.concat([df, new_data])

# make predictions
y_pred = model.predict(new_data[['lastmonth_activity', 'lastyear_activity',
                                    'number_of_employees']].values)
y_true = new_data['exited'].values

# calculate the F1 score 
f1_score = metrics.f1_score(y_true, y_pred)


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here

if f1_score < latest_score:
    # model drift occured
    # re-training
    subprocess.run(['python', 'training.py'])
else:
    sys.exit()

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
# re-deployment
subprocess.run(['python', 'deployment.py'])

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

subprocess.run(['python', 'apicalls.py'])
subprocess.run(['python', 'reporting.py'])







