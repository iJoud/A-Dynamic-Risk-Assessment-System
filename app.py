from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis
# import predict_exited_from_saved_model
import json
import os

from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list
from scoring import score_model

# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # call the prediction function you created in Step 3
    dataset_location = request.form["dataset_location"]

    if dataset_location.endswith('.csv'):
        data = pd.read_csv(dataset_location)
        predictions = model_predictions(data)

        # add return value for prediction outputs
        return {"model_predictions": str(predictions)}
    else:
        return "please provide CSV file location "


# Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def stats1():
    # check the score of the deployed model
    f1_score = score_model()
    # add return value (a single F1 score number)
    return {"model_score": f1_score}


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats2():
    # check means, medians, and modes for each column
    summary_stat = dataframe_summary()
    # return a list of all calculated summary statistics
    return {"summary_statistics": summary_stat}


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def stats3():
    # check timing and percent NA values
    timing = execution_time()
    na_values = missing_data()
    dependency_check = outdated_packages_list()

    # add return value for all diagnostics
    return {"timing": timing, "missing_values": dict(na_values), 'outdated_packages_list': dependency_check}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
