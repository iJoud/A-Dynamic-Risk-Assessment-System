import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import matplotlib.pyplot as plt

from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model

    test_data = pd.DataFrame()
    # read data
    for file in os.listdir(os.path.join(os.getcwd(), test_data_path)):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, file))
            test_data = pd.concat([df, test_data])
    
    y_true = test_data['exited'].values
    y_pred = model_predictions(test_data)


    plot_path = os.path.join(os.getcwd(), model_path, "confusionmatrix.png")
    #write the confusion matrix to the workspace
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred).plot()

    plt.savefig(plot_path)





if __name__ == '__main__':
    score_model()
