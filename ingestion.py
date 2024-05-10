import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']

if not os.path.exists(os.path.join(os.getcwd(), output_folder_path)):
    os.makedirs(os.path.join(os.getcwd(), output_folder_path))


# Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    full_input_path = os.path.join(os.getcwd(), input_folder_path)
    final_df = pd.DataFrame()
    files_list = []

    for file in os.listdir(full_input_path):
        if '.csv' in file:
            files_list.append(file)
            df = pd.read_csv(os.path.join(full_input_path, file))
            final_df = pd.concat([final_df, df])

    # drop duplicates
    final_df.drop_duplicates(inplace=True)

    # save dataframe
    final_df.to_csv(os.path.join(
        os.getcwd(), output_folder_path, "finaldata.csv"))

    with open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), 'a+') as f:
        for file in files_list:
            f.write(file + '\n')


if __name__ == '__main__':
    merge_multiple_dataframe()
