import requests
import os
import json

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"

# Load config.json and get model path variable
with open('config.json', 'r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])


# Call each API endpoint and store the responses

data = {'dataset_location': "./testdata/testdata.csv"}
response1 = requests.post(URL+'prediction', data=data).text

response2 = requests.get(URL+'scoring').text  # put an API call here
response3 = requests.get(URL+'summarystats').text  # put an API call here
response4 = requests.get(URL+'diagnostics').text  # put an API call here

# combine all API responses
responses = str(response1) + '\n\n' + str(response2) + '\n\n' + \
    str(response3) + '\n\n' + str(response4)  # combine reponses here

# write the responses to your workspace
with open(os.path.join(os.getcwd(), model_path, 'apireturns.txt'), mode="+a") as file:
    file.write(responses)
