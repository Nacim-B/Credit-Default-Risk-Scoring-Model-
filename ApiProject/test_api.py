import requests

# Define the API URL
url_predict = "http://127.0.0.1:8000/predict"
url_predict_proba = "http://127.0.0.1:8000/predict_proba"

# Define the input data
data = {
    "EXT_SOURCE_2": 0,
    "EXT_SOURCE_1": 20,
    "EXT_SOURCE_3": 0,
    "DAYS_EMPLOYED": 0,
    "CODE_GENDER": 1
}

# Send a POST request
response = requests.post(url_predict, json=data)

# Send a POST request
response_proba = requests.post(url_predict_proba, json=data)

# Print the prediction result
print(response.json())
print(response_proba.json())
