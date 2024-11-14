import requests

# Define the API URL
url = "http://127.0.0.1:8000/predict"

# Define the input data
data = {
    "EXT_SOURCE_2": 200,
    "EXT_SOURCE_1": 0,
    "EXT_SOURCE_3": 200,
    "DAYS_EMPLOYED": -3000,
    "CODE_GENDER": 1
}

# Send a POST request
response = requests.post(url, json=data)

# Print the prediction result
print(response.json())
