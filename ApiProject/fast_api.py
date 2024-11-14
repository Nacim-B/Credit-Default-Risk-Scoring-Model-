from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the trained pipeline
with open('../pipeline_api.pkl', 'rb') as f:
    pipeline_api = pickle.load(f)

# Initialize FastAPI
app = FastAPI()


# Define the input data schema using Pydantic
class PredictionRequest(BaseModel):
    EXT_SOURCE_2: float
    EXT_SOURCE_1: float
    EXT_SOURCE_3: float
    DAYS_EMPLOYED: int
    CODE_GENDER: int
    # Add more features as necessary


# Define the endpoint for probability predictions with custom threshold
@app.post("/predict_proba")
def predict_proba(request: PredictionRequest):
    # Convert the request data to a DataFrame
    data = pd.DataFrame([request.dict()])

    # Get the prediction probabilities
    probabilities = pipeline_api.predict_proba(data)[0]

    # Adjust the threshold based on cost ratio
    cost_ratio = 5  # False negative is 10 times worse than false positive
    adjusted_threshold = 1 / (1 + cost_ratio)

    # Apply the adjusted threshold for the positive class
    predicted_class = 1 if probabilities[1] >= adjusted_threshold else 0

    # Return both the probabilities and the adjusted prediction
    return {
        "probabilities": probabilities.tolist(),
        "adjusted_prediction": predicted_class,
        "adjusted_threshold": adjusted_threshold
    }
# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert the request data to a DataFrame
    data = pd.DataFrame([request.dict()])

    # Make predictions using the loaded pipeline
    prediction = pipeline_api.predict(data)

    # Return the prediction as a response
    return {"prediction": prediction[0]}

