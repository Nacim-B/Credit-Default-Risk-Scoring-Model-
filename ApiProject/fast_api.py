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
    INSTAL_AMT_PAYMENT_MIN: float
    PREV_CNT_PAYMENT_MEAN: float
    # Add more features as necessary


# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert the request data to a DataFrame
    data = pd.DataFrame([request.dict()])

    # Make predictions using the loaded pipeline
    prediction = pipeline_api.predict(data)

    # Return the prediction as a response
    return {"prediction": prediction[0]}

