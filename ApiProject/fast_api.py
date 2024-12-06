from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle

# Load the trained pipeline
with open('pipeline_api.pkl', 'rb') as f:
    pipeline_api = pickle.load(f)

# Initialize FastAPI app
api_app = FastAPI()

# Add CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input schema for the API
class PredictionRequest(BaseModel):
    EXT_SOURCE_2: float
    EXT_SOURCE_1: float
    EXT_SOURCE_3: float
    DAYS_EMPLOYED: int
    CODE_GENDER: int


# FastAPI endpoint for predictions
@api_app.post("/predict_proba")
def predict_proba(request: PredictionRequest):
    try:
        data = pd.DataFrame([request.dict()])
        probabilities = pipeline_api.predict_proba(data)[0]
        cost_ratio = 3
        adjusted_threshold = 1 / (1 + cost_ratio)
        predicted_class = 1 if probabilities[1] >= adjusted_threshold else 0

        return {
            "probabilities": probabilities.tolist(),
            "adjusted_prediction": predicted_class,
            "adjusted_threshold": adjusted_threshold,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api_app, host="0.0.0.0", port=8000)
