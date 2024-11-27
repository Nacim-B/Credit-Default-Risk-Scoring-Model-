import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import threading
import uvicorn
import requests

# Load the trained pipeline
with open('pipeline_api.pkl', 'rb') as f:
    pipeline_api = pickle.load(f)

# Initialize FastAPI app
api_app = FastAPI()

# Add CORS middleware to allow requests from the frontend
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
        # Convert input to DataFrame
        data = pd.DataFrame([request.dict()])
        # Predict probabilities
        probabilities = pipeline_api.predict_proba(data)[0]
        # Apply a custom threshold
        cost_ratio = 5
        adjusted_threshold = 1 / (1 + cost_ratio)
        predicted_class = 1 if probabilities[1] >= adjusted_threshold else 0

        return {
            "probabilities": probabilities.tolist(),
            "adjusted_prediction": predicted_class,
            "adjusted_threshold": adjusted_threshold,
        }
    except Exception as e:
        return {"error": str(e)}

# Function to run the FastAPI app in a separate thread
def run_api():
    uvicorn.run(api_app, host="127.0.0.1", port=8000)

# Streamlit dashboard
def run_streamlit():
    # API URL (local for testing)
    API_URL = "http://127.0.0.1:8000/predict_proba"

    st.title("Credit House Classification")

    # Input fields
    income_1 = st.number_input("Source de revenue 1", min_value=0., value=1300., step=100.)
    income_2 = st.number_input("Source de revenue 2", min_value=0., value=0., step=100.)
    income_3 = st.number_input("Source de revenue 3", min_value=0., value=0., step=100.)
    genre = st.radio("Genre", [0, 1], index=0)
    work_duration = st.number_input("Temps d'emploi", max_value=0., value=-300., step=50.)

    # Predict button
    if st.button("Prédire"):
        data = {
            "EXT_SOURCE_2": income_2,
            "EXT_SOURCE_1": income_1,
            "EXT_SOURCE_3": income_3,
            "DAYS_EMPLOYED": work_duration,
            "CODE_GENDER": genre,
        }

        try:
            # Send POST request to the API
            response = requests.post(API_URL, json=data)
            if response.status_code == 200:
                result = response.json()
                st.write(f"Prédiction : {result['adjusted_prediction']}")
                st.write(f"Probabilités : {result['probabilities']}")
            else:
                st.error(f"Erreur de l'API : {response.status_code}, {response.text}")
        except Exception as e:
            st.error(f"Erreur lors de la requête : {str(e)}")

# Main function to run both FastAPI and Streamlit
if __name__ == "__main__":
    # Run FastAPI in a separate thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    # Run Streamlit
    run_streamlit()
