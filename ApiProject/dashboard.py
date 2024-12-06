import streamlit as st
import requests

API_URL = "https://credit-default-risk-scoring-model.onrender.com/predict_proba"  # onRender Service

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
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            decision = 'Prêt refusé' if result['adjusted_prediction'] == 1 else 'Prêt accordé'
            st.write(f"Prédiction : {decision}")
            st.write(f"Probabilités : {result['probabilities']}")
        else:
            st.error(f"Erreur de l'API : {response.status_code}, {response.text}")
    except Exception as e:
        st.error(f"Erreur lors de la requête : {str(e)}")
