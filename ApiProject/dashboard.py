import pandas as pd
import streamlit as st
import requests


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

@st.fragment
def main():
    URL = 'http://127.0.0.1:8000/predict'

    st.title('Credit House Classification')

    income_1 = st.number_input('Source de revenue 1',
                               min_value=0., value=1300., step=100.)

    income_2 = st.number_input('Source de revenue 2',
                               min_value=0., value=0., step=100.)
    income_3 = st.number_input('Source de revenue 3',
                               min_value=0., value=0., step=100.)

    genre = st.radio(
        "Genre",
        [0, 1],
        index=None,
    )

    work_duration = st.number_input('Temps demploi',
                                    max_value=0., value=-300., step=50.)

    predict_btn = st.button('Prédire')

    if predict_btn:
        data = {
            "EXT_SOURCE_2": income_2,
            "EXT_SOURCE_1": income_1,
            "EXT_SOURCE_3": income_3,
            "DAYS_EMPLOYED": work_duration,
            "CODE_GENDER": genre
        }
        pred = request_prediction(URL, data)
        st.write(
            f'Prediciton pour ce client : {pred['prediction']}')


if __name__ == '__main__':
    main()
