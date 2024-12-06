from fastapi.testclient import TestClient
from ..fast_api import api_app
from unittest.mock import patch


client = TestClient(api_app)


def test_predict_proba_success():
    # Valid input data
    request_data = {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_1": 0.3,
        "EXT_SOURCE_3": 0.7,
        "DAYS_EMPLOYED": -500,
        "CODE_GENDER": 1
    }

    response = client.post("/predict_proba", json=request_data)

    assert response.status_code == 200
    json_data = response.json()
    assert "probabilities" in json_data
    assert "adjusted_prediction" in json_data
    assert "adjusted_threshold" in json_data


def test_predict_proba_missing_field():
    # Missing a required field
    request_data = {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_1": 0.3,
        "EXT_SOURCE_3": 0.7,
        "DAYS_EMPLOYED": -500
        # CODE_GENDER is missing
    }

    response = client.post("/predict_proba", json=request_data)

    assert response.status_code == 422


def test_predict_proba_invalid_data_type():
    # Invalid data type for EXT_SOURCE_2 (should be a float)
    request_data = {
        "EXT_SOURCE_2": "invalid",  # String instead of float
        "EXT_SOURCE_1": 0.3,
        "EXT_SOURCE_3": 0.7,
        "DAYS_EMPLOYED": -500,
        "CODE_GENDER": 1
    }

    response = client.post("/predict_proba", json=request_data)

    assert response.status_code == 422


@patch("fast_api.pipeline_api.predict_proba")
def test_predict_proba_internal_error(mock_predict_proba):
    # Simulate an exception in the pipeline
    mock_predict_proba.side_effect = Exception("Internal Error")

    request_data = {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_1": 0.3,
        "EXT_SOURCE_3": 0.7,
        "DAYS_EMPLOYED": -500,
        "CODE_GENDER": 1
    }

    response = client.post("/predict_proba", json=request_data)

    assert response.status_code == 200
    json_data = response.json()
    assert "error" in json_data
    assert json_data["error"] == "Internal Error"
