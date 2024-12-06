from fastapi.testclient import TestClient
from ..fast_api import api_app
import pytest

client = TestClient(api_app)


def test_predict_proba_success(mocker):
    """
    Teste si l'API répond correctement avec une entrée valide
    """
    # Simulez la méthode pipeline_api.predict_proba
    mock_predict_proba = mocker.patch("ApiProject.fast_api.pipeline_api.predict_proba")
    mock_predict_proba.return_value = [[0.2, 0.8]]  # Probabilités fictives

    # Données d'entrée valides
    request_data = {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_1": 0.3,
        "EXT_SOURCE_3": 0.7,
        "DAYS_EMPLOYED": -500,
        "CODE_GENDER": 1
    }

    # Envoi de la requête POST à l'API
    response = client.post("/predict_proba", json=request_data)

    # Vérifications
    assert response.status_code == 200
    json_data = response.json()

    # Vérifiez que les champs attendus sont dans la réponse
    assert "probabilities" in json_data
    assert "adjusted_prediction" in json_data
    assert "adjusted_threshold" in json_data

    # Vérifiez que le mock a été appelé
    mock_predict_proba.assert_called_once()


def test_predict_proba_missing_field(mocker):
    """
    Teste si l'API retourne une erreur 422 lorsqu'un champ requis est manquant
    """
    # Données d'entrée invalides (champ manquant)
    request_data = {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_1": 0.3,
        "EXT_SOURCE_3": 0.7,
        "DAYS_EMPLOYED": -500
        # CODE_GENDER est manquant
    }

    # Envoi de la requête POST à l'API
    response = client.post("/predict_proba", json=request_data)

    # Vérifiez que l'API retourne un code 422
    assert response.status_code == 422


def test_predict_proba_invalid_data_type(mocker):
    """
    Teste si l'API retourne une erreur 422 lorsqu'un type de donnée est invalide
    """
    # Données d'entrée invalides (mauvais type pour EXT_SOURCE_2)
    request_data = {
        "EXT_SOURCE_2": "invalid",  # String au lieu de float
        "EXT_SOURCE_1": 0.3,
        "EXT_SOURCE_3": 0.7,
        "DAYS_EMPLOYED": -500,
        "CODE_GENDER": 1
    }

    # Envoi de la requête POST à l'API
    response = client.post("/predict_proba", json=request_data)

    # Vérifiez que l'API retourne un code 422
    assert response.status_code == 422


def test_predict_proba_internal_error(mocker):
    """
    Teste si l'API gère correctement une exception interne
    """
    # Simulez une exception dans pipeline_api.predict_proba
    mock_predict_proba = mocker.patch("fast_api.pipeline_api.predict_proba")
    mock_predict_proba.side_effect = Exception("Internal Error")

    # Données d'entrée valides
    request_data = {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_1": 0.3,
        "EXT_SOURCE_3": 0.7,
        "DAYS_EMPLOYED": -500,
        "CODE_GENDER": 1
    }

    # Envoi de la requête POST à l'API
    response = client.post("/predict_proba", json=request_data)

    # Vérifiez que l'API retourne un code 200 avec un message d'erreur
    assert response.status_code == 200
    json_data = response.json()
    assert "error" in json_data
    assert json_data["error"] == "Internal Error"


def test_predict_proba_threshold_calculation(mocker):
    """
    Teste si le seuil (threshold) et la classification sont calculés correctement
    """
    # Simulez les probabilités de sortie du modèle
    mock_predict_proba = mocker.patch("fast_api.pipeline_api.predict_proba")
    mock_predict_proba.return_value = [[0.2, 0.8]]  # Probabilités pour les classes 0 et 1

    # Données d'entrée valides
    request_data = {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_1": 0.3,
        "EXT_SOURCE_3": 0.7,
        "DAYS_EMPLOYED": -500,
        "CODE_GENDER": 1
    }

    # Envoi de la requête POST à l'API
    response = client.post("/predict_proba", json=request_data)

    # Vérifiez que l'API retourne un code 200
    assert response.status_code == 200
    json_data = response.json()

    # Vérifiez le calcul du seuil
    expected_threshold = 1 / (1 + 3)  # Cost ratio = 3
    assert json_data["adjusted_threshold"] == expected_threshold

    # Vérifiez que la classification est correcte
    assert json_data["adjusted_prediction"] == 1  # Probabilité 0.8 > seuil 0.25
