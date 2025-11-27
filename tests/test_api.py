
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_read_main(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Multilingual AI API for Low-Resource Languages"
    assert "endpoints" in response.json()

def test_predict_sentiment(client):
    # Test positive sentiment
    response = client.post(
        "/predict/sentiment",
        json={"text": "I love this movie! It is amazing."},
        headers={"x-api-key": "supersecretkey"}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.5

    # Test negative sentiment
    response = client.post(
        "/predict/sentiment",
        json={"text": "I hate this. It is terrible."},
        headers={"x-api-key": "supersecretkey"}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["sentiment"] == "negative"

def test_translate_text(client):
    # Test Nepali translation
    response = client.post(
        "/translate",
        json={"text": "Hello", "language": "en"},
        params={"target_language": "ne"},
        headers={"x-api-key": "supersecretkey"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["translated_text"]) > 0
    
    # Test French translation
    response = client.post(
        "/translate",
        json={"text": "Hello", "language": "en"},
        params={"target_language": "fr"},
        headers={"x-api-key": "supersecretkey"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["translated_text"]) > 0
