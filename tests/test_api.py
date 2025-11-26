
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Multilingual AI API for Low-Resource Languages"
    assert "endpoints" in response.json()

def test_predict_sentiment():
    response = client.post(
        "/predict/sentiment",
        json={"text": "यो राम्रो छ"},
        headers={"x-api-key": "supersecretkey"}
    )
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert "confidence" in response.json()

def test_detect_misinformation():
    response = client.post(
        "/predict/misinformation",
        json={"text": "This is false information."},
        headers={"x-api-key": "supersecretkey"}
    )
    assert response.status_code == 200
    assert "is_misinformation" in response.json()
    assert "confidence" in response.json()

def test_translate_text():
    response = client.post(
        "/translate",
        json={"text": "यो राम्रो छ", "language": "ne"},
        headers={"x-api-key": "supersecretkey"}
    )
    assert response.status_code == 200
    assert "translated_text" in response.json()
    assert "source_language" in response.json()
