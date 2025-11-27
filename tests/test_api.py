
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
    response = client.post(
        "/predict/sentiment",
        json={"text": "यो राम्रो छ"},
        headers={"x-api-key": "supersecretkey"}
    )
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert "confidence" in response.json()

def test_detect_misinformation(client):
    response = client.post(
        "/predict/misinformation",
        json={"text": "This is false information."},
        headers={"x-api-key": "supersecretkey"}
    )
    assert response.status_code == 200
    assert "is_misinformation" in response.json()
    assert "confidence" in response.json()

def test_translate_text(client):
    response = client.post(
        "/translate",
        json={"text": "यो राम्रो छ", "language": "ne"},
        headers={"x-api-key": "supersecretkey"}
    )
    if response.status_code != 200:
        print(f"Test failed with status {response.status_code}")
        print(f"Response: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert "translated_text" in data
    assert data["translated_text"] != "Translation coming soon..."
    assert len(data["translated_text"]) > 0
    assert "source_language" in data
