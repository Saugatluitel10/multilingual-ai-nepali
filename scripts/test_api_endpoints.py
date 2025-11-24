"""
Test all main API endpoints for basic QA.
"""
import requests
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "supersecretkey")

headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

def test_endpoint(path, payload=None, method="POST"):
    url = f"{API_URL}{path}"
    try:
        if method == "POST":
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
        else:
            resp = requests.get(url, headers=headers, timeout=10)
        print(f"{method} {path}: {resp.status_code}")
        if resp.status_code == 200:
            print(resp.json())
        else:
            print(resp.text)
    except Exception as e:
        print(f"Error testing {path}: {e}")

if __name__ == "__main__":
    print("\nTesting /predict_sentiment...")
    test_endpoint("/predict_sentiment", {"text": "यो राम्रो छ"})
    print("\nTesting /detect_misinformation...")
    test_endpoint("/detect_misinformation", {"text": "This is fake news."})
    print("\nTesting /translate_text...")
    test_endpoint("/translate_text", {"text": "यो राम्रो छ", "tgt_lang": "en"})
    print("\nTesting /health...")
    test_endpoint("/health", method="GET")
    print("\nTesting /metrics...")
    test_endpoint("/metrics", method="GET")
