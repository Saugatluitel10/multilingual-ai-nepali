"""
Streamlit demo UI for Multilingual AI REST API
Allows interactive testing of sentiment, misinformation, and translation endpoints.
"""

import streamlit as st
import requests

API_URL = st.secrets["API_URL"] if "API_URL" in st.secrets else "http://localhost:8000"
DEFAULT_API_KEY = st.secrets["API_KEY"] if "API_KEY" in st.secrets else "supersecretkey"

st.title("🌐 Multilingual AI API Demo")
api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")

endpoint = st.selectbox("Choose Task", [
    "Sentiment Analysis",
    "Misinformation Detection",
    "Translation"
])

text = st.text_area("Input Text", "यो राम्रो छ")

if endpoint == "Translation":
    tgt_lang = st.selectbox("Target Language", ["en", "ne", "hi"])
    src_lang = st.text_input("Source Language (optional)", "")

if st.button("Submit"):
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    if endpoint == "Sentiment Analysis":
        url = f"{API_URL}/predict_sentiment"
        payload = {"text": text}
    elif endpoint == "Misinformation Detection":
        url = f"{API_URL}/detect_misinformation"
        payload = {"text": text}
    elif endpoint == "Translation":
        url = f"{API_URL}/translate_text"
        payload = {"text": text, "tgt_lang": tgt_lang}
        if src_lang.strip():
            payload["src_lang"] = src_lang.strip()
    else:
        st.error("Select a valid endpoint.")
        st.stop()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            st.success("Response:")
            st.json(resp.json())
        else:
            st.error(f"Error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.markdown("---")
st.markdown("**API Docs:** [Swagger UI](/docs) | [ReDoc](/redoc)")
st.markdown("**Rate limit:** 10 requests/minute/IP. All endpoints require API key.")
