"""
Streamlit Web App for Multilingual AI Inference
- Sentiment prediction and confidence
- Misinformation probability score
- Translation output (optional)
"""
import streamlit as st
import requests

API_URL = st.secrets["API_URL"] if "API_URL" in st.secrets else "http://localhost:8000"
DEFAULT_API_KEY = st.secrets["API_KEY"] if "API_KEY" in st.secrets else "supersecretkey"

st.title("🌐 Multilingual AI Web App")
st.markdown("Enter Nepali, English, or code-mixed text. Choose task(s) below.")

api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")
text = st.text_area("Input Text", "यो राम्रो छ")

col1, col2, col3 = st.columns(3)
predict_sentiment = col1.checkbox("Sentiment", value=True)
predict_misinfo = col2.checkbox("Misinformation", value=True)
show_translation = col3.checkbox("Translate", value=False)

tgt_lang = None
if show_translation:
    tgt_lang = st.selectbox("Target Language for Translation", ["en", "ne", "hi"], index=0)

if st.button("Run Inference"):
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    if predict_sentiment:
        resp = requests.post(f"{API_URL}/predict_sentiment", headers=headers, json={"text": text})
        if resp.status_code == 200:
            label = resp.json()["label"]
            probs = resp.json()["probs"]
            st.success(f"Sentiment: {['Negative','Neutral','Positive'][label]} ({max(probs)*100:.1f}% confidence)")
            st.json({"probs": probs})
        else:
            st.error(f"Sentiment API error: {resp.status_code} {resp.text}")
    if predict_misinfo:
        resp = requests.post(f"{API_URL}/detect_misinformation", headers=headers, json={"text": text})
        if resp.status_code == 200:
            label = resp.json()["label"]
            probs = resp.json()["probs"]
            st.success(f"Misinformation Score: {probs[1]*100:.1f}% (1=misinfo, 0=not)")
            st.json({"probs": probs})
        else:
            st.error(f"Misinformation API error: {resp.status_code} {resp.text}")
    if show_translation and tgt_lang:
        payload = {"text": text, "tgt_lang": tgt_lang}
        resp = requests.post(f"{API_URL}/translate_text", headers=headers, json=payload)
        if resp.status_code == 200:
            out = resp.json()
            st.success(f"Translation ({out['src_lang']}→{out['tgt_lang']}): {out['translation']}")
        else:
            st.error(f"Translation API error: {resp.status_code} {resp.text}")

st.markdown("---")
st.markdown("**API Docs:** [Swagger UI](/docs) | [ReDoc](/redoc)")
st.markdown("**Rate limit:** 10 requests/minute/IP. All endpoints require API key.")
