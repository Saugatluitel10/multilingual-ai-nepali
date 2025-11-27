"""
Streamlit demo UI for Multilingual AI REST API
Allows interactive testing of sentiment, misinformation, and translation endpoints.
"""

import streamlit as st
import requests
import json

# Page Config
st.set_page_config(
    page_title="Multilingual AI - Nepali",
    page_icon="🇳🇵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #D4EDDA;
        color: #155724;
        border: 1px solid #C3E6CB;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9b/Flag_of_Nepal.svg", width=100)
    st.title("Configuration")
    
    # API Settings
    api_url = st.text_input("API URL", value=st.secrets.get("API_URL", "http://localhost:8000"))
    api_key = st.text_input("API Key", type="password", value=st.secrets.get("API_KEY", ""))
    
    st.divider()
    st.markdown("### About")
    st.info(
        "This AI system supports:\n"
        "- Sentiment Analysis\n"
        "- Misinformation Detection\n"
        "- Translation (En ↔ Ne)"
    )
    st.markdown("---")
    st.caption("v1.0.0 | Built with FastAPI & Streamlit")

# Main Content
st.title("🇳🇵 Multilingual AI for Nepal")
st.markdown("### Advanced NLP for Low-Resource Languages")

# Tabs for different tasks
tab1, tab2, tab3 = st.tabs(["😊 Sentiment Analysis", "🕵️ Misinformation", "🌐 Translation"])

# --- Sentiment Analysis Tab ---
with tab1:
    st.header("Sentiment Analysis")
    st.markdown("Analyze the sentiment of Nepali, English, or code-mixed text.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sentiment_text = st.text_area("Enter text to analyze", height=150, key="sentiment_input", placeholder="यो फिल्म एकदम राम्रो छ! (This movie is very good!)")
        analyze_btn = st.button("Analyze Sentiment")
    
    with col2:
        st.markdown("#### Results")
        if analyze_btn and sentiment_text:
            try:
                with st.spinner("Analyzing..."):
                    response = requests.post(
                        f"{api_url}/predict/sentiment",
                        json={"text": sentiment_text},
                        headers={"x-api-key": api_key}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        sentiment = result['sentiment']
                        confidence = result['confidence']
                        
                        # Display metrics
                        st.metric("Sentiment", sentiment.upper(), delta=None)
                        st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                        
                        # Detailed probabilities
                        st.markdown("##### Probabilities")
                        st.json(result['probabilities'])
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
        elif not sentiment_text:
            st.info("Enter text to see results.")

# --- Misinformation Detection Tab ---
with tab2:
    st.header("Misinformation Detection")
    st.markdown("Detect potential misinformation in news or social media text.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        misinfo_text = st.text_area("Enter text to verify", height=150, key="misinfo_input", placeholder="Breaking news: Aliens landed in Kathmandu today!")
        verify_btn = st.button("Verify Information")
    
    with col2:
        st.markdown("#### Results")
        if verify_btn and misinfo_text:
            try:
                with st.spinner("Verifying..."):
                    response = requests.post(
                        f"{api_url}/predict/misinformation",
                        json={"text": misinfo_text},
                        headers={"x-api-key": api_key}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        is_misinfo = result['is_misinformation']
                        confidence = result['confidence']
                        
                        if is_misinfo:
                            st.error("⚠️ Potential Misinformation Detected")
                        else:
                            st.success("✅ Likely True Information")
                            
                        st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                        st.json(result['probabilities'])
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
        elif not misinfo_text:
            st.info("Enter text to verify.")

# --- Translation Tab ---
with tab3:
    st.header("Translation")
    st.markdown("Translate text between English and Nepali.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trans_text = st.text_area("Source Text", height=150, key="trans_input", placeholder="Hello, how are you?")
        target_lang = st.selectbox("Target Language", ["Nepali (ne)", "English (en)"])
        translate_btn = st.button("Translate")
        
        lang_code = "ne" if "Nepali" in target_lang else "en"
    
    with col2:
        st.markdown("#### Translation")
        if translate_btn and trans_text:
            try:
                with st.spinner("Translating..."):
                    response = requests.post(
                        f"{api_url}/translate",
                        json={"text": trans_text, "language": "auto"},
                        params={"target_language": lang_code},
                        headers={"x-api-key": api_key}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        translated = result['translated_text']
                        src_lang = result['source_language']
                        
                        st.text_area("Translated Text", value=translated, height=150, disabled=True)
                        st.caption(f"Detected Source Language: {src_lang}")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
        else:
            st.text_area("Translated Text", value="", height=150, disabled=True)

st.markdown("---")
st.markdown("**API Docs:** [Swagger UI](/docs) | [ReDoc](/redoc)")
st.markdown("**Rate limit:** 10 requests/minute/IP. All endpoints require API key.")
