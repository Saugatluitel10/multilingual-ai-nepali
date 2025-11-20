"""
FastAPI REST API for multilingual sentiment, misinformation, and translation.
Integrates model loading, tokenization, inference, request logging, and rate limiting.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
import torch
import time
import logging
from collections import defaultdict
from threading import Lock

# --- Prometheus Metrics ---
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_ENABLED = True
except ImportError:
    PROMETHEUS_ENABLED = False

# --- Sentry Integration ---
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.1)

# --- Simple Rate Limiter ---
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = Lock()
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        with self.lock:
            reqs = self.requests[client_ip]
            reqs = [t for t in reqs if now - t < self.window_seconds]
            self.requests[client_ip] = reqs
            if len(reqs) < self.max_requests:
                self.requests[client_ip].append(now)
                return True
            return False
rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")

# --- API Key Authentication ---
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY must be set in environment or .env file!")

def api_key_auth(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# --- FastAPI App ---
app = FastAPI(
    title="Multilingual AI REST API",
    description="Endpoints for sentiment, misinformation detection, and translation. Secured with API key.",
    version="1.0.0"
)
if SENTRY_DSN:
    app = SentryAsgiMiddleware(app)
if PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app, include_in_schema=True, should_gzip=True)

# --- Load Models and Tokenizers ---
from models.classification_model import MultilingualClassificationModel
from models.translation_model import MultilingualTranslationModel
import yaml

with open("configs/sentiment_config.yaml") as f:
    sentiment_cfg = yaml.safe_load(f)
with open("configs/misinfo_config.yaml") as f:
    misinfo_cfg = yaml.safe_load(f)

sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_cfg['model']['name'])
sentiment_model = MultilingualClassificationModel(
    model_name=sentiment_cfg['model']['name'],
    num_labels=sentiment_cfg['model']['num_labels'],
    adapter_config=sentiment_cfg['model'].get('adapter_config', None)
)
sentiment_model.eval()

misinfo_tokenizer = AutoTokenizer.from_pretrained(misinfo_cfg['model']['name'])
misinfo_model = MultilingualClassificationModel(
    model_name=misinfo_cfg['model']['name'],
    num_labels=misinfo_cfg['model']['num_labels'],
    adapter_config=misinfo_cfg['model'].get('adapter_config', None)
)
misinfo_model.eval()

translation_model = MultilingualTranslationModel()

# --- Request Models ---
class TextRequest(BaseModel):
    text: str

class TranslationRequest(BaseModel):
    text: str
    src_lang: str = None  # If None, auto-detect
    tgt_lang: str = 'en'

# --- Dependency for Rate Limiting ---
def rate_limit(request: Request):
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

# --- Endpoints ---
@app.post("/predict_sentiment", summary="Predict Sentiment", response_description="Predicted sentiment label and probabilities", tags=["Sentiment"],
          response_model=dict, responses={401: {"description": "Unauthorized"}, 429: {"description": "Rate limit exceeded"}})
async def predict_sentiment(
    req: TextRequest,
    request: Request = None,
    rl: None = Depends(rate_limit),
    auth: None = Depends(api_key_auth)
):
    """
    Predict sentiment for Nepali/English/code-mixed text.
    - **label**: 0=Negative, 1=Neutral, 2=Positive
    - **probs**: Probability scores for each class
    """
    logger.info(f"/predict_sentiment from {request.client.host}: {req.text}")
    inputs = sentiment_tokenizer(req.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = sentiment_model(**inputs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()[0]
        label = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
    return {"label": label, "probs": probs}

@app.post("/detect_misinformation", summary="Detect Misinformation", response_description="Predicted misinformation label and probabilities", tags=["Misinformation"],
          response_model=dict, responses={401: {"description": "Unauthorized"}, 429: {"description": "Rate limit exceeded"}})
async def detect_misinformation(
    req: TextRequest,
    request: Request = None,
    rl: None = Depends(rate_limit),
    auth: None = Depends(api_key_auth)
):
    """
    Detect whether the text contains misinformation (binary/multiclass).
    - **label**: 0=Not Misinformation, 1=Misinformation
    - **probs**: Probability scores for each class
    """
    logger.info(f"/detect_misinformation from {request.client.host}: {req.text}")
    inputs = misinfo_tokenizer(req.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = misinfo_model(**inputs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()[0]
        label = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
    return {"label": label, "probs": probs}

from langdetect import detect

# Supported language codes for mbart-50
LANG_MAP = {
    'en': 'en_XX',
    'ne': 'ne_NEP',
    'hi': 'hi_IN',
}

@app.post("/translate_text", summary="Translate Text", response_description="Translation result", tags=["Translation"],
          response_model=dict, responses={401: {"description": "Unauthorized"}, 429: {"description": "Rate limit exceeded"}, 400: {"description": "Bad request"}})
async def translate_text(
    req: TranslationRequest,
    request: Request = None,
    rl: None = Depends(rate_limit),
    auth: None = Depends(api_key_auth)
):
    """
    Translate text between Nepali, English, and Hindi (auto-detects language if not provided).
    - **src_lang**: Source language code (en, ne, hi)
    - **tgt_lang**: Target language code (en, ne, hi)
    """
    # Auto-detect source language if not provided
    src_lang = req.src_lang or detect(req.text)
    tgt_lang = req.tgt_lang
    logger.info(f"/translate_text from {request.client.host}: {req.text} ({src_lang}->{tgt_lang})")
    # Validate language codes
    if src_lang not in LANG_MAP or tgt_lang not in LANG_MAP:
        raise HTTPException(status_code=400, detail=f"Supported languages: {list(LANG_MAP.keys())}")
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
    src_lang_token = LANG_MAP[src_lang]
    tgt_lang_token = LANG_MAP[tgt_lang]
    inputs = tokenizer(req.text, return_tensors="pt")
    inputs['forced_bos_token_id'] = tokenizer.lang_code_to_id[tgt_lang_token]
    with torch.no_grad():
        translated = translation_model.model.generate(**inputs, max_length=128)
        out_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return {"translation": out_text, "src_lang": src_lang, "tgt_lang": tgt_lang}

@app.get("/", tags=["Info"])
def root():
    """
    Welcome and usage summary.
    """
    return {
        "message": "Welcome to the Multilingual AI REST API!",
        "docs": "/docs",
        "endpoints": [
            "/predict_sentiment",
            "/detect_misinformation",
            "/translate_text",
            "/health"
        ],
        "authentication": "All endpoints require x-api-key header.",
        "rate_limit": "10 requests per minute per IP.",
        "example_curl": "curl -X POST http://localhost:8000/predict_sentiment -H 'x-api-key: YOURKEY' -H 'Content-Type: application/json' -d '{\"text\": \"यो राम्रो छ\"}'"
    }

@app.get("/health", tags=["Info"])
def health():
    """
    Health check endpoint for uptime monitoring.
    """
    try:
        sentiment_loaded = sentiment_model is not None
        misinfo_loaded = misinfo_model is not None
        translation_loaded = translation_model is not None
        return {
            "status": "ok",
            "sentiment_model": sentiment_loaded,
            "misinfo_model": misinfo_loaded,
            "translation_model": translation_loaded
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response
