"""
FastAPI-based REST API for multilingual NLP tasks
Supports sentiment analysis, misinformation detection, and translation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

from contextlib import asynccontextmanager

# Request/Response Models
class TextInput(BaseModel):
    text: str
    language: Optional[str] = "auto"  # auto, ne, en, mixed

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict

class MisinfoResponse(BaseModel):
    text: str
    is_misinformation: bool
    confidence: float
    probabilities: dict

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str

# Global model storage (will be loaded on startup)
models = {
    "sentiment": None,
    "misinfo": None,
    "translation": None
}

tokenizers = {
    "sentiment": None,
    "misinfo": None,
    "translation": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on API startup"""
    print("Loading models...")
    try:
        # Load Translation Model (NLLB-200)
        print("Loading NLLB-200 model...")
        model_name = "facebook/nllb-200-distilled-600M"
        models["translation"] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizers["translation"] = AutoTokenizer.from_pretrained(model_name)
        print("✅ Translation model loaded!")

        # Load Sentiment Model (XLM-RoBERTa)
        print("Loading Sentiment model...")
        sent_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        models["sentiment"] = AutoModelForSequenceClassification.from_pretrained(sent_model_name)
        tokenizers["sentiment"] = AutoTokenizer.from_pretrained(sent_model_name)
        print("✅ Sentiment model loaded!")

        # Load Misinformation Model
        print("Loading Misinformation model...")
        misinfo_model_path = "models/misinfo/checkpoints"
        models["misinfo"] = AutoModelForSequenceClassification.from_pretrained(misinfo_model_path)
        tokenizers["misinfo"] = AutoTokenizer.from_pretrained(misinfo_model_path)
        print("✅ Misinformation model loaded!")
    except Exception as e:
        print(f"❌ Error loading translation model: {e}")
    
    print("Models loaded successfully!")
    yield
    # Clean up resources if needed
    models.clear()
    tokenizers.clear()

app = FastAPI(
    title="Multilingual AI API",
    description="NLP API for Nepali-English code-mixed text",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multilingual AI API for Low-Resource Languages",
        "version": "1.0.0",
        "endpoints": [
            "/predict/sentiment",
            "/predict/misinformation",
            "/translate",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict/sentiment", response_model=SentimentResponse)
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for given text
    Supports Nepali, English, and code-mixed text
    """
    try:
        if models["sentiment"] is None:
             # Fallback to mock if model fails to load (or raise error)
             # For now, let's raise error to be explicit
             raise HTTPException(status_code=503, detail="Sentiment model not loaded")

        tokenizer = tokenizers["sentiment"]
        model = models["sentiment"]

        # Tokenize and predict
        encoded_input = tokenizer(input_data.text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
        labels = ["negative", "neutral", "positive"]
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        top_sentiment = labels[ranking[0]]
        confidence = float(scores[ranking[0]])
        
        probabilities = {
            "negative": float(scores[0]),
            "neutral": float(scores[1]),
            "positive": float(scores[2])
        }

        return SentimentResponse(
            text=input_data.text,
            sentiment=top_sentiment,
            confidence=confidence,
            probabilities=probabilities
        )
    except Exception as e:
        print(f"❌ Error in predict_sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/misinformation", response_model=MisinfoResponse)
async def predict_misinformation(input_data: TextInput):
    """
    Detect misinformation in given text
    Supports Nepali, English, and code-mixed text
    """
    try:
        if models["misinfo"] is None:
            raise HTTPException(status_code=503, detail="Misinformation model not loaded")

        tokenizer = tokenizers["misinfo"]
        model = models["misinfo"]

        # Tokenize and predict
        encoded_input = tokenizer(input_data.text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        # Labels: 0 -> Real, 1 -> Fake
        labels = ["real", "fake"]
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        top_label = labels[ranking[0]]
        is_misinfo = top_label == "fake"
        confidence = float(scores[ranking[0]])
        
        probabilities = {
            "real": float(scores[0]),
            "fake": float(scores[1])
        }

        return MisinfoResponse(
            text=input_data.text,
            is_misinformation=is_misinfo,
            confidence=confidence,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(input_data: TextInput, target_language: str = "en"):
    """
    Translate text between Nepali and English
    """
    try:
        if models["translation"] is None:
            raise HTTPException(status_code=503, detail="Translation model not loaded")
            
        # Map languages to NLLB codes
        lang_map = {
            "en": "eng_Latn",
            "ne": "npi_Deva",
            "hi": "hin_Deva",
            "fr": "fra_Latn",
            "es": "spa_Latn",
            "zh": "zho_Hans"
        }
        
        # Detect source language if auto
        # Simple heuristic or default to Nepali if input looks like Devanagari, else English?
        # For NLLB, providing the correct source language code improves quality.
        # If 'auto', we might default to 'npi_Deva' if target is 'en', and 'eng_Latn' if target is 'ne'.
        # But let's stick to the map.
        
        src_lang_code = lang_map.get(input_data.language, "npi_Deva") 
        # If input language is not in map, default to Nepali (common use case here)
        
        tgt_lang_code = lang_map.get(target_language, "eng_Latn")
        
        tokenizer = tokenizers["translation"]
        model = models["translation"]
        
        # Tokenize and translate
        inputs = tokenizer(input_data.text, return_tensors="pt")
        
        # NLLB requires setting the forced_bos_token_id to the target language
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_code), 
            max_length=128
        )
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        return TranslationResponse(
            original_text=input_data.text,
            translated_text=translated_text,
            source_language=input_data.language,
            target_language=target_language
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
