"""
FastAPI-based REST API for multilingual NLP tasks
Supports sentiment analysis, misinformation detection, and translation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
        # TODO: Implement sentiment prediction
        # For now, return mock response
        return SentimentResponse(
            text=input_data.text,
            sentiment="positive",
            confidence=0.85,
            probabilities={
                "positive": 0.85,
                "negative": 0.10,
                "neutral": 0.05
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/misinformation", response_model=MisinfoResponse)
async def predict_misinformation(input_data: TextInput):
    """
    Detect misinformation in given text
    Supports Nepali, English, and code-mixed text
    """
    try:
        # TODO: Implement misinformation detection
        return MisinfoResponse(
            text=input_data.text,
            is_misinformation=False,
            confidence=0.92,
            probabilities={
                "true": 0.92,
                "false": 0.08
            }
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
            "hi": "hin_Deva"
        }
        
        # Detect source language if auto
        src_lang_code = lang_map.get(input_data.language, "npi_Deva") # Default to Nepali if unknown/auto for now
        tgt_lang_code = lang_map.get(target_language, "eng_Latn")
        
        tokenizer = tokenizers["translation"]
        model = models["translation"]
        
        # Tokenize and translate
        inputs = tokenizer(input_data.text, return_tensors="pt")
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
