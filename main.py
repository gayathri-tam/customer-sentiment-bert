"""FastAPI inference service for a fine-tuned local BERT model."""
from pathlib import Path

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/bert_sentiment"))
tokenizer = None
model = None


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000,
                      description="Customer review to classify")


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float


@asynccontextmanager
async def lifespan(_: FastAPI):
    global tokenizer, model
    if MODEL_PATH.exists():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
    yield


app = FastAPI(title="Customer Sentiment Analysis API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready" if model is not None else "model_not_found"}


@app.post("/predict", response_model=SentimentResponse)
def predict(payload: SentimentRequest) -> SentimentResponse:
    if model is None or tokenizer is None:
        raise HTTPException(503, "Model unavailable. Train it first, then set MODEL_PATH if needed.")
    inputs = tokenizer(payload.text.strip(), return_tensors="pt", truncation=True,
                       max_length=256, padding=True)
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=1)[0]
    prediction = int(probabilities.argmax())
    label = model.config.id2label.get(prediction, str(prediction))
    return SentimentResponse(sentiment=label, confidence=round(float(probabilities[prediction]), 4))
