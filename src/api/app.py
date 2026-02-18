"""
FastAPI приложение для инференса
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

from src.api.inference import TradingSignalPredictor

app = FastAPI(
    title="Trading Signals API",
    description="API для генерации торговых сигналов (BUY/SELL/HOLD)",
    version="1.0.0"
)

# Загрузка модели при старте
predictor = TradingSignalPredictor("models/final_model.pkl")

class PredictionRequest(BaseModel):
    features: List[List[float]]
    
class PredictionResponse(BaseModel):
    prediction: int
    signal: str
    confidence: float

@app.get("/")
def root():
    return {"message": "Trading Signals API v1.0", "status": "running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Генерация торгового сигнала
    
    Request body:
        features: Массив массивов фичей (N шагов × M признаков)
        
    Returns:
        prediction: 1 (BUY), -1 (SELL), 0 (HOLD)
        signal: Текстовое описание
        confidence: Уверенность модели
    """
    try:
        prediction = predictor.predict(request.features)
        
        signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        
        return PredictionResponse(
            prediction=prediction,
            signal=signal_map[prediction],
            confidence=0.85  # TODO: добавить реальную confidence из модели
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy"}