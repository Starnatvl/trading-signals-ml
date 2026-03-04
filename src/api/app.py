from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from .inference import predict

app = FastAPI(
    title="Trading Signals API",
    description="API для генерации торговых сигналов (BUY/SELL/HOLD)",
    version="1.0.0"
)

class FeaturesRequest(BaseModel):
    features: List[List[float]]
    feature_columns: List[str]
    symbol: str
    window_end_timestamp: int

class PredictionResponse(BaseModel):
    prediction: int
    signal: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(request: FeaturesRequest):
    try:
        df = pd.DataFrame(request.features, columns=request.feature_columns)
        df['symbol'] = request.symbol
        base_ts = request.window_end_timestamp
        df['timestamp'] = [base_ts - (len(df)-1-i)*60000 for i in range(len(df))]
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        if 'close' in df.columns and 'close_price' not in df.columns:
            df.rename(columns={'close': 'close_price'}, inplace=True)

        signal, confidence = predict(df)

        signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        return PredictionResponse(
            prediction=signal,
            signal=signal_map[signal],
            confidence=round(confidence, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Trading Signals API v1.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}