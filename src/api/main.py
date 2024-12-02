import os
import logging
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from prometheus_client import make_asgi_app, Counter, Histogram

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Crypto Price Prediction API")

# Add prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Define metrics
PREDICTION_REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction requests"
)

class PredictionInput(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    sma_5: float
    sma_20: float
    rsi: float

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence: float

@app.on_event("startup")
async def startup_event():
    """Load the ML model on startup."""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Load the latest model from MLflow
        runs = mlflow.search_runs(experiment_ids=["1"])
        if len(runs) == 0:
            raise Exception("No models found in MLflow")
            
        latest_run = runs.sort_values("start_time", ascending=False).iloc[0]
        model_uri = f"runs:/{latest_run.run_id}/model"
        
        global model
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Crypto Price Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Make price predictions."""
    try:
        PREDICTION_REQUEST_COUNT.inc()
        with PREDICTION_LATENCY.time():
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data.dict()])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Calculate confidence (using model's feature importances)
            confidence = np.mean(model.feature_importances_)
            
            return PredictionResponse(
                predicted_price=float(prediction),
                confidence=float(confidence)
            )
            
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 