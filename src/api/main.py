import os
import logging
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from prometheus_client import make_asgi_app, Counter, Histogram
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API",
    description="API for cryptocurrency price predictions",
    version="1.0.0",
)

# Security configurations
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost").split(",")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Add security middlewares
if ENVIRONMENT == "production":
    app.add_middleware(HTTPSRedirectMiddleware)  # Redirect HTTP to HTTPS

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=ALLOWED_HOSTS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
    return response

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

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
    ema_5: float
    ema_20: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_middle: float
    bb_upper: float
    bb_lower: float
    price_momentum: float
    volume_ma: float
    volume_momentum: float

class TimeToTargetPrediction(BaseModel):
    minutes_to_target: float
    confidence: float

class TradingSignal(BaseModel):
    signal: str  # "buy", "sell", or "hold"
    confidence: float

class PredictionResponse(BaseModel):
    current_price: float
    predictions: Dict[str, Dict[str, TimeToTargetPrediction]]  # e.g., {"1%": {"up": {...}, "down": {...}}}
    trading_signals: Dict[str, TradingSignal]  # e.g., {"1%": {...}, "2%": {...}}
    historical_prices: Dict[str, List[float]]  # timestamps and prices for plotting

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataset."""
    df = df.copy()
    
    # Basic indicators
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
    df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
    
    # Price momentum
    df['price_momentum'] = df['close'].pct_change(periods=5)
    
    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_momentum'] = df['volume'].pct_change(periods=5)
    
    return df

@app.on_event("startup")
async def startup_event():
    """Load the ML models on startup."""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5001")
        
        # Get all experiments
        experiments = mlflow.search_experiments()
        if not experiments:
            logger.warning("No experiments found in MLflow")
            return
            
        # Search across all experiments
        runs = mlflow.search_runs()
        if len(runs) == 0:
            logger.warning("No models found in MLflow")
            return
            
        latest_run = runs.sort_values("start_time", ascending=False).iloc[0]
        model_uri = f"runs:/{latest_run.run_id}/models"
        
        global models
        models = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Models loaded successfully from run {latest_run.run_id}")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.warning("API will start without models. Predictions will not be available.")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Crypto Price Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Make predictions for time-to-target and trading signals."""
    try:
        if not globals().get('models'):
            raise HTTPException(
                status_code=503,
                detail="Models are not loaded. Please wait for models to be available."
            )
            
        PREDICTION_REQUEST_COUNT.inc()
        with PREDICTION_LATENCY.time():
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data.dict()])
            
            # Make predictions for different target percentages
            predictions = {}
            trading_signals = {}
            
            target_percentages = [0.01, 0.02, 0.05]  # 1%, 2%, 5%
            
            for pct in target_percentages:
                pct_str = f"{int(pct*100)}%"
                
                # Predict time to reach targets
                time_up = models[f"up_{int(pct*100)}pct"].predict(input_df)[0]
                time_down = models[f"down_{int(pct*100)}pct"].predict(input_df)[0]
                
                # Get trading signal
                signal_proba = models[f"signal_{int(pct*100)}pct"].predict_proba(input_df)[0]
                signal_pred = models[f"signal_{int(pct*100)}pct"].predict(input_df)[0]
                
                # Convert signal to string
                if signal_pred == 1:
                    signal = "buy"
                elif signal_pred == -1:
                    signal = "sell"
                else:
                    signal = "hold"
                
                # Store predictions
                predictions[pct_str] = {
                    "up": TimeToTargetPrediction(
                        minutes_to_target=float(time_up),
                        confidence=float(models[f"up_{int(pct*100)}pct"].score(input_df, [time_up]))
                    ),
                    "down": TimeToTargetPrediction(
                        minutes_to_target=float(time_down),
                        confidence=float(models[f"down_{int(pct*100)}pct"].score(input_df, [time_down]))
                    )
                }
                
                # Store trading signals
                trading_signals[pct_str] = TradingSignal(
                    signal=signal,
                    confidence=float(max(signal_proba))
                )
            
            # Get historical prices from raw data files
            data_files = sorted(os.listdir('data/raw'))[-100:]  # Last 100 files
            historical_data = []
            
            for file in data_files:
                df = pd.read_csv(os.path.join('data/raw', file))
                historical_data.append({
                    'timestamp': pd.to_datetime(df['timestamp']).iloc[-1],
                    'price': df['close'].iloc[-1]
                })
            
            # Sort historical data by timestamp
            historical_data = sorted(historical_data, key=lambda x: x['timestamp'])
            
            return PredictionResponse(
                current_price=float(input_data.close),
                predictions=predictions,
                trading_signals=trading_signals,
                historical_prices={
                    'timestamps': [d['timestamp'].isoformat() for d in historical_data],
                    'prices': [float(d['price']) for d in historical_data]
                }
            )
            
    except HTTPException:
        raise
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