# Cryptocurrency Price Prediction MLOps Project

This project implements a complete MLOps pipeline for real-time cryptocurrency price prediction with a web-based dashboard.

## Project Structure
```
.
├── data/                  # Data storage
│   └── raw/              # Raw data files
├── models/               # Model storage
├── src/                  # Source code
│   ├── data/            # Data ingestion and processing
│   ├── models/          # Model training and evaluation
│   ├── api/             # FastAPI service
│   ├── ui/              # Streamlit dashboard
│   └── monitoring/      # Monitoring and logging
├── tests/               # Unit and integration tests
└── configs/             # Configuration files
```

## Setup
1. Create Conda environment:
```bash
conda env create -f conda_env.yml
conda activate crypto_prediction
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Binance API credentials
```

## Running the Application

1. Start data collection (Terminal 1):
```bash
conda activate crypto_prediction
python src/data/ingest.py
```

2. Start MLflow server (Terminal 2):
```bash
conda activate crypto_prediction
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

3. Train initial model (Terminal 3):
```bash
conda activate crypto_prediction
python src/models/train.py
```

4. Start API server (Terminal 4):
```bash
conda activate crypto_prediction
uvicorn src.api.main:app --reload
```

5. Launch dashboard (Terminal 5):
```bash
conda activate crypto_prediction
streamlit run src/ui/app.py
```

## Accessing the Application

- Dashboard: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

## Features

- Real-time data collection from Binance
- Automated model training and evaluation
- Interactive price charts with technical indicators
- Real-time price predictions
- Performance monitoring and logging