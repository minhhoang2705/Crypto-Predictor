# Crypto Currency Price Prediction

A modern web application for cryptocurrency price prediction using machine learning, featuring real-time market data visualization and trading signals.

***Disclaimer: This web application is not intended for real cryptocurrency investment. It is just a side project.***

## Features

- Real-time cryptocurrency price charts with TradingView-style interface
- Machine learning-based price predictions
- Technical indicators and trading signals
- Secure API with rate limiting and CORS protection
- Modern React frontend with Material-UI
- FastAPI backend with MLflow integration

## Tech Stack

### Frontend
- React 18 with TypeScript
- Material-UI for components
- Lightweight Charts for real-time charting
- Axios for API communication
- Rate limiting and security features

### Backend
- FastAPI for high-performance API
- MLflow for model management
- Pandas & NumPy for data processing
- Scikit-learn for machine learning
- Prometheus for metrics

## Getting Started

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- MLflow server

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Crypto-Currency-Prediction.git
cd Crypto-Currency-Prediction
```

2. Install backend dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd src/frontend
npm install
```

4. Configure environment variables:

Backend (.env in src/api/):
```env
ENVIRONMENT=development
ALLOWED_HOSTS=localhost,127.0.0.1,your-server-ip
ALLOWED_ORIGINS=http://localhost:3000,http://your-server-ip:3000
SECRET_KEY=your-secret-key
MODEL_SERVER_URL=http://localhost:5000
API_PORT=8000
MLFLOW_TRACKING_URI=http://localhost:5000
```

Frontend (.env in src/frontend/):
```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_API_TIMEOUT=30000
REACT_APP_USE_HTTPS=false
REACT_APP_ENABLE_AUTH=false
REACT_APP_API_KEY=your-api-key
```

### Running the Application

1. Start the MLflow server:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

2. Start the FastAPI backend:
```bash
cd src/api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. Start the React frontend:
```bash
cd src/frontend
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

## Security Features

- CORS protection
- Rate limiting
- API key authentication
- Security headers
- HTTPS redirect in production
- Request/Response validation

## API Endpoints

### /predict
- Method: POST
- Description: Get price predictions and trading signals
- Request Body: Technical indicators and market data
- Response: Predictions, confidence scores, and trading signals

### /market-data/{symbol}
- Method: GET
- Description: Get historical market data
- Parameters: symbol, interval
- Response: OHLCV data and technical indicators

## Development

### Code Structure
```
src/
├── api/              # FastAPI backend
│   ├── main.py      # Main API endpoints
│   └── .env         # Backend configuration
├── models/          # ML models
│   └── train.py     # Model training script
├── frontend/        # React frontend
│   ├── src/         # Source code
│   ├── public/      # Static files
│   └── .env         # Frontend configuration
└── requirements.txt # Python dependencies
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TradingView Lightweight Charts
- MLflow
- FastAPI
- React and Material-UI teams
