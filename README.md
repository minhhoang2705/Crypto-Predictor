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
<<<<<<< HEAD

## Deployment

### Server Setup

1. Install required packages:
```bash
sudo apt update
sudo apt install nginx python3-venv python3-dev build-essential
```

2. Create application directory:
```bash
sudo mkdir -p /var/www/crypto-prediction
sudo chown -R $USER:$USER /var/www/crypto-prediction
```

3. Set up Python virtual environment:
```bash
python3 -m venv /home/$USER/app/venv
source /home/$USER/app/venv/bin/activate
pip install -r requirements.txt
```

4. Configure systemd service:
```bash
sudo cp deployment/crypto-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable crypto-api
sudo systemctl start crypto-api
```

5. Configure Nginx:
```bash
sudo cp deployment/nginx.conf /etc/nginx/sites-available/crypto-prediction
sudo ln -s /etc/nginx/sites-available/crypto-prediction /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

6. Set up SSL with Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### CI/CD Setup

1. Add the following secrets to your GitHub repository:
   - `SSH_PRIVATE_KEY`: SSH key for server access
   - `SERVER_IP`: Your server's IP address
   - `SERVER_USER`: SSH user for deployment

2. Configure your domain's DNS to point to your server's IP address

3. Update the following files with your domain and paths:
   - `deployment/nginx.conf`: Replace `your-domain.com`
   - `deployment/crypto-api.service`: Replace `USER` with your username
   - `src/frontend/.env.production`: Update API URL

### CI/CD Pipeline

The project includes two GitHub Actions workflows:

1. Backend CI/CD (`backend.yml`):
   - Runs tests on Python 3.8 and 3.9
   - Performs linting with flake8
   - Runs pytest with coverage
   - Deploys to server on main/master branch

2. Frontend CI/CD (`frontend.yml`):
   - Tests on Node.js 16.x and 18.x
   - Runs ESLint
   - Runs tests with coverage
   - Builds production bundle
   - Deploys to server on main/master branch

### Monitoring

1. Set up Prometheus monitoring:
```bash
sudo apt install prometheus
sudo cp deployment/prometheus.yml /etc/prometheus/
sudo systemctl restart prometheus
```

2. Set up Grafana dashboards:
```bash
sudo apt install grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

Access Grafana at `http://your-domain.com:3000`
=======
>>>>>>> fecff429b869aaaf9d5b96ff7e159129422ef06e
