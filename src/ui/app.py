import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import ta
import numpy as np

# Load environment variables
load_dotenv()

# Create necessary directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Configure Streamlit
st.set_page_config(
    page_title="Crypto Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Get API URL from environment or use default
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Title
st.title("Cryptocurrency Price Prediction Dashboard")

def load_data():
    """Load the most recent data from the data directory."""
    try:
        data_dir = os.getenv('DATA_DIR', './data/raw')
        if not os.path.exists(data_dir):
            st.error(f"Data directory {data_dir} does not exist!")
            return None
            
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not files:
            st.warning("No data files found. Please ensure the data collection service is running.")
            return None
            
        latest_file = max(files)
        df = pd.read_csv(os.path.join(data_dir, latest_file))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators."""
    try:
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # Moving averages
        df['SMA_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return df

def create_price_chart(df):
    """Create candlestick chart with indicators."""
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3])

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTCUSDT'
        ), row=1, col=1)

        # Add Moving Averages
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['SMA_5'],
            name='SMA 5',
            line=dict(color='orange')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='blue')
        ), row=1, col=1)

        # Volume bars
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume'
        ), row=2, col=1)

        # Update layout
        fig.update_layout(
            title='BTC/USDT Price Chart',
            yaxis_title='Price (USDT)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def get_prediction(data):
    """Get prediction from FastAPI endpoint."""
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=5)
        if response.status_code == 200:
            return response.json()
        st.warning(f"Prediction service returned status code: {response.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        st.warning("Could not connect to prediction service. Please ensure the API server is running.")
        return None
    except Exception as e:
        st.error(f"Error getting prediction: {str(e)}")
        return None

# Add status indicators
st.sidebar.header("Service Status")

# Check API Service
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✅ API Service: Online")
    else:
        st.sidebar.error("❌ API Service: Offline")
except:
    st.sidebar.error("❌ API Service: Offline")

# Load and process data
df = load_data()

if df is not None:
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Create main price chart
    fig = create_price_chart(df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    
    # Sidebar for predictions
    st.sidebar.header("Price Prediction")
    
    try:
        # Get latest data for prediction
        latest_data = {
            "open": float(df['open'].iloc[-1]),
            "high": float(df['high'].iloc[-1]),
            "low": float(df['low'].iloc[-1]),
            "close": float(df['close'].iloc[-1]),
            "volume": float(df['volume'].iloc[-1]),
            "sma_5": float(df['SMA_5'].iloc[-1]),
            "sma_20": float(df['SMA_20'].iloc[-1]),
            "rsi": float(df['RSI'].iloc[-1])
        }
        
        # Get prediction
        prediction = get_prediction(latest_data)
        
        if prediction:
            st.sidebar.success(f"Predicted Price: ${prediction['predicted_price']:.2f}")
            st.sidebar.info(f"Confidence: {prediction['confidence']:.2%}")
    except Exception as e:
        st.sidebar.warning(f"Error making prediction: {str(e)}")
    
    # Display technical indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Technical Indicators")
        st.write(f"RSI: {df['RSI'].iloc[-1]:.2f}")
        st.write(f"MACD: {df['MACD'].iloc[-1]:.2f}")
        st.write(f"Signal: {df['MACD_Signal'].iloc[-1]:.2f}")
    
    with col2:
        st.subheader("Price Information")
        st.write(f"Current Price: ${df['close'].iloc[-1]:.2f}")
        st.write(f"24h High: ${df['high'].max():.2f}")
        st.write(f"24h Low: ${df['low'].min():.2f}")
    
    # Display raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(df.tail(100))
else:
    st.warning("Please follow these steps to get started:")
    st.markdown("""
    1. Ensure your `.env` file is set up with Binance API credentials
    2. Start the data collection service:
       ```
       python src/data/ingest.py
       ```
    3. Start the API service:
       ```
       uvicorn src.api.main:app --host 0.0.0.0 --port 8000
       ```
    4. Wait a few minutes for initial data collection
    """)

# Add auto-refresh
if st.button("Refresh Data"):
    st.experimental_rerun() 