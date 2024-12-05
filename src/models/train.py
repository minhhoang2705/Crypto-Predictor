import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from glob import glob
import ta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataset."""
    logger.info(f"Calculating technical indicators for DataFrame with shape: {df.shape}")
    
    # Sort by timestamp to ensure correct calculations
    df = df.sort_values('timestamp')
    
    # Use smaller windows for indicators due to limited data
    # Basic indicators
    df['sma_5'] = df['close'].rolling(window=2, min_periods=1).mean()  # Added min_periods=1
    df['sma_20'] = df['close'].rolling(window=3, min_periods=1).mean()  # Added min_periods=1
    df['ema_5'] = df['close'].ewm(span=2, adjust=False, min_periods=1).mean()  # Added min_periods=1
    df['ema_20'] = df['close'].ewm(span=3, adjust=False, min_periods=1).mean()  # Added min_periods=1
    
    # RSI with smaller window
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=3, min_periods=1).mean()  # Added min_periods=1
    loss = (-delta.where(delta < 0, 0)).rolling(window=3, min_periods=1).mean()  # Added min_periods=1
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)  # Fill NaN RSI values with neutral value
    
    # MACD with smaller windows
    df['macd'] = df['close'].ewm(span=3, adjust=False, min_periods=1).mean() - df['close'].ewm(span=6, adjust=False, min_periods=1).mean()
    df['macd_signal'] = df['macd'].ewm(span=2, adjust=False, min_periods=1).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands with smaller window
    df['bb_middle'] = df['close'].rolling(window=3, min_periods=1).mean()
    df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=3, min_periods=1).std()
    df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=3, min_periods=1).std()
    
    # Price momentum with smaller window
    df['price_momentum'] = df['close'].pct_change(periods=2)
    df['price_momentum'] = df['price_momentum'].fillna(0)  # Fill NaN momentum with 0
    
    # Volume indicators with smaller windows
    df['volume_ma'] = df['volume'].rolling(window=3, min_periods=1).mean()
    df['volume_momentum'] = df['volume'].pct_change(periods=2)
    df['volume_momentum'] = df['volume_momentum'].fillna(0)  # Fill NaN momentum with 0
    
    # Forward fill any remaining NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"Technical indicators calculated. DataFrame shape: {df.shape}")
    return df

def create_target_variables(df, target_percentages=[0.01, 0.02, 0.05]):
    """Create target variables for price movement prediction."""
    logger.info(f"Creating target variables for DataFrame with shape: {df.shape}")
    df = df.copy()
    
    for pct in target_percentages:
        logger.info(f"Processing {pct*100}% target")
        # Calculate target price levels
        target_price_up = df['close'] * (1 + pct)
        target_price_down = df['close'] * (1 - pct)
        
        # Initialize columns for time to reach targets
        col_name_up = f'time_to_{int(pct*100)}pct_up'
        col_name_down = f'time_to_{int(pct*100)}pct_down'
        df[col_name_up] = np.nan
        df[col_name_down] = np.nan
        
        # Calculate trading signals
        signal_col = f'signal_{int(pct*100)}pct'
        df[signal_col] = 0  # 0: hold, 1: buy, -1: sell
        
        # Look forward to find when price reaches targets
        for i in range(len(df)-1):
            if i % 100 == 0:  # Log progress every 100 rows
                logger.info(f"Processing row {i}/{len(df)} for {pct*100}% target")
                
            future_prices = df['close'].iloc[i+1:]
            future_times = df['timestamp'].iloc[i+1:]
            
            # Time to reach upper target
            up_idx = future_prices[future_prices >= target_price_up.iloc[i]].index
            if len(up_idx) > 0:
                time_diff = (df['timestamp'].loc[up_idx[0]] - df['timestamp'].iloc[i]).total_seconds() / 60
                df.loc[df.index[i], col_name_up] = time_diff
                if time_diff <= 5:  # If target reached within 5 minutes
                    df.loc[df.index[i], signal_col] = 1  # Buy signal
            else:
                # If target not reached, use a default value
                df.loc[df.index[i], col_name_up] = 60  # Assume it takes more than 60 minutes
            
            # Time to reach lower target
            down_idx = future_prices[future_prices <= target_price_down.iloc[i]].index
            if len(down_idx) > 0:
                time_diff = (df['timestamp'].loc[down_idx[0]] - df['timestamp'].iloc[i]).total_seconds() / 60
                df.loc[df.index[i], col_name_down] = time_diff
                if time_diff <= 5:  # If target reached within 5 minutes
                    df.loc[df.index[i], signal_col] = -1  # Sell signal
            else:
                # If target not reached, use a default value
                df.loc[df.index[i], col_name_down] = 60  # Assume it takes more than 60 minutes
    
    logger.info(f"Target variables created. DataFrame shape: {df.shape}")
    logger.info(f"Columns after target creation: {df.columns.tolist()}")
    return df

def load_and_prepare_data():
    """Load and prepare data from raw files."""
    # Get all CSV files in the raw data directory
    data_files = glob('data/raw/*.csv')
    logger.info(f"Found {len(data_files)} data files")
    
    if not data_files:
        raise ValueError("No data files found in data/raw directory")
    
    # Combine all files
    dfs = []
    for file in data_files:
        logger.info(f"Reading file: {file}")
        try:
            df = pd.read_csv(file)
            if len(df) > 0:  # Only add non-empty files
                logger.info(f"File {file} shape: {df.shape}")
                dfs.append(df)
            else:
                logger.warning(f"Skipping empty file: {file}")
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid data found in any files")
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {df.shape}")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Remove duplicates and sort by timestamp
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
    logger.info(f"DataFrame shape after removing duplicates: {df.shape}")
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    logger.info(f"DataFrame shape after technical indicators: {df.shape}")
    
    # Create target variables
    df = create_target_variables(df)
    logger.info(f"DataFrame shape after target creation: {df.shape}")
    
    # Drop the last few rows since they don't have enough future data
    df = df.iloc[:-5]  # Drop last 5 rows
    
    logger.info(f"Final DataFrame shape: {df.shape}")
    return df

def train_models():
    """Train and log price movement and time-to-target models to MLflow."""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print(f"DataFrame shape after preparation: {df.shape}")
    
    # Features for prediction
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_5', 'sma_20', 'ema_5', 'ema_20', 'rsi',
        'macd', 'macd_signal', 'macd_hist',
        'bb_middle', 'bb_upper', 'bb_lower',
        'price_momentum', 'volume_ma', 'volume_momentum'
    ]
    
    print(f"Features shape: {df[features].shape}")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5001")
    
    # Train models for different target percentages
    target_percentages = [0.01, 0.02, 0.05]
    
    with mlflow.start_run() as run:
        models = {}
        
        for pct in target_percentages:
            print(f"\nTraining models for {pct*100}% price movement...")
            
            # Prepare data for time-to-target prediction
            X = df[features]
            y_time_up = df[f'time_to_{int(pct*100)}pct_up']
            y_time_down = df[f'time_to_{int(pct*100)}pct_down']
            y_signal = df[f'signal_{int(pct*100)}pct']
            
            print(f"Target shape for {pct*100}%: {y_time_up.shape}")
            print(f"Available columns in df: {df.columns.tolist()}")
            
            if len(X) == 0:
                raise ValueError("No data available for training after preparation!")
            
            # Split data
            X_train, X_test, y_time_up_train, y_time_up_test = train_test_split(X, y_time_up, test_size=0.2, random_state=42)
            _, _, y_time_down_train, y_time_down_test = train_test_split(X, y_time_down, test_size=0.2, random_state=42)
            _, _, y_signal_train, y_signal_test = train_test_split(X, y_signal, test_size=0.2, random_state=42)
            
            # Train time-to-target up model
            model_up = RandomForestRegressor(n_estimators=100, random_state=42)
            model_up.fit(X_train, y_time_up_train)
            
            # Train time-to-target down model
            model_down = RandomForestRegressor(n_estimators=100, random_state=42)
            model_down.fit(X_train, y_time_down_train)
            
            # Train trading signal model
            model_signal = RandomForestClassifier(n_estimators=100, random_state=42)
            model_signal.fit(X_train, y_signal_train)
            
            # Calculate metrics
            up_score = model_up.score(X_test, y_time_up_test)
            down_score = model_down.score(X_test, y_time_down_test)
            signal_score = model_signal.score(X_test, y_signal_test)
            
            # Log metrics
            mlflow.log_metric(f"r2_score_up_{int(pct*100)}pct", up_score)
            mlflow.log_metric(f"r2_score_down_{int(pct*100)}pct", down_score)
            mlflow.log_metric(f"accuracy_signal_{int(pct*100)}pct", signal_score)
            
            # Store models
            models[f"up_{int(pct*100)}pct"] = model_up
            models[f"down_{int(pct*100)}pct"] = model_down
            models[f"signal_{int(pct*100)}pct"] = model_signal
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("data_size", len(df))
        mlflow.log_param("features", features)
        mlflow.log_param("target_percentages", target_percentages)
        
        # Log models
        mlflow.sklearn.log_model(models, "models")
        
        print(f"\nModels trained and logged with run_id: {run.info.run_id}")
        print("\nModel Scores:")
        for pct in target_percentages:
            print(f"\n{pct*100}% price movement:")
            print(f"Time-to-target up R² score: {models[f'up_{int(pct*100)}pct'].score(X_test, y_time_up_test):.4f}")
            print(f"Time-to-target down R² score: {models[f'down_{int(pct*100)}pct'].score(X_test, y_time_down_test):.4f}")
            print(f"Trading signal accuracy: {models[f'signal_{int(pct*100)}pct'].score(X_test, y_signal_test):.4f}")

if __name__ == "__main__":
    train_models()
