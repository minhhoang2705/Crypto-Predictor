import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib
from dotenv import load_dotenv
import ta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CryptoPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))
        mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME', 'crypto_prediction'))

    def create_features(self, df):
        """Create advanced technical indicators as features."""
        try:
            # Price-based indicators
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log1p(df['returns'])
            
            # Trend indicators
            df['SMA_5'] = ta.trend.sma_indicator(df['close'], window=5)
            df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['EMA_5'] = ta.trend.ema_indicator(df['close'], window=5)
            df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
            
            # Momentum indicators
            df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['MACD'] = ta.trend.MACD(df['close']).macd()
            df['MACD_Signal'] = ta.trend.MACD(df['close']).macd_signal()
            df['MACD_Diff'] = ta.trend.MACD(df['close']).macd_diff()
            df['Stoch_RSI'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
            
            # Volatility indicators
            df['BB_High'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
            df['BB_Low'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
            df['BB_Width'] = df['BB_High'] - df['BB_Low']
            df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Volume indicators
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['VWAP'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
            
            # Price patterns
            df['Higher_High'] = df['high'] > df['high'].shift(1)
            df['Lower_Low'] = df['low'] < df['low'].shift(1)
            
            # Time-based features
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            
            # Target variable (next minute's return)
            df['target'] = df['close'].shift(-1) / df['close'] - 1
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise

    def prepare_data(self, df):
        """Prepare data for training."""
        try:
            # Create features
            df = self.create_features(df)
            
            # Define features for training
            feature_columns = [
                'returns', 'log_returns', 
                'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20',
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff', 'Stoch_RSI',
                'BB_High', 'BB_Low', 'BB_Width', 'ATR',
                'OBV', 'VWAP',
                'Higher_High', 'Lower_Low',
                'hour', 'day_of_week'
            ]
            
            # Prepare features and target
            X = df[feature_columns]
            y = df['target']
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            return X_scaled, y, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def train(self):
        """Train the model with advanced features and hyperparameter tuning."""
        try:
            # Load data
            data_dir = os.getenv('DATA_DIR', './data/raw')
            files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if not files:
                raise Exception("No data files found")
            
            # Combine all data files
            dfs = []
            for file in files:
                df = pd.read_csv(os.path.join(data_dir, file))
                dfs.append(df)
            df = pd.concat(dfs, ignore_index=True)
            df = df.sort_values('timestamp')
            
            # Prepare data
            X, y, feature_columns = self.prepare_data(df)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Model parameters
            model_params = {
                'n_estimators': 200,
                'max_depth': 10,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
            
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(model_params)
                
                # Initialize and train model
                self.model = XGBRegressor(**model_params)
                
                # Train with cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    self.model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        # early_stopping_rounds=10,
                        verbose=True
                    )
                    
                    y_pred = self.model.predict(X_val)
                    mse = mean_squared_error(y_val, y_pred)
                    mape = mean_absolute_percentage_error(y_val, y_pred)
                    cv_scores.append({'mse': mse, 'mape': mape})
                
                # Calculate and log metrics
                avg_mse = np.mean([score['mse'] for score in cv_scores])
                avg_mape = np.mean([score['mape'] for score in cv_scores])
                mlflow.log_metric('avg_mse', avg_mse)
                mlflow.log_metric('avg_mape', avg_mape)
                
                # Log feature importance
                importance_dict = dict(zip(feature_columns, self.model.feature_importances_))
                mlflow.log_params(importance_dict)
                
                # Save model and scalers
                model_dir = os.getenv('MODEL_DIR', './models')
                os.makedirs(model_dir, exist_ok=True)
                
                model_path = os.path.join(model_dir, 'model.json')
                scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
                
                self.model.save_model(model_path)
                joblib.dump(self.feature_scaler, scaler_path)
                
                # Log artifacts
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(scaler_path)
                
                logger.info(f"Model trained successfully. MSE: {avg_mse:.6f}, MAPE: {avg_mape:.2%}")
                
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

if __name__ == "__main__":
    predictor = CryptoPricePredictor()
    predictor.train() 