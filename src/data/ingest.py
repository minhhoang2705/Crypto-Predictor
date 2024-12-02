import os
import time
import logging
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CryptoDataIngestion:
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        self.symbol = 'BTCUSDT'
        self.interval = Client.KLINE_INTERVAL_1MINUTE
        
    def fetch_real_time_data(self):
        """Fetch real-time cryptocurrency data from Binance."""
        try:
            klines = self.client.get_historical_klines(
                self.symbol,
                self.interval,
                str(int(time.time() * 1000) - 60000)  # Last minute
            )
            
            if not klines:
                logger.warning("No data received from Binance")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_asset_volume',
                'number_of_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Process DataFrame
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.astype({
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            })
            
            # Save data
            self._save_data(df)
            logger.info(f"Successfully fetched and saved data at {datetime.now()}")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Error fetching data from Binance: {e}")
            return None
            
    def _save_data(self, df):
        """Save data to CSV file."""
        os.makedirs('data/raw', exist_ok=True)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/raw/crypto_data_{current_time}.csv'
        df.to_csv(filename, index=False)
        
    def run(self, interval_seconds=60):
        """Run continuous data ingestion."""
        logger.info("Starting continuous data ingestion...")
        while True:
            self.fetch_real_time_data()
            time.sleep(interval_seconds)

if __name__ == "__main__":
    ingestion = CryptoDataIngestion()
    ingestion.run() 