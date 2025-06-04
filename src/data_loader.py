from pathlib import Path
import pandas as pd

class StockDataLoader:
    """
    Task-2: Load Historical Stock Data
    """
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load(self, ticker: str) -> pd.DataFrame:
        """
        Load stock data CSV by ticker.

        Parameters:
        - ticker (str): e.g., "AAPL"

        Returns:
        - pd.DataFrame: Stock data with Date as datetime index
        """
        file_path = self.data_dir / f"{ticker}_historical_data.csv"

        try:
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date').sort_index()
            return df

        except FileNotFoundError:
            print(f"[ERROR] File not found: {file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load data for {ticker}: {e}")

        return pd.DataFrame()
    
    
class NewsDataLoader:
    """
    Task-2: Load Financial News Data
    """
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load(self, filename: str) -> pd.DataFrame:
        """
        Load financial news data CSV.

        Parameters:
        - filename (str): e.g., "raw_analyst_ratings.csv"

        Returns:
        - pd.DataFrame: News data with a parsed datetime and date column
        """
        file_path = self.data_dir / filename

        try:
            df = pd.read_csv(file_path)

            # Try datetime → then date → fallback to auto-coercion
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
                df['date'] = df['datetime'].dt.date
                df.set_index('datetime', inplace=True)

            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.set_index('date', inplace=True)

            else:
                # Attempt to guess a date column
                print("[WARNING] No 'datetime' or 'date' column found. Attempting auto-detection.")
                df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')  # Assume first column
                df.set_index('date', inplace=True)

            df.sort_index(inplace=True)
            return df

        except FileNotFoundError:
            print(f"[ERROR] File not found: {file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load news data: {e}")

        return pd.DataFrame()