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
