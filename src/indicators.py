# TA-Lib based technical indicators
import pandas as pd
import talib

class TechnicalIndicatorCalculator:
    """
    Task-2: Calculate technical indicators using TA-Lib
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a copy of the stock DataFrame.
        """
        self.df = df.copy()

    def add_indicators(self) -> pd.DataFrame:
        """
        Add SMA, RSI, MACD to the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new indicator columns.
        """
        try:
            self._add_sma()
            self._add_rsi()
            self._add_macd()
        except Exception as e:
            print(f"[ERROR] Failed to calculate indicators: {e}")
        return self.df

    def _add_sma(self):
        self.df['SMA_20'] = talib.SMA(self.df['Close'], timeperiod=20)
        self.df['SMA_50'] = talib.SMA(self.df['Close'], timeperiod=50)

    def _add_rsi(self):
        self.df['RSI_14'] = talib.RSI(self.df['Close'], timeperiod=14)

    def _add_macd(self):
        macd, macdsignal, macdhist = talib.MACD(
            self.df['Close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = macdsignal
        self.df['MACD_Hist'] = macdhist