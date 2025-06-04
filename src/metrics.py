import pandas as pd
import numpy as np


class PerformanceMetrics:
    """
    Class for computing common financial metrics on stock data.
    """

    @staticmethod
    def calculate_daily_returns(df: pd.DataFrame) -> pd.Series:
        if "Close" not in df.columns:
            raise KeyError("Missing 'Close' column in input DataFrame.")
        return df["Close"].pct_change().dropna()

    @staticmethod
    def calculate_volatility(returns: pd.Series, trading_days: int = 252) -> float:
        if not isinstance(returns, pd.Series):
            raise TypeError("Input returns must be a pandas Series.")
        return returns.std() * np.sqrt(trading_days)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.01, trading_days: int = 252
    ) -> float:
        if not isinstance(returns, pd.Series):
            raise TypeError("Input returns must be a pandas Series.")
        excess_returns = returns - (risk_free_rate / trading_days)
        return excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days)

    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        if not isinstance(returns, pd.Series):
            raise TypeError("Input returns must be a pandas Series.")
        return (1 + returns).cumprod()

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        if not isinstance(cumulative_returns, pd.Series):
            raise TypeError("Input cumulative_returns must be a pandas Series.")
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def calculate_rolling_volatility(df: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        Calculate rolling volatility over a specified window.
        """
        returns = PerformanceMetrics.calculate_daily_returns(df)
        rolling_volatility = returns.rolling(window).std() * np.sqrt(252)
        return rolling_volatility

    @staticmethod
    def calculate_rolling_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.01, window: int = 30) -> pd.Series:
        """
        Calculate rolling Sharpe ratio over a specified window.
        """
        returns = PerformanceMetrics.calculate_daily_returns(df)
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        excess_returns = rolling_mean - (risk_free_rate / 252)
        rolling_sharpe = excess_returns / rolling_std * np.sqrt(252)
        return rolling_sharpe

    @staticmethod
    def summarize_metrics(df: pd.DataFrame, risk_free_rate: float = 0.01) -> dict:
        """
        Summarizes key financial metrics from a stock price DataFrame.
        Returns a dictionary with:
            - total_return
            - volatility
            - sharpe_ratio
            - max_drawdown
        """
        try:
            # Correct: Compute daily returns first
            returns = PerformanceMetrics.calculate_daily_returns(df)

            # Correct: Compute cumulative returns from daily returns
            cumulative = PerformanceMetrics.calculate_cumulative_returns(returns)

            summary = {
                "total_return": cumulative.iloc[-1] - 1,
                "volatility": PerformanceMetrics.calculate_volatility(returns),
                "sharpe_ratio": PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate),
                "max_drawdown": PerformanceMetrics.calculate_max_drawdown(cumulative)
            }
            return summary

        except Exception as e:
            print(f"[ERROR] Failed to summarize metrics: {e}")
            return {}