import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from collections import Counter
    
from scipy.stats import pearsonr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class FinancialNewsEDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.analyzer = SentimentIntensityAnalyzer()
        self.preprocess()

    def preprocess(self):
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df.dropna(subset=["date"], inplace=True)
        self.df["headline_length"] = self.df["headline"].str.len()
        self.df["word_count"] = self.df["headline"].str.split().str.len()
        self.df["hour"] = self.df["date"].dt.hour
        self.df["weekday"] = self.df["date"].dt.day_name()
        self.clean_publishers()

    def clean_publishers(self):
        self.df["domain"] = self.df["publisher"].str.extract(r"@([\w\.-]+)")
        self.df["publisher_grouped"] = self.df["domain"].where(
            self.df["domain"].notna(), self.df["publisher"]
        ).str.strip().str.replace(r"[\s_]+", " ", regex=True)

        aliases = {
            "Benzinga Newsdesk": "Benzinga",
            "Benzinga Insights": "Benzinga",
            "Benzinga News Desk": "Benzinga",
            "Benzinga_Newsdesk": "Benzinga",
        }
        self.df["publisher_grouped"] = self.df["publisher_grouped"].replace(aliases)
        self.df["is_benzinga"] = self.df["publisher_grouped"].apply(
            lambda x: "Benzinga" if x == "Benzinga" else "Other Publishers"
        )

    def plot_headline_length_dist(self):
        plt.figure(figsize=(8, 4))
        sns.histplot(self.df["headline_length"], bins=30, kde=True)
        plt.title("Distribution of Headline Lengths")
        plt.xlabel("Number of Characters")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_articles_by_hour(self):
        plt.figure(figsize=(10, 4))
        sns.countplot(x="hour", data=self.df, palette="viridis")
        plt.title("Articles by Hour of Day")
        plt.xlabel("Hour")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.show()

    def plot_articles_by_weekday(self):
        plt.figure(figsize=(10, 4))
        sns.countplot(
            x="weekday",
            data=self.df,
            order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )
        plt.title("Articles by Day of Week")
        plt.xlabel("Day")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.show()

    def plot_top_publishers(self, top_n=10):
        top_groups = self.df["publisher_grouped"].value_counts().head(top_n)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_groups.values, y=top_groups.index, palette="viridis")
        plt.title(f"Top {top_n} Publisher Groups", fontsize=14)
        plt.xlabel("Number of Articles")
        plt.ylabel("Publisher Group")
        plt.tight_layout()
        plt.show()
        print(top_groups)

    def compare_headline_stats(self):
        plt.figure(figsize=(12, 5))
        sns.boxplot(data=self.df, x="is_benzinga", y="headline_length", palette="Set2")
        plt.title("Headline Length: Benzinga vs. Other Publishers")
        plt.show()

        plt.figure(figsize=(12, 5))
        sns.boxplot(data=self.df, x="is_benzinga", y="word_count", palette="Set2")
        plt.title("Word Count: Benzinga vs. Other Publishers")
        plt.show()

    def compare_publication_time(self):
        plt.figure(figsize=(12, 5))
        sns.histplot(data=self.df, x="hour", hue="is_benzinga", multiple="stack", bins=24)
        plt.title("Publishing Hour Distribution: Benzinga vs. Others")
        plt.xlabel("Hour of Day")
        plt.show()

    def get_top_keywords(self, group_value, n=15):
        subset = self.df[self.df["is_benzinga"] == group_value]
        text = " ".join(subset["headline"].dropna()).lower()
        words = re.findall(r'\b[a-z]{3,}\b', text)
        return Counter(words).most_common(n)

    def display_top_keywords(self):
        benzinga = self.get_top_keywords("Benzinga")
        others = self.get_top_keywords("Other Publishers")
        print("🔷 Benzinga Keywords:")
        for word, freq in benzinga:
            print(f"{word}: {freq}")
        print("\n🔶 Other Publisher Keywords:")
        for word, freq in others:
            print(f"{word}: {freq}")

    def compute_sentiment(self):
        self.df["sentiment"] = self.df["headline"].apply(self._get_sentiment)

    def _get_sentiment(self, text):
        if isinstance(text, str):
            return self.analyzer.polarity_scores(text)["compound"]
        return None

    def plot_sentiment_comparison(self):
        if "sentiment" not in self.df.columns:
            self.compute_sentiment()
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=self.df, x="is_benzinga", y="sentiment", palette="coolwarm")
        plt.title("Sentiment Comparison: Benzinga vs. Other Publishers")
        plt.ylabel("VADER Compound Sentiment Score")
        plt.xlabel("")
        plt.show()

    def topic_modeling(self, n_topics=5, n_top_words=7):
        tfidf = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_matrix = tfidf.fit_transform(self.df["headline"].fillna(""))
        nmf = NMF(n_components=n_topics, random_state=42)
        nmf.fit(tfidf_matrix)
        feature_names = tfidf.get_feature_names_out()

        print("Topics discovered:")
        for topic_idx, topic in enumerate(nmf.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

    def summary(self):
        print("Total articles:", len(self.df))
        print("Total unique publishers:", self.df["publisher"].nunique())
        print("Time range:", self.df["date"].min(), "to", self.df["date"].max())
        print("Missing values:\n", self.df.isna().sum())
        

class TechnicalIndicatorPlotter:
    """
    Task-2: Visualization of Stock Technical Indicators
    """
    def __init__(self, df: pd.DataFrame, ticker: str):
        self.df = df
        self.ticker = ticker

    def plot_price_and_indicators(self):
        """
        Plot Close Price + SMAs, RSI, and MACD in subplots.
        """
        try:
            plt.figure(figsize=(16, 10))

            self._plot_price_with_smas()
            self._plot_rsi()
            self._plot_macd()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"[ERROR] Plotting failed: {e}")

    def _plot_price_with_smas(self):
        plt.subplot(3, 1, 1)
        plt.plot(self.df.index, self.df['Close'], label='Close', color='black')
        plt.plot(self.df.index, self.df['SMA_20'], label='SMA 20', color='blue')
        plt.plot(self.df.index, self.df['SMA_50'], label='SMA 50', color='orange')
        plt.title(f'{self.ticker} - Close Price & Moving Averages')
        plt.legend()

    def _plot_rsi(self):
        plt.subplot(3, 1, 2)
        plt.plot(self.df.index, self.df['RSI_14'], label='RSI 14', color='green')
        plt.axhline(70, color='red', linestyle='--')
        plt.axhline(30, color='red', linestyle='--')
        plt.title('Relative Strength Index (RSI)')
        plt.legend()

    def _plot_macd(self):
        plt.subplot(3, 1, 3)
        plt.plot(self.df.index, self.df['MACD'], label='MACD', color='purple')
        plt.plot(self.df.index, self.df['MACD_Signal'], label='Signal Line', color='gray')
        plt.bar(self.df.index, self.df['MACD_Hist'], label='Histogram', color='lightcoral')
        plt.title('MACD')
        plt.legend()
        
    
    def plot_cumulative_returns(cumulative_returns: pd.Series, stock_name: str = "Stock"):
        """
        Plots cumulative returns.
        """
        plt.figure(figsize=(14, 5))
        plt.plot(cumulative_returns, label="Cumulative Returns", color="green")
        plt.title(f"{stock_name} Cumulative Returns Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.tight_layout()
        plt.show()


class FinancialNewsCorrelation:

    def preprocess_stock_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and preprocess stock data to compute daily returns.
        
        Parameters:
            df (pd.DataFrame): Must contain 'Date' as index or column and 'Close' price.
        
        Returns:
            pd.DataFrame: DataFrame with 'date' and 'daily_return' columns.
        """
        try:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.dropna(subset=['Date'], inplace=True)
                df.set_index('Date', inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Stock DataFrame must have a datetime index or 'Date' column.")

            if 'Close' not in df.columns:
                raise KeyError("Missing 'Close' column in stock data.")

            df = df.sort_index()
            df['daily_return'] = df['Close'].pct_change()
            df.dropna(subset=['daily_return'], inplace=True)

            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'date'}, inplace=True)

            return df[['date', 'daily_return']]

        except Exception as e:
            print(f"[ERROR] Stock preprocessing failed: {e}")
            return pd.DataFrame()
        
    def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate average daily sentiment from news headlines.

        Parameters:
            df (pd.DataFrame): DataFrame containing 'date' (datetime) and 'sentiment' columns.

        Returns:
            pd.DataFrame: DataFrame with 'date' and 'avg_sentiment' columns.
        """
        try:
            if 'date' not in df.columns or 'sentiment' not in df.columns:
                raise KeyError("The DataFrame must contain 'date' and 'sentiment' columns.")
            
            df = df.copy()
            df['date'] = pd.to_datetime(df['date']).dt.date  # Remove time component
            sentiment_daily = df.groupby('date')['sentiment'].mean().reset_index()
            sentiment_daily.rename(columns={'sentiment': 'avg_sentiment'}, inplace=True)
            sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])  # Normalize back to datetime
            
            return sentiment_daily

        except Exception as e:
            print(f"[ERROR] Sentiment aggregation failed: {e}")
            return pd.DataFrame()


    def merge_sentiment_with_returns(sentiment_df: pd.DataFrame, stock_returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge average daily sentiment scores with daily stock returns.

        Parameters:
            sentiment_df (pd.DataFrame): DataFrame with 'date' and 'avg_sentiment'.
            stock_returns_df (pd.DataFrame): DataFrame with 'date' and 'daily_return'.

        Returns:
            pd.DataFrame: Merged DataFrame with columns ['date', 'avg_sentiment', 'daily_return'].
        """
        try:
            required_sent_cols = {'date', 'avg_sentiment'}
            required_stock_cols = {'date', 'daily_return'}

            if not required_sent_cols.issubset(sentiment_df.columns):
                raise KeyError(f"Sentiment DataFrame must contain columns: {required_sent_cols}")
            if not required_stock_cols.issubset(stock_returns_df.columns):
                raise KeyError(f"Stock returns DataFrame must contain columns: {required_stock_cols}")

            sentiment_df = sentiment_df.copy()
            stock_returns_df = stock_returns_df.copy()

            merged_df = pd.merge(sentiment_df, stock_returns_df, on='date', how='inner')

            if merged_df.empty:
                print("[WARNING] No overlapping dates found between sentiment and stock returns.")
            else:
                print(f"[INFO] Successfully merged {len(merged_df)} records on common dates.")

            return merged_df.dropna()

        except Exception as e:
            print(f"[ERROR] Merging failed: {e}")
            return pd.DataFrame()
        

    def compute_sentiment_return_correlation(merged_df: pd.DataFrame) -> dict:
        """
        Compute Pearson correlation between average sentiment and stock returns.

        Parameters:
            merged_df (pd.DataFrame): DataFrame containing 'avg_sentiment' and 'daily_return' columns.

        Returns:
            dict: Dictionary with 'correlation', 'p_value', and optional message.
        """
        try:
            if merged_df.empty:
                raise ValueError("Input DataFrame is empty. Cannot compute correlation.")

            if 'avg_sentiment' not in merged_df.columns or 'daily_return' not in merged_df.columns:
                raise KeyError("Merged DataFrame must contain 'avg_sentiment' and 'daily_return' columns.")

            x = merged_df['avg_sentiment']
            y = merged_df['daily_return']

            # Drop NaNs
            valid_data = merged_df.dropna(subset=['avg_sentiment', 'daily_return'])
            if len(valid_data) < 2:
                raise ValueError("Not enough valid data points to compute correlation.")

            correlation, p_value = pearsonr(valid_data['avg_sentiment'], valid_data['daily_return'])

            result = {
                "correlation": round(correlation, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05
            }

            msg = f"[INFO] Correlation: {result['correlation']}, p-value: {result['p_value']} "
            msg += "(Statistically significant)" if result['significant'] else "(Not statistically significant)"
            print(msg)

            return result

        except Exception as e:
            print(f"[ERROR] Correlation analysis failed: {e}")
            return {"correlation": None, "p_value": None, "significant": None}
        
        
    def compute_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily percentage returns from 'Adj Close' column.

        Parameters:
        - df (pd.DataFrame): DataFrame with 'Adj Close' and 'date' column or index.

        Returns:
        - pd.DataFrame: With 'daily_return' column
        """
        df = df.copy()
        df["daily_return"] = df["Adj Close"].pct_change()
        return df.dropna(subset=["daily_return"])


    def plot_sentiment_vs_returns(df: pd.DataFrame, sentiment_col: str, return_col: str, title: str = ""):
        """
        Create a scatter plot with regression line between sentiment and returns.

        Parameters:
        - df (pd.DataFrame): DataFrame containing sentiment and return columns
        - sentiment_col (str): Column name for sentiment score
        - return_col (str): Column name for daily return
        - title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x=sentiment_col, y=return_col, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.xlabel("Average Sentiment")
        plt.ylabel("Daily Return")
        plt.title(title or "Sentiment vs. Daily Return")
        plt.grid(True)
        plt.tight_layout()
        plt.show()