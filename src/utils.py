import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

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