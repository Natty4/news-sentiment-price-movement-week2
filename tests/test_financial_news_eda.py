import pandas as pd
import pytest
from src.utils import FinancialNewsEDA


@pytest.fixture
def mock_news_data():
    data = {
        "date": ["2024-06-01", "2024-06-02", "2024-06-03"],
        "headline": [
            "Stock markets rally on strong earnings.",
            "Investors worry about inflation trends.",
            "Benzinga releases bullish Apple forecast."
        ],
        "publisher": [
            "Benzinga Newsdesk",
            "Reuters",
            "Benzinga Insights"
        ]
    }
    return pd.DataFrame(data)


def test_preprocess_columns(mock_news_data):
    eda = FinancialNewsEDA(mock_news_data)
    assert "headline_length" in eda.df.columns
    assert "word_count" in eda.df.columns
    assert "hour" in eda.df.columns
    assert "weekday" in eda.df.columns
    assert "publisher_grouped" in eda.df.columns
    assert "is_benzinga" in eda.df.columns
    assert eda.df["is_benzinga"].isin(["Benzinga", "Other Publishers"]).all()


def test_sentiment_computation(mock_news_data):
    eda = FinancialNewsEDA(mock_news_data)
    eda.compute_sentiment()
    assert "sentiment" in eda.df.columns
    assert pd.api.types.is_float_dtype(eda.df["sentiment"])


def test_keyword_extraction(mock_news_data):
    eda = FinancialNewsEDA(mock_news_data)
    benzinga_keywords = eda.get_top_keywords("Benzinga", n=5)
    other_keywords = eda.get_top_keywords("Other Publishers", n=5)

    assert isinstance(benzinga_keywords, list)
    assert isinstance(benzinga_keywords[0], tuple)
    assert len(benzinga_keywords) <= 5
    assert len(other_keywords) <= 5