import pandas as pd
from src.utils import FinancialNewsEDA  # Replace with actual import path

def test_financial_news_eda_basic_preprocessing():
    # Sample input data
    data = {
        "headline": [
            "Stocks hit 52-week highs",
            "Analyst upgrades Tesla",
            "Benzinga reports on Apple earnings"
        ],
        "publisher": [
            "Benzinga Newsdesk",
            "Lisa Levin",
            "Benzinga_Newsdesk"
        ],
        "date": [
            "2021-06-01 10:00:00",
            "2021-06-02 14:30:00",
            "2021-06-03 09:15:00"
        ]
    }

    df = pd.DataFrame(data)
    eda = FinancialNewsEDA(df)

    # Check columns are added
    expected_cols = ["headline_length", "word_count", "hour", "weekday", "publisher_grouped", "is_benzinga"]
    for col in expected_cols:
        assert col in eda.df.columns, f"{col} column not found after preprocessing"

    # Check word count
    assert eda.df.loc[0, "word_count"] == 4
    assert eda.df.loc[1, "hour"] == 14
    assert eda.df.loc[2, "weekday"] == "Thursday"

    # Benzinga label
    assert eda.df["is_benzinga"].value_counts().to_dict() == {
        "Benzinga": 2,
        "Other Publishers": 1
    }