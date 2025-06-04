# 📊 Financial News & Stock Technical Analysis

A professional Python project for analyzing financial news sentiment and performing technical analysis on stock data. The repository is designed for clarity, reproducibility, and extensibility — suitable for both research and production-ready pipelines.

---

## 🧭 Project Structure

```
├── notebooks/      # Jupyter Notebooks for EDA, technical indicators, and correlation analysis
├── src/            # Core Python modules (data loaders, indicators, sentiment analysis, utils)
├── data/           # Raw and processed datasets (CSV format)
├── scripts/        # (Optional) CLI or automation scripts
├── tests/          # Unit tests for core functionality
├── requirements.txt
└── README.md
```

---

## 🚀 Main Features

* **Task 1:** Financial news EDA, text preprocessing, and sentiment analysis using VADER
* **Task 2:** Stock technical indicators (SMA, RSI, MACD) and performance metrics visualization
* **Task 3:** Correlation analysis between daily news sentiment and daily stock returns (e.g., AAPL)
* Clean, modular, object-oriented design for reusable workflows
* Visual insights powered by `matplotlib` and `seaborn`
* Unit-tested core logic with continuous integration via GitHub Actions

---

## 🛠 Getting Started

1. Clone the repository and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Place your data files into the `data/` directory as required by notebooks and modules.
3. Run the Jupyter Notebooks in the `notebooks/` folder for guided, task-based analysis.

---

## ✅ Testing

Run all tests using `pytest`:

```bash
pytest tests/
```

---

## 📦 Requirements

* Python 3.10+
* See `requirements.txt` for full dependency list

---

© 2025 10Academy — For learning and research purposes only.
