# Stock Trend Predictor

Explainable multi-modal stock prediction using XGBoost, technical indicators, and financial sentiment analysis.

**Now with Indian stocks!** Supports both US (NASDAQ) and Indian (NSE) equities.

## Supported Stocks

### US Stocks

- AAPL (Apple)
- TSLA (Tesla)
- MSFT (Microsoft)

### Indian Stocks (NSE)

- RELIANCE.NS (Reliance Industries)
- TCS.NS (Tata Consultancy Services)
- INFY.NS (Infosys)
- HDFCBANK.NS (HDFC Bank)
- WIPRO.NS (Wipro)

## Setup

```bash
pip install -r requirements.txt
python generate_all.py
```

## Run

```bash
# Full pipeline
python pipeline.py

# Single prediction
python predict.py --ticker AAPL

# Streamlit dashboard
streamlit run app.py
```

## Model

- **XGBoost** with 20+ technical indicators
- **FinBERT** sentiment from financial news
- **Optuna** hyperparameter tuning with **TimeSeriesSplit**
- **SHAP** explainability

Expected accuracy: 54-65%
