"""Live prediction CLI."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

from src import utils, data_fetch, indicators, sentiment, preprocessing


def predict_single(ticker: str, config_path: str = "configs/config.yaml"):
    """Predict for a single ticker."""
    config = utils.load_config(config_path)
    
    # Load model and scaler
    model_path = config["model"]["save_path"]
    scaler_path = config["model"]["scaler_path"]
    
    model = xgb.Booster(model_file=model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Fetch recent data
    prices = data_fetch.fetch_price_data(
        [ticker], "2024-01-01", "2024-12-31",
        config["data"]["raw_dir"]
    )[ticker]
    
    prices = indicators.add_all_indicators(prices, config)
    news = data_fetch.fetch_news(ticker, ticker, "2024-01-01", "2024-12-31",
                                 save_dir=config["data"]["raw_dir"])
    sentiment_df = sentiment.get_sentiment_features(news, config["sentiment"]["engine"], config)
    
    # Build features for last day
    prices.index = pd.to_datetime(prices.index)
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    merged = prices.join(sentiment_df, how="left")
    merged["sentiment_positive"].fillna(config["sentiment"]["neutral_default"], inplace=True)
    merged["sentiment_negative"].fillna(config["sentiment"]["neutral_default"], inplace=True)
    merged["sentiment_neutral"].fillna(config["sentiment"]["neutral_default"], inplace=True)
    merged["num_articles"].fillna(0, inplace=True)
    merged.dropna(inplace=True)
    
    feature_cols = [col for col in merged.columns
                   if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    X = merged[feature_cols].iloc[-1:].values
    X_scaled = scaler.transform(X)
    
    pred_proba = model.predict(X_scaled)[0]
    pred_label = 1 if pred_proba > 0.5 else 0
    confidence = max(pred_proba, 1 - pred_proba) * 100
    
    # Print output
    print("\n" + "=" * 50)
    print(f"  Stock Trend Prediction — {ticker}")
    print(f"  Date: {merged.index[-1].strftime('%Y-%m-%d')}")
    print("=" * 50)
    print(f"  Prediction:  {'📈 UP' if pred_label == 1 else '📉 DOWN'}")
    print(f"  Confidence:  {confidence:.1f}%")
    print("\n  ⚠️  Research model. Not financial advice.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker")
    parser.add_argument("--config", default="configs/config.yaml", help="Config path")
    args = parser.parse_args()
    
    import pandas as pd
    predict_single(args.ticker, args.config)
