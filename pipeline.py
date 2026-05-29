"""End-to-end pipeline orchestration."""

import os

# Suppress TensorFlow warnings early
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src import utils, data_fetch, indicators, sentiment, preprocessing, train, explainability

load_dotenv()
logger = utils.get_logger(__name__)


def main():
    """Run complete pipeline."""
    config = utils.load_config("configs/config.yaml")
    utils.setup_dirs(config)
    utils.set_seed(config["model"]["random_state"])
    
    logger.info("=" * 60)
    logger.info("STOCK PREDICTOR PIPELINE")
    logger.info("=" * 60)
    
    # Fetch prices
    logger.info("\n[1/8] Fetching price data...")
    prices = data_fetch.fetch_price_data(
        config["data"]["tickers"],
        config["data"]["start_date"],
        config["data"]["end_date"],
        config["data"]["raw_dir"]
    )
    
    # Fetch news
    logger.info("\n[2/8] Fetching news...")
    api_key = os.getenv("NEWS_API_KEY", config["news"]["api_key"]) or None
    news_all = {}
    for ticker in config["data"]["tickers"]:
        news_all[ticker] = data_fetch.fetch_news(
            ticker, ticker, config["data"]["start_date"],
            config["data"]["end_date"], api_key, config["data"]["raw_dir"]
        )
    
    # Add indicators
    logger.info("\n[3/8] Computing technical indicators...")
    for ticker in prices:
        prices[ticker] = indicators.add_all_indicators(prices[ticker], config)
    
    # Get sentiment
    logger.info("\n[4/8] Analyzing sentiment...")
    sentiment_all = {}
    for ticker in news_all:
        sentiment_all[ticker] = sentiment.get_sentiment_features(
            news_all[ticker], config["sentiment"]["engine"], config
        )
    
    # Build datasets
    logger.info("\n[5/8] Building datasets...")
    datasets = []
    for ticker in config["data"]["tickers"]:
        if ticker in prices and ticker in sentiment_all:
            df = preprocessing.build_dataset(
                ticker, prices[ticker], sentiment_all[ticker], config
            )
            datasets.append(df)
    
    combined_df = pd.concat(datasets, ignore_index=False)
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    
    # Feature columns
    feature_cols = [col for col in combined_df.columns
                   if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    feature_cols = [col for col in feature_cols if not pd.isna(combined_df[col]).all()]
    
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")
    
    # Split and scale
    logger.info("\n[6/8] Splitting and scaling...")
    X_train, X_test, y_train, y_test, feat_names = preprocessing.split_and_scale(
        combined_df, feature_cols, config["model"]["test_size"],
        config["model"]["random_state"], config["model"]["scaler_path"]
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Train
    logger.info("\n[7/8] Training XGBoost...")
    model, metrics = train.train_xgboost(X_train, y_train, X_test, y_test, config)
    
    # Explainability
    logger.info("\n[8/8] Generating SHAP reports...")
    shap_importance = explainability.generate_shap_report(
        model, X_test, feat_names, config["outputs"]["plots_dir"]
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    top_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    logger.info(f"Top 3 SHAP features: {top_features}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
