"""Data preprocessing: merge, split, scale."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import utils

logger = utils.get_logger(__name__)


def build_dataset(
    ticker: str,
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Merge price, indicators, sentiment; add target."""
    df = price_df.copy()
    df.index = pd.to_datetime(df.index)
    
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    df = df.join(sentiment_df, how="left")
    
    df["sentiment_positive"].fillna(config["sentiment"]["neutral_default"], inplace=True)
    df["sentiment_negative"].fillna(config["sentiment"]["neutral_default"], inplace=True)
    df["sentiment_neutral"].fillna(config["sentiment"]["neutral_default"], inplace=True)
    df["num_articles"].fillna(0, inplace=True)
    
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1]
    df.dropna(inplace=True)
    
    save_path = Path(config["data"]["processed_dir"]) / f"{ticker}_dataset.csv"
    df.to_csv(save_path)
    logger.info(f"Saved dataset for {ticker} to {save_path}")
    
    return df


def split_and_scale(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float,
    random_state: int,
    scaler_path: str | Path,
) -> tuple:
    """Split chronologically, scale with MinMaxScaler."""
    split_idx = int(len(df) * (1 - test_size))
    
    X_train = df[feature_cols].iloc[:split_idx].values
    X_test = df[feature_cols].iloc[split_idx:].values
    y_train = df["target"].iloc[:split_idx].values
    y_test = df["target"].iloc[split_idx:].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
