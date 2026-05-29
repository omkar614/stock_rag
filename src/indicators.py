"""Technical indicators computation."""

import pandas as pd
import numpy as np

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    import warnings
    warnings.warn("ta library not found. Technical indicators will use simplified versions.")


def add_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add 18 technical indicators to DataFrame."""
    df = df.copy()
    
    # Price-derived indicators
    df["daily_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["co_spread"] = (df["Close"] - df["Open"]) / df["Open"]
    
    if TA_AVAILABLE:
        # Moving averages - using ta library
        for window in config["indicators"]["sma_windows"]:
            df[f"sma_{window}"] = ta.trend.sma_indicator(df["Close"], window=window)
        
        for window in config["indicators"]["ema_windows"]:
            df[f"ema_{window}"] = ta.trend.ema_indicator(df["Close"], window=window)
        
        # Momentum - using ta library
        df["rsi"] = ta.momentum.rsi(df["Close"], window=config["indicators"]["rsi_window"])
        
        macd = ta.trend.MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        
        # Volatility (Bollinger Bands) - using ta library
        bb = ta.volatility.BollingerBands(
            df["Close"],
            window=config["indicators"]["bb_window"],
            window_dev=config["indicators"]["bb_std"]
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_width"] = bb.bollinger_wband()
        
        # Volume - using ta library
        df["obv"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    else:
        # Fallback: Simple implementations using pandas
        for window in config["indicators"]["sma_windows"]:
            df[f"sma_{window}"] = df["Close"].rolling(window).mean()
        
        for window in config["indicators"]["ema_windows"]:
            df[f"ema_{window}"] = df["Close"].ewm(span=window).mean()
        
        # RSI fallback
        rsi_window = config["indicators"]["rsi_window"]
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_window).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD fallback
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands fallback
        bb_window = config["indicators"]["bb_window"]
        bb_std = config["indicators"]["bb_std"]
        df["bb_mid"] = df["Close"].rolling(bb_window).mean()
        bb_std_calc = df["Close"].rolling(bb_window).std()
        df["bb_upper"] = df["bb_mid"] + (bb_std_calc * bb_std)
        df["bb_lower"] = df["bb_mid"] - (bb_std_calc * bb_std)
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        
        # OBV fallback (simplified)
        df["obv"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    
    df["rolling_std_20"] = df["Close"].rolling(20).std()
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    
    return df
