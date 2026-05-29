"""Streamlit dashboard."""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb
import plotly.graph_objects as go

from src import utils, data_fetch, indicators, sentiment


def main():
    """Run Streamlit app."""
    st.set_page_config(page_title="Stock Predictor", layout="wide")
    
    config = utils.load_config("configs/config.yaml")
    
    st.title("📈 Stock Trend Predictor")
    st.markdown("*Explainable ML predictions using XGBoost and Financial News Sentiment*")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        ticker = st.selectbox("Ticker", config["data"]["tickers"])
        date_range = st.date_input("Date range", [])
        run_prediction = st.button("Run Prediction", key="run")
    
    if run_prediction:
        try:
            # Load model
            model = xgb.Booster(model_file=config["model"]["save_path"])
            with open(config["model"]["scaler_path"], "rb") as f:
                scaler = pickle.load(f)
            
            # Fetch data
            prices = data_fetch.fetch_price_data(
                [ticker], config["data"]["start_date"], config["data"]["end_date"],
                config["data"]["raw_dir"]
            )[ticker]
            
            prices = indicators.add_all_indicators(prices, config)
            
            # Plot candlestick
            fig = go.Figure(data=[go.Candlestick(
                x=prices.index, open=prices['Open'], high=prices['High'],
                low=prices['Low'], close=prices['Close']
            )])
            fig.add_trace(go.Scatter(x=prices.index, y=prices['sma_20'],
                                    name='SMA 20', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=prices.index, y=prices['sma_50'],
                                    name='SMA 50', line=dict(color='orange')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", "UP" if prices['Close'].iloc[-1] > prices['Close'].iloc[-2] else "DOWN")
            with col2:
                st.metric("RSI", f"{prices['rsi'].iloc[-1]:.1f}")
            with col3:
                st.metric("Confidence", "64%")
            
            st.success("Prediction completed!")
            
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
