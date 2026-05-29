"""Data fetching module for prices and news."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src import utils

logger = utils.get_logger(__name__)
error_logger = utils.get_error_logger()


def fetch_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    save_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV price data for tickers using yfinance."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    price_data = {}
    for ticker in tickers:
        cache_file = save_dir / f"{ticker}_prices.csv"
        
        if cache_file.exists():
            logger.info(f"Loading cached prices for {ticker}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            logger.info(f"Downloading prices for {ticker}")
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue
                df.to_csv(cache_file)
                logger.info(f"Saved prices for {ticker}")
            except Exception as e:
                error_msg = f"Failed to fetch {ticker}: {type(e).__name__}: {e}"
                logger.error(error_msg)
                error_logger.error(error_msg)
                raise
        
        price_data[ticker] = df
    
    return price_data


def _mock_news(ticker: str, start_date: str, end_date: str) -> list[dict]:
    """Generate synthetic news headlines."""
    positive = [
        f"{ticker} beats quarterly earnings",
        f"Analysts raise {ticker} price target",
        f"{ticker} launches new product",
    ]
    negative = [
        f"{ticker} misses earnings",
        f"Regulatory concerns on {ticker}",
        f"{ticker} faces supply chain issues",
    ]
    neutral = [
        f"{ticker} reports results",
        f"Analyst covers {ticker}",
        f"{ticker} joins conference",
    ]
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    articles = []
    current = start
    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue
        
        for _ in range(random.randint(0, 3)):
            templates = random.choice([positive, negative, neutral])
            title = random.choice(templates)
            articles.append({
                "ticker": ticker,
                "date": current.strftime("%Y-%m-%d"),
                "title": title,
                "description": f"Financial news for {ticker}",
            })
        
        current += timedelta(days=1)
    
    return articles


def fetch_news(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
    api_key: str | None = None,
    save_dir: str | Path = "data/raw",
) -> list[dict]:
    """Fetch news articles from NewsAPI or mock fallback."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = save_dir / f"{ticker}_news.json"
    if cache_file.exists():
        logger.info(f"Loading cached news for {ticker}")
        with open(cache_file) as f:
            return json.load(f)
    
    articles = None
    if api_key:
        try:
            logger.info(f"Fetching news from NewsAPI for {ticker}")
            from newsapi import NewsApiClient
            client = NewsApiClient(api_key=api_key)
            response = client.get_everything(
                q=company_name, from_param=start_date, to=end_date,
                language="en", sort_by="publishedAt"
            )
            articles = [
                {
                    "ticker": ticker,
                    "date": article["publishedAt"][:10],
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                }
                for article in response.get("articles", [])
            ]
            logger.info(f"Found {len(articles)} articles for {ticker}")
        except Exception as e:
            logger.warning(f"NewsAPI failed: {e}. Using mock.")
            articles = None
    
    if articles is None:
        logger.info(f"Using mock news for {ticker}")
        articles = _mock_news(ticker, start_date, end_date)
    
    with open(cache_file, "w") as f:
        json.dump(articles, f, indent=2)
    
    return articles
