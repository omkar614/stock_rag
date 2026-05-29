"""Sentiment analysis using FinBERT or VADER."""

import pandas as pd

from src import utils

logger = utils.get_logger(__name__)
error_logger = utils.get_error_logger()


class SentimentAnalyzer:
    """Sentiment analyzer using FinBERT or VADER."""
    
    def __init__(self, engine: str = "finbert", model_name: str = "ProsusAI/finbert"):
        """Initialize analyzer."""
        self.engine = engine
        self.model_name = model_name
        self.vader = None
        
        if engine == "finbert":
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.pipe = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            except Exception as e:
                error_msg = f"FinBERT init failed: {type(e).__name__}: {e}. Falling back to VADER."
                print(error_msg)
                error_logger.warning(error_msg)
                self.engine = "vader"
        
        if self.engine == "vader":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
    
    def score_texts(self, texts: list[str]) -> list[dict]:
        """Score multiple texts."""
        if self.engine == "finbert":
            results = []
            for text in texts:
                try:
                    output = self.pipe(text[:512])[0]
                    results.append({
                        "sentiment_positive": output["score"] if output["label"] == "positive" else 0.0,
                        "sentiment_negative": output["score"] if output["label"] == "negative" else 0.0,
                        "sentiment_neutral": output["score"] if output["label"] == "neutral" else 0.0,
                    })
                except:
                    results.append({"sentiment_positive": 0.33, "sentiment_negative": 0.33, "sentiment_neutral": 0.34})
            return results
        else:
            return [
                {
                    "sentiment_positive": max(0, self.vader.polarity_scores(t)["compound"]) / 2 + 0.25,
                    "sentiment_negative": max(0, -self.vader.polarity_scores(t)["compound"]) / 2 + 0.25,
                    "sentiment_neutral": 0.33,
                    "compound": self.vader.polarity_scores(t)["compound"],
                }
                for t in texts
            ]
    
    def aggregate_daily(self, articles: list[dict]) -> pd.DataFrame:
        """Aggregate sentiment by date."""
        df = pd.DataFrame(articles)
        df["date"] = pd.to_datetime(df["date"])
        
        texts = df["title"].fillna("") + " " + df["description"].fillna("")
        scores = self.score_texts(texts.tolist())
        
        for key in ["sentiment_positive", "sentiment_negative", "sentiment_neutral"]:
            df[key] = [s.get(key, 0.33) for s in scores]
        
        daily = df.groupby("date")[["sentiment_positive", "sentiment_negative", "sentiment_neutral"]].mean()
        daily["num_articles"] = df.groupby("date").size()
        
        return daily


def get_sentiment_features(articles: list[dict], engine: str, config: dict) -> pd.DataFrame:
    """Get sentiment features from articles."""
    analyzer = SentimentAnalyzer(engine=engine)
    df = analyzer.aggregate_daily(articles)
    
    # Fill missing dates with neutrals
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(date_range)
    df.fillna({
        "sentiment_positive": config["sentiment"]["neutral_default"],
        "sentiment_negative": config["sentiment"]["neutral_default"],
        "sentiment_neutral": config["sentiment"]["neutral_default"],
        "num_articles": 0,
    }, inplace=True)
    
    df.index.name = "date"
    return df
