# CLAUDE.md
# Project Rules for Claude — Stock Trend Predictor

This file tells Claude exactly how to behave when working on this project.
Place this file in the project root. Claude reads it automatically.

---

## Project Identity

**Name:** Explainable Multi-Modal Stock Trend Prediction using XGBoost and Financial News Sentiment
**Type:** ML research project (academic/demo)
**Language:** Python 3.10+
**Primary model:** XGBoost with Optuna hyperparameter tuning
**Explainability:** SHAP TreeExplainer
**Sentiment:** FinBERT (primary), VADER (fallback)

---

## Absolute Rules

### Never do these:

1. **Never use random KFold on time series data.** Always use `TimeSeriesSplit`. Financial data has temporal order — random splits cause lookahead bias and inflate accuracy.

2. **Never hardcode any value.** All tickers, dates, paths, hyperparameters, API keys go in `configs/config.yaml`. Load with `utils.load_config()`.

3. **Never write incomplete functions.** No `pass`, no `# TODO`, no `raise NotImplementedError`. Every function must be fully implemented.

4. **Never fake metrics.** If accuracy is 55%, print 55%. Do not round up, do not adjust thresholds to inflate numbers. Realistic range is 54–65% for stock direction prediction.

5. **Never use deprecated XGBoost parameters.** `use_label_encoder` was removed in XGBoost 1.6+. Wrap in try/except or use `enable_categorical=False` instead.

6. **Never skip Optuna tuning.** It is a core feature, not optional. Implement the full `objective()` function with TimeSeriesSplit inside.

7. **Never use string path concatenation.** Use `pathlib.Path` everywhere. `Path(config["data"]["raw_dir"]) / f"{ticker}_prices.csv"` not `"data/raw/" + ticker + "_prices.csv"`.

8. **Never put loose code outside functions** (except `if __name__ == "__main__"` blocks). No notebook-style cells.

---

## Code Style Rules

### Structure
- Every `.py` file starts with a module-level docstring
- Every function has a docstring: one-line summary + Args + Returns
- Type hints on every function signature
- Imports grouped: stdlib → third-party → local, separated by blank lines

### Example of correct function style:
```python
def fetch_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    save_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV price data for a list of tickers using yfinance.

    Args:
        tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        save_dir: Directory to cache downloaded CSV files

    Returns:
        Dictionary mapping ticker symbol to DataFrame with OHLCV columns

    Raises:
        ValueError: If ticker returns empty data from yfinance
    """
```

### Logging
- Use `logger = utils.get_logger(__name__)` at module level
- Use `logger.info()`, `logger.warning()`, `logger.error()` — not `print()`
- Exception: `predict.py` output uses `print()` for clean CLI formatting

### Error handling
```python
# Correct
try:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        logger.warning(f"No data returned for {ticker}")
        return None
except Exception as e:
    logger.error(f"Failed to fetch {ticker}: {e}")
    raise
```

---

## Architecture Decisions — Defend These in Viva

### Why XGBoost over LSTM?

> "LSTMs require large sequential datasets and are highly sensitive to hyperparameters and noise. Financial time series is inherently noisy with low signal-to-noise ratio. XGBoost is more robust on structured tabular data, converges faster, and provides native feature importance — critical for explainability. SHAP integration is also seamless with tree-based models."

### Why Optuna over GridSearch?

> "GridSearch is exhaustive and scales exponentially with parameters — O(n^k) where k is parameter count. Optuna uses Tree-structured Parzen Estimator (TPE), a Bayesian optimization method that learns from previous trials. For our 9-parameter search space, GridSearch would require thousands of evaluations. Optuna achieves comparable or better results in 50 trials."

### Why TimeSeriesSplit over KFold?

> "Standard KFold randomly shuffles data, allowing future data to appear in training sets. This creates lookahead bias — the model appears to predict the future but is actually seeing it. TimeSeriesSplit preserves temporal order: each fold's validation set is strictly after its training set, matching real-world deployment conditions."

### Why FinBERT over generic BERT?

> "FinBERT is pre-trained on financial corpora — earnings calls, analyst reports, financial news. It understands domain-specific language: 'beats estimates', 'margin compression', 'guidance raised'. Generic BERT treats these as ordinary phrases. On financial sentiment benchmarks, FinBERT outperforms generic models by 10–15% F1."

### Why SHAP for explainability?

> "SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory. Unlike simple feature importance which measures global frequency of feature use, SHAP computes each feature's marginal contribution to each individual prediction. This gives both global importance (which features matter most overall) and local explanations (why did the model predict UP for this specific day)."

---

## Hyperparameter Tuning — Exact Behavior

### Optuna objective function — non-negotiable:

```python
def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """Optuna objective: maximize mean F1 across TimeSeriesSplit folds."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 2.0),
        "tree_method": "hist",
        "random_state": 42,
    }
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for tr_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        m = XGBClassifier(**params, eval_metric="logloss")
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              early_stopping_rounds=30, verbose=False)
        scores.append(f1_score(y_val, m.predict(X_val)))
    return float(np.mean(scores))
```

### What each XGBoost parameter does — comment in code:

| Parameter | Role | Financial Context |
|-----------|------|-------------------|
| `max_depth` | Tree complexity | Keep 4–7. Deep trees memorize noise in financial data |
| `learning_rate` | Step size per tree | Lower (0.01–0.05) + more trees = better generalization |
| `n_estimators` | Number of trees | Let early stopping find optimal. More is fine with low LR |
| `subsample` | Row sampling per tree | <1.0 reduces overfit. Financial data has many redundant patterns |
| `colsample_bytree` | Feature sampling per tree | Prevents RSI/MACD from dominating every single tree |
| `min_child_weight` | Minimum node weight | Higher = more conservative. Avoids splits on tiny samples |
| `gamma` | Min loss for a split | Regularization. Higher = fewer splits = simpler trees |
| `reg_alpha` | L1 regularization | Drives some feature weights to zero. Useful for feature selection |
| `reg_lambda` | L2 regularization | Shrinks all weights. More stable than L1 alone |
| `scale_pos_weight` | Class imbalance weight | Set to neg/pos ratio if market is trending strongly one direction |

---

## Feature Engineering Rules

### Technical indicators — compute in this order:
1. Price-derived: daily_return, log_return, hl_spread, co_spread
2. Moving averages: SMA20, SMA50, EMA12, EMA26
3. Momentum: RSI, MACD, MACD_signal, MACD_hist
4. Volatility: Bollinger Bands (upper, lower, mid, width), rolling_std_20
5. Volume: volume_ratio (vs 20-day avg), OBV

### Target definition — exact:
```python
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.iloc[:-1]  # drop last row — no target available
```

### Scaling — exact:
```python
# Fit scaler on TRAIN only. Never fit on test data. Never fit on full data.
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only, not fit_transform
```

---

## Sentiment Rules

### FinBERT output — label mapping:
FinBERT `ProsusAI/finbert` outputs labels: `positive`, `negative`, `neutral`
Map probabilities exactly as:
```python
sentiment = {
    "sentiment_positive": float(probs[label_map["positive"]]),
    "sentiment_negative": float(probs[label_map["negative"]]),
    "sentiment_neutral":  float(probs[label_map["neutral"]]),
}
```

### Missing news days:
```python
# When no articles exist for a date, use neutral defaults
defaults = {
    "sentiment_positive": 0.33,
    "sentiment_negative": 0.33,
    "sentiment_neutral": 0.34,
    "num_articles": 0,
}
```

### News-price alignment — critical:
- News published **before 4pm ET** on day T → feature for predicting day T+1 close
- News published **after 4pm ET** on day T → feature for predicting day T+1 close (same bucket)
- Use `date` field of article, not exact timestamp for simplicity

---

## Output Files — Exact Names

Every output file must be saved to these exact paths:

```
models/
  xgboost_model.json          ← trained model
  scaler.pkl                  ← fitted MinMaxScaler

outputs/plots/
  confusion_matrix.png        ← heatmap
  shap_bar.png                ← mean abs SHAP bar chart (top 15)
  shap_beeswarm.png           ← beeswarm dot plot
  shap_waterfall.png          ← single prediction waterfall
  optuna_history.png          ← optimization history (if tuning enabled)
  optuna_param_importance.png ← parameter importance (if tuning enabled)

outputs/reports/
  classification_report.csv   ← sklearn classification_report as CSV
  shap_importance.csv         ← feature, mean_abs_shap, rank
  best_params.json            ← best Optuna params (if tuning enabled)
```

---

## When Asked to Modify Existing Code

1. Read the existing file first — do not rewrite from scratch unless asked
2. Make targeted edits — change only what was asked
3. Preserve all existing docstrings and type hints
4. Do not change config.yaml keys unless explicitly asked — other modules depend on them
5. After any change to a module, check if `pipeline.py` still calls it correctly

---

## When Debugging

Common issues and how to handle them:

**yfinance MultiIndex columns:**
```python
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
```

**XGBoost use_label_encoder deprecation:**
```python
try:
    model = XGBClassifier(**params, use_label_encoder=False)
except TypeError:
    model = XGBClassifier(**params)  # newer versions don't need it
```

**FinBERT CUDA OOM:**
```python
# Reduce batch_size in config.yaml sentiment.batch_size from 16 to 4
```

**Optuna pruning noisy trials:**
```python
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
)
```

---

## Presentation — What to Say

### Problem statement (one sentence):
"Traditional stock prediction models rely solely on historical price data, ignoring the significant impact of real-time market sentiment on price movements."

### Innovation (three bullets):
- Multi-modal feature fusion: combining technical indicators with transformer-based financial sentiment
- Bayesian hyperparameter optimization using Optuna with time-series-aware cross-validation
- Prediction explainability via SHAP values — the model can say *why* it predicted UP, not just *that* it predicted UP

### Accuracy expectations:
- Acceptable: 54–58%
- Good: 58–63%
- Excellent: 63–68%
- Do not claim above 70% — quant funds with billion-dollar infrastructure barely achieve this consistently

### If asked "why not LSTM?":
> "LSTMs require large sequential datasets, are sensitive to hyperparameter choices, and behave as black boxes. XGBoost with SHAP gives us comparable or better performance on structured tabular data with full explainability — which is the actual research contribution of this work."
