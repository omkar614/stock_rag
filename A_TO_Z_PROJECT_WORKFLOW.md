# A-to-Z Project Workflow

## 1. Project in one sentence

This project predicts next-day stock direction using a multi-modal machine learning pipeline that combines historical price data, technical indicators, and financial news sentiment, then explains the result with SHAP.

## 2. What the project uses

### Core technologies

- Python 3
- pandas and NumPy for data handling
- yfinance for historical price data
- NewsAPI for news, with a mock-news fallback when an API key is not available
- FinBERT (`ProsusAI/finbert`) for sentiment analysis, with VADER as fallback
- ta for technical indicators, with pandas-based fallback implementations
- scikit-learn for scaling, splitting, and evaluation
- XGBoost for classification
- Optuna for hyperparameter tuning
- SHAP for explainability
- Matplotlib and Seaborn for plots
- Streamlit and Plotly for the dashboard
- PyYAML for config loading

### Main project files

- `pipeline.py`: runs the full training workflow end to end
- `predict.py`: runs a single prediction flow
- `app.py`: Streamlit dashboard
- `configs/config.yaml`: all important settings
- `src/data_fetch.py`: prices and news collection
- `src/indicators.py`: technical indicator engineering
- `src/sentiment.py`: FinBERT/VADER sentiment features
- `src/preprocessing.py`: dataset build, split, and scaling
- `src/train.py`: Optuna tuning, XGBoost training, evaluation, and saving artifacts
- `src/explainability.py`: SHAP reports and plots
- `src/utils.py`: config, logging, seeding, and directory setup

## 3. What the project predicts

The target is binary next-day direction:

- `1` means the next close is higher than the current close
- `0` means the next close is not higher than the current close

In code, the label is built from:

- `df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)`

## 4. High-level workflow

The workflow runs in this order:

1. Load configuration from `configs/config.yaml`
2. Create required folders
3. Set the random seed
4. Download or load cached price data
5. Download or load cached news data
6. Compute technical indicators
7. Convert news into daily sentiment features
8. Merge price, indicator, and sentiment data into a dataset
9. Split the data chronologically into train and test sets
10. Scale features with `MinMaxScaler`
11. Train XGBoost, optionally with Optuna tuning
12. Evaluate the model and save metrics/plots
13. Generate SHAP explainability outputs
14. Use the saved model in the dashboard or prediction script

## 5. Data flow in detail

### A. Price data

`src/data_fetch.py` gets OHLCV price history from yfinance.

Behavior:

- Reads tickers from config
- Uses `start_date` and `end_date` from config
- Caches each ticker to `data/raw/<ticker>_prices.csv`
- Handles yfinance MultiIndex columns when present
- Falls back to cached data if it already exists

### B. News data

`src/data_fetch.py` also fetches news.

Behavior:

- Tries NewsAPI when an API key is available
- Saves articles to `data/raw/<ticker>_news.json`
- Uses a mock news generator when NewsAPI is unavailable or fails
- Keeps the workflow runnable even without external news access

### C. Technical indicators

`src/indicators.py` adds price-based and trend-based features.

Features include:

- Daily return
- Log return
- High-low spread
- Close-open spread
- SMA 20 and SMA 50
- EMA 12 and EMA 26
- RSI
- MACD, MACD signal, MACD histogram
- Bollinger upper/lower/middle bands and width
- Rolling standard deviation over 20 days
- Volume ratio
- On-balance volume

If the `ta` library is missing, the module falls back to simplified pandas implementations.

### D. Sentiment features

`src/sentiment.py` converts news into daily sentiment.

Behavior:

- Uses FinBERT when available
- Falls back to VADER if FinBERT cannot load
- Scores each article as positive, negative, or neutral
- Aggregates article scores by date
- Adds `num_articles`
- Fills missing dates with neutral defaults:
  - positive: `0.33`
  - negative: `0.33`
  - neutral: `0.34`
  - articles: `0`

### E. Dataset construction

`src/preprocessing.py` merges the price and sentiment data.

Behavior:

- Joins sentiment onto the price dataframe by date
- Fills missing sentiment values with neutral defaults
- Creates the binary target
- Drops the last row because it has no next-day label
- Removes remaining missing values
- Saves the final ticker dataset to `data/processed/<ticker>_dataset.csv`

## 6. Train/test handling

The project uses a time-aware split, not random shuffling.

Behavior:

- Splits by chronological order
- Uses the first portion as training data
- Uses the last portion as test data
- Fits `MinMaxScaler` only on the training set
- Transforms the test set with the already-fitted scaler
- Saves the scaler to `models/scaler.pkl`

This avoids lookahead bias and keeps the evaluation realistic for time-series data.

## 7. Model training

`src/train.py` trains the classifier.

### Default model

- `XGBClassifier`
- `tree_method: hist`
- `eval_metric: logloss`
- `random_state: 42`

### Hyperparameter tuning

When enabled in config, Optuna searches over:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`
- `reg_alpha`
- `reg_lambda`
- `scale_pos_weight`

Important details:

- Uses `TimeSeriesSplit`
- Optimizes mean F1 score across folds
- Uses a TPE sampler and median pruning
- Saves best parameters to `outputs/reports/best_params.json`

### Training output

After the final fit, the code saves:

- Model: `models/xgboost_model.json`
- Confusion matrix: `outputs/plots/confusion_matrix.png`
- Classification report: `outputs/reports/classification_report.csv`

### Metrics reported

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

### How to interpret accuracy

Accuracy tells you how often the model gets the next-day direction right.
For this project, a realistic result is usually in the mid-50% to mid-60% range because stock movement is noisy and difficult to predict consistently.

Why it matters:

- A value above random guessing means the model is finding some usable signal
- Accuracy alone is not enough, so it is checked together with F1, precision, recall, and ROC-AUC
- Strong-looking accuracy can still be misleading if the classes are imbalanced or the model is overfitting

## 8. Explainability

`src/explainability.py` uses SHAP TreeExplainer to explain the trained model.

Outputs saved in `outputs/plots/` and `outputs/reports/`:

- `shap_importance.csv`
- `shap_bar.png`
- `shap_beeswarm.png`
- `shap_waterfall.png`

What SHAP gives here:

- Global feature importance through mean absolute SHAP values
- A bar chart of the top features
- A beeswarm plot showing feature effects across samples
- A waterfall plot for one specific prediction

Why these figures matter:

- `confusion_matrix.png` shows where the model is getting UP and DOWN predictions correct or wrong
- `shap_bar.png` shows which features matter most overall
- `shap_beeswarm.png` shows how feature values push predictions up or down across many samples
- `shap_waterfall.png` explains one individual prediction in detail
- `classification_report.csv` gives class-wise precision, recall, and F1, which is important when one class is predicted better than the other
- `best_params.json` helps explain which hyperparameters Optuna selected and why the final model performed the way it did

If the accuracy is weak, the figures help diagnose whether the issue is class confusion, poor feature quality, weak sentiment signal, or overfitting. If the accuracy is strong, the figures help prove that the model is using sensible signals instead of memorizing noise.

If SHAP initialization fails, the pipeline still writes a feature importance CSV with zeros so the process does not stop entirely.

## 9. Dashboard workflow

`app.py` provides a Streamlit UI.

What it does:

- Loads config
- Lets the user choose a ticker from the supported list
- Loads the trained XGBoost model and scaler
- Fetches recent price data
- Recomputes indicators
- Shows a candlestick chart with SMA overlays
- Displays simple metrics in the sidebar area

Note: the dashboard is mainly for visualization and quick inspection of the trained pipeline rather than a full retraining interface.

## 10. Prediction workflow

`predict.py` is the single-run inference path.

Typical use:

- Load the saved model and scaler
- Fetch and preprocess the selected ticker
- Build the current feature vector
- Produce a direction prediction for the next day

Run example:

```bash
python predict.py --ticker AAPL
```

## 11. Full pipeline workflow

`pipeline.py` is the main orchestration script.

It runs these phases in order:

- price fetch
- news fetch
- indicator generation
- sentiment generation
- dataset construction
- train/test split and scaling
- XGBoost training
- SHAP explainability generation

Run example:

```bash
python pipeline.py
```

## 12. Configuration file

`configs/config.yaml` controls the project.

Important settings include:

- Supported tickers
- Date range
- Raw and processed data directories
- Sentiment engine and fallback behavior
- Indicator windows
- Model save locations
- Test split size
- XGBoost defaults
- Optuna trial count and search space
- Output directories

The project is designed so that most behavior changes should be made in this config file instead of hardcoding values in code.

## 13. Folder structure and what each folder holds

- `data/raw/`: cached prices and news JSON
- `data/processed/`: merged feature datasets
- `models/`: trained model and scaler
- `outputs/plots/`: confusion matrix, SHAP plots, Optuna plots
- `outputs/reports/`: CSV and JSON reports
- `logs/`: runtime and error logs
- `src/`: reusable project code

## 14. Supported markets and tickers

The project supports both US and Indian stocks.

Supported tickers from config:

- AAPL
- TSLA
- MSFT
- RELIANCE.NS
- TCS.NS
- INFY.NS
- HDFCBANK.NS
- WIPRO.NS

## 15. Fallback behavior and robustness

The code is built to keep running when optional dependencies or APIs are unavailable.

Examples:

- If NewsAPI fails, mock news is generated
- If FinBERT fails, VADER is used
- If `ta` is unavailable, pandas fallback indicators are used
- If SHAP fails, the pipeline still saves feature importance output
- If some XGBoost APIs differ by version, the code tries a fallback fit call

## 16. What gets produced after a successful run

Expected artifacts:

- `models/xgboost_model.json`
- `models/scaler.pkl`
- `data/processed/<ticker>_dataset.csv`
- `outputs/plots/confusion_matrix.png`
- `outputs/plots/shap_bar.png`
- `outputs/plots/shap_beeswarm.png`
- `outputs/plots/shap_waterfall.png`
- `outputs/reports/classification_report.csv`
- `outputs/reports/shap_importance.csv`
- `outputs/reports/best_params.json` when tuning is enabled

## 17. How to run the project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate project directories and baseline files

```bash
python generate_all.py
```

### Run the full training pipeline

```bash
python pipeline.py
```

### Run a prediction

```bash
python predict.py --ticker AAPL
```

### Launch the dashboard

```bash
streamlit run app.py
```

## 18. Plain-English summary of the whole system

This project takes stock prices and financial news, converts them into machine-learning features, trains an XGBoost classifier to predict whether the next closing price will go up or down, evaluates the model on a future test window, and explains the prediction using SHAP. The Streamlit app wraps the trained model so the results can be viewed interactively.

## 19. In short

If you want the shortest possible mental model, it is this:

- fetch data
- engineer indicators
- compute sentiment
- merge everything into one dataset
- split by time
- scale training data only
- tune/train XGBoost
- evaluate
- explain with SHAP
- serve with Streamlit
