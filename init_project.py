#!/usr/bin/env python3
"""Initialize project: create dirs and generate all source files."""

import json
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

def create_dirs():
    """Create directory structure."""
    dirs = [
        'configs',
        'src',
        'data/raw',
        'data/processed',
        'models',
        'outputs/plots',
        'outputs/reports',
        'logs'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")

def create_config_yaml():
    """Create configs/config.yaml."""
    config_yaml = """data:
  tickers: [AAPL, TSLA, MSFT]
  start_date: "2021-01-01"
  end_date: "2024-01-01"
  raw_dir: "data/raw"
  processed_dir: "data/processed"

news:
  api_key: ""
  articles_per_ticker: 10
  use_mock_fallback: true

sentiment:
  engine: "finbert"
  model_name: "ProsusAI/finbert"
  batch_size: 16
  neutral_default: 0.33

indicators:
  sma_windows: [20, 50]
  ema_windows: [12, 26]
  rsi_window: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bb_window: 20
  bb_std: 2.0

model:
  save_path: "models/xgboost_model.json"
  scaler_path: "models/scaler.pkl"
  test_size: 0.2
  random_state: 42

xgboost:
  n_estimators: 1000
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 5
  gamma: 0.1
  reg_alpha: 0.1
  reg_lambda: 1.0
  scale_pos_weight: 1
  eval_metric: "logloss"
  early_stopping_rounds: 50
  tree_method: "hist"

hyperparameter_tuning:
  enabled: true
  method: "optuna"
  n_trials: 50
  cv_folds: 5
  scoring: "f1"
  search_space:
    max_depth: [3, 10]
    learning_rate: [0.01, 0.3]
    n_estimators: [100, 1000]
    subsample: [0.6, 1.0]
    colsample_bytree: [0.6, 1.0]
    min_child_weight: [1, 10]
    gamma: [0.0, 0.5]
    reg_alpha: [0.0, 1.0]
    reg_lambda: [0.5, 2.0]

outputs:
  plots_dir: "outputs/plots"
  reports_dir: "outputs/reports"
"""
    Path("configs").mkdir(exist_ok=True)
    (Path("configs") / "config.yaml").write_text(config_yaml)
    print("✓ configs/config.yaml created")

def create_src_init():
    """Create src/__init__.py."""
    src_init = '"""Stock predictor ML system."""\n'
    Path("src").mkdir(exist_ok=True)
    (Path("src") / "__init__.py").write_text(src_init)
    print("✓ src/__init__.py created")

def create_requirements():
    """Create requirements.txt."""
    reqs = """yfinance>=0.2.28
newsapi-python>=0.2.7
ta>=0.11.0
xgboost>=2.0.0
scikit-learn>=1.3.0
optuna>=3.4.0
shap>=0.44.0
transformers>=4.35.0
torch>=2.0.0
vaderSentiment>=3.3.2
streamlit>=1.28.0
plotly>=5.17.0
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.66.0
"""
    Path("requirements.txt").write_text(reqs)
    print("✓ requirements.txt created")

def create_env_example():
    """Create .env.example."""
    env_example = """# NewsAPI.org free API key (optional - falls back to mock news if not set)
NEWS_API_KEY=your_newsapi_key_here

# Optional: FinBERT model caching directory
HF_HOME=./models/huggingface
"""
    Path(".env.example").write_text(env_example)
    print("✓ .env.example created")

if __name__ == "__main__":
    create_dirs()
    create_config_yaml()
    create_src_init()
    create_requirements()
    create_env_example()
    print("\n✅ Project initialization complete!")
