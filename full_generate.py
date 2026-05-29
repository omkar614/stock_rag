#!/usr/bin/env python3
"""
COMPLETE PROJECT GENERATOR
Generates entire stock predictor ML system with all modules.
Run: python full_generate.py
"""

from pathlib import Path

def main():
    """Create all project files and directories."""
    
    # Create directories
    for d in [
        'configs', 'src', 'data/raw', 'data/processed',
        'models', 'outputs/plots', 'outputs/reports', 'logs'
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # Dictionary of all files
    files = {}
    
    # ===== CONFIG =====
    files['configs/config.yaml'] = '''data:
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
'''
    
    # ===== SRC MODULE: __init__ =====
    files['src/__init__.py'] = '''"""Stock predictor ML system."""
'''
    
    # ===== SRC MODULE: utils.py =====
    files['src/utils.py'] = '''"""Utility functions for configuration, logging, and setup."""

import logging
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device() -> str:
    """Get available device: cuda or cpu."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def setup_dirs(config: dict) -> None:
    """Create all required directories."""
    dirs = [
        config["data"]["raw_dir"],
        config["data"]["processed_dir"],
        config["outputs"]["plots_dir"],
        config["outputs"]["reports_dir"],
        "models",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Get configured logger with file and console handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    log_file = Path("logs")
    log_file.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_file / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(console_format)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seed for all libraries."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
'''
    
    # ===== REQUIREMENTS =====
    files['requirements.txt'] = '''yfinance>=0.2.28
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
'''
    
    # ===== ENV EXAMPLE =====
    files['.env.example'] = '''NEWS_API_KEY=your_newsapi_key_here
HF_HOME=./models/huggingface
'''
    
    # Write all files
    for file_path, content in files.items():
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        print(f"✓ {file_path}")
    
    print(f"\n✅ Created {len(files)} files and directories!")

if __name__ == "__main__":
    main()
