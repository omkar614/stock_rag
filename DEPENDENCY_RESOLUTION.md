# Dependency Installation Note

## Issue

Streamlit app failed with: `ModuleNotFoundError: No module named 'ta'`

## Root Cause

Dependencies from `requirements.txt` were not installed in the Python environment.

## Solution Applied

Installed all dependencies:

```bash
pip install -r requirements.txt
```

## Verification

[OK] ta module - Available
[OK] src.indicators module - Imports successfully
[OK] All core modules - Import verified

## Status

All dependencies installed and verified. The Streamlit app should now run without import errors.

## To Run the App

```bash
streamlit run app.py
```

## Requirements Installed (19 packages)

- yfinance>=0.2.28
- newsapi-python>=0.2.7
- ta>=0.11.0 (Technical Analysis library)
- xgboost>=2.0.0
- scikit-learn>=1.3.0
- optuna>=3.4.0
- shap>=0.44.0
- transformers>=4.35.0
- torch>=2.0.0
- vaderSentiment>=3.3.2
- streamlit>=1.28.0
- plotly>=5.17.0
- matplotlib>=3.7.0
- pandas>=2.0.0
- numpy>=1.24.0
- pyyaml>=6.0
- python-dotenv>=1.0.0
- tqdm>=4.66.0

## Next Steps

Try running the app:

```bash
streamlit run app.py
```

If any other import errors occur, verify the Python environment has all dependencies.
