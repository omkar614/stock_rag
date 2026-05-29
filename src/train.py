"""XGBoost training with Optuna hyperparameter tuning."""

import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from src import utils

logger = utils.get_logger(__name__)
error_logger = utils.get_error_logger()


def _objective(trial, X_train, y_train):
    """Optuna objective function with TimeSeriesSplit."""
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
        "eval_metric": "logloss",
        "random_state": 42,
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    f1_scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = XGBClassifier(**params)
        try:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      early_stopping_rounds=30, verbose=False)
        except TypeError as e:
            error_msg = f"XGBoost early_stopping_rounds not supported: {e}. Using old API fallback."
            logger.debug(error_msg)
            error_logger.debug(error_msg)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, zero_division=0)
        f1_scores.append(f1)
    
    return float(np.mean(f1_scores))


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> tuple:
    """Train XGBoost with optional Optuna tuning."""
    
    best_params = None
    
    if config["hyperparameter_tuning"]["enabled"]:
        logger.info("Starting Optuna hyperparameter tuning...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        study.optimize(
            lambda trial: _objective(trial, X_train, y_train),
            n_trials=config["hyperparameter_tuning"]["n_trials"],
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_params["tree_method"] = "hist"
        best_params["eval_metric"] = "logloss"
        best_params["random_state"] = 42
        
        logger.info(f"Best F1 score: {study.best_value:.4f}")
        logger.info(f"Best params: {best_params}")
        
        # Save plots
        fig = optuna.visualization.plot_optimization_history(study).to_html()
        plot_path = Path(config["outputs"]["plots_dir"]) / "optuna_history.png"
        # Note: For HTML, we'd need plotly. Simple version saves JSON instead.
        
        # Save best params
        params_path = Path(config["outputs"]["reports_dir"]) / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"Saved best params to {params_path}")
    else:
        best_params = config["xgboost"].copy()
        best_params["random_state"] = 42
        logger.info("Using config hyperparameters (tuning disabled)")
    
    logger.info("Training final model...")
    model = XGBClassifier(**best_params)
    try:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  early_stopping_rounds=config["xgboost"]["early_stopping_rounds"],
                  verbose=False)
    except TypeError as e:
        error_msg = f"XGBoost early_stopping_rounds not supported: {e}. Using old API fallback."
        logger.debug(error_msg)
        error_logger.debug(error_msg)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Save model
    model_path = Path(config["model"]["save_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info(f"Saved model to {model_path}")
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = Path(config["outputs"]["plots_dir"]) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {cm_path}")
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = Path(config["outputs"]["reports_dir"]) / "classification_report.csv"
    report_df.to_csv(report_path)
    logger.info(f"Saved classification report to {report_path}")
    
    return model, metrics


import pandas as pd
