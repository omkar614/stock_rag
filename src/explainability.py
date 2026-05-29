"""SHAP explainability analysis."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings before importing SHAP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shap

from src import utils

logger = utils.get_logger(__name__)
error_logger = utils.get_error_logger()


def generate_shap_report(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
) -> dict:
    """Generate SHAP analysis and save plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Computing SHAP values...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except (RuntimeError, TypeError) as e:
        error_msg = f"SHAP initialization failed: {type(e).__name__}: {e}"
        logger.warning(error_msg)
        error_logger.warning(error_msg)  # Also log to error file
        logger.warning("Skipping SHAP plots. Continuing with importance metrics...")
        
        # Return empty dict - plots will be skipped
        mean_abs_shap = np.zeros(len(feature_names))
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)
        
        csv_path = output_dir / "shap_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        logger.warning(f"Saved empty SHAP importance to {csv_path}")
        
        return dict(zip(importance_df["feature"], importance_df["mean_abs_shap"]))
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    # Mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    
    logger.info("Top 5 features:")
    for idx, row in importance_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    # Save importance CSV
    csv_path = output_dir / "shap_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    logger.info(f"Saved SHAP importance to {csv_path}")
    
    # Bar plot (top 15)
    try:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                         plot_type="bar", show=False)
        plt.tight_layout()
        bar_path = output_dir / "shap_bar.png"
        plt.savefig(bar_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved bar plot to {bar_path}")
    except Exception as e:
        logger.warning(f"Could not generate bar plot: {e}")
    
    # Beeswarm plot
    try:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        beeswarm_path = output_dir / "shap_beeswarm.png"
        plt.savefig(beeswarm_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved beeswarm plot to {beeswarm_path}")
    except Exception as e:
        logger.warning(f"Could not generate beeswarm plot: {e}")
    
    # Waterfall (first sample)
    try:
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test[0],
            feature_names=feature_names
        ), show=False)
        plt.tight_layout()
        waterfall_path = output_dir / "shap_waterfall.png"
        plt.savefig(waterfall_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved waterfall plot to {waterfall_path}")
    except Exception as e:
        logger.warning(f"Could not generate waterfall plot: {e}")
    
    return dict(zip(importance_df["feature"], importance_df["mean_abs_shap"]))
