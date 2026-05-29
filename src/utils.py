"""Utility functions for configuration, logging, and setup."""

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
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    
    log_file = Path("logs")
    log_file.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_file / f"{name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_error_logger() -> logging.Logger:
    """Get error tracking logger - appends to error_log.txt."""
    error_logger = logging.getLogger("error_tracker")
    
    if error_logger.handlers:
        return error_logger
    
    error_logger.setLevel(logging.WARNING)
    
    log_file = Path("logs")
    log_file.mkdir(exist_ok=True)
    error_handler = logging.FileHandler(log_file / "error_log.txt")
    error_handler.setLevel(logging.WARNING)
    
    # Format with timestamp, error type, message
    error_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    error_handler.setFormatter(error_format)
    error_logger.addHandler(error_handler)
    
    return error_logger

