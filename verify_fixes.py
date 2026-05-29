#!/usr/bin/env python3
"""Verify that all fixes are in place."""

import sys
from pathlib import Path

def check_file_contains(file_path: str, search_str: str, description: str) -> bool:
    """Check if file contains expected string."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        found = search_str in content
        status = "[OK]" if found else "[FAIL]"
        print(f"{status} {description}")
        return found

print("=" * 60)
print("VERIFYING FIXES")
print("=" * 60)

all_good = True

print("\n1. SHAP FIX - explainability.py")
all_good &= check_file_contains(
    "src/explainability.py",
    "error_logger = utils.get_error_logger()",
    "error_logger imported"
)
all_good &= check_file_contains(
    "src/explainability.py",
    "explainer = shap.TreeExplainer(model)",
    "TreeExplainer called WITHOUT check_additivity"
)

print("\n2. ERROR LOGGER - utils.py")
all_good &= check_file_contains(
    "src/utils.py",
    "def get_error_logger()",
    "get_error_logger() function exists"
)
all_good &= check_file_contains(
    "src/utils.py",
    'log_file / "error_log.txt"',
    "error_log.txt path configured"
)

print("\n3. ERROR LOGGING IN MODULES")
all_good &= check_file_contains(
    "src/data_fetch.py",
    "error_logger.error(error_msg)",
    "data_fetch.py logs errors"
)
all_good &= check_file_contains(
    "src/sentiment.py",
    "error_logger.warning(error_msg)",
    "sentiment.py logs errors"
)
all_good &= check_file_contains(
    "src/train.py",
    "error_logger.debug(error_msg)",
    "train.py logs debug errors"
)

print("\n4. ERROR MESSAGE FORMAT")
all_good &= check_file_contains(
    "src/explainability.py",
    'f"SHAP initialization failed: {type(e).__name__}: {e}"',
    "Error messages include error type and details"
)

print("\n" + "=" * 60)
if all_good:
    print("✓ ALL FIXES VERIFIED")
    print("=" * 60)
    sys.exit(0)
else:
    print("✗ SOME FIXES MISSING")
    print("=" * 60)
    sys.exit(1)
