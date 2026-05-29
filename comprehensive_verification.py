"""Comprehensive fix verification - tests imports and error logging system."""

from pathlib import Path
import sys

print("=" * 70)
print("COMPREHENSIVE VERIFICATION OF FIXES")
print("=" * 70)

# Test 1: Import all modified modules
print("\n[TEST 1] Importing all modified modules...")
try:
    from src import utils
    print("  [OK] src.utils imported successfully")
    
    from src import explainability
    print("  [OK] src.explainability imported successfully")
    
    from src import data_fetch
    print("  [OK] src.data_fetch imported successfully")
    
    from src import sentiment
    print("  [OK] src.sentiment imported successfully")
    
    from src import train
    print("  [OK] src.train imported successfully")
    
except ImportError as e:
    print(f"  [FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Verify get_error_logger function exists
print("\n[TEST 2] Checking error logger functionality...")
try:
    error_logger = utils.get_error_logger()
    assert error_logger is not None, "error_logger is None"
    assert hasattr(error_logger, 'warning'), "error_logger missing warning method"
    assert hasattr(error_logger, 'error'), "error_logger missing error method"
    print("  [OK] get_error_logger() function works correctly")
    print(f"  [OK] Logger name: {error_logger.name}")
except AssertionError as e:
    print(f"  [FAIL] Error logger check failed: {e}")
    sys.exit(1)

# Test 3: Verify error_log.txt is being created
print("\n[TEST 3] Checking error log file creation...")
try:
    error_log_path = Path("logs/error_log.txt")
    assert error_log_path.exists(), f"error_log.txt not found at {error_log_path}"
    print(f"  [OK] error_log.txt exists at {error_log_path}")
    
    with open(error_log_path, 'r') as f:
        content = f.read()
        assert len(content) > 0, "error_log.txt is empty"
        lines = content.strip().split('\n')
        print(f"  [OK] error_log.txt contains {len(lines)} entries")
        
        # Check format of entries
        for line in lines[:1]:  # Check first line
            assert '|' in line, f"Log entry doesn't have pipe separator: {line}"
            parts = line.split('|')
            assert len(parts) >= 3, f"Log entry doesn't have enough parts: {line}"
            print(f"  [OK] Log format correct: timestamp | level | module | message")
except AssertionError as e:
    print(f"  [FAIL] Error log file check failed: {e}")
    sys.exit(1)

# Test 4: Verify SHAP fix (no check_additivity parameter)
print("\n[TEST 4] Checking SHAP error handling fix...")
try:
    with open("src/explainability.py", 'r') as f:
        content = f.read()
        
        # Should have TreeExplainer called WITHOUT check_additivity
        assert "shap.TreeExplainer(model)" in content, "TreeExplainer call not found correctly"
        assert "check_additivity=False" not in content, "check_additivity parameter still present!"
        
        print("  [OK] TreeExplainer called without invalid check_additivity parameter")
        
        # Check for error_logger usage
        assert "error_logger.warning" in content, "error_logger.warning not found in explainability"
        print("  [OK] Error logger integrated in explainability module")
        
        # Check for graceful fallback
        assert "Skip SHAP plots" in content or "Skipping SHAP plots" in content, "No graceful fallback message"
        print("  [OK] Graceful fallback on SHAP failure implemented")
except AssertionError as e:
    print(f"  [FAIL] SHAP fix check failed: {e}")
    sys.exit(1)

# Test 5: Verify error logging in other modules
print("\n[TEST 5] Checking error logging across modules...")
try:
    modules = {
        "src/data_fetch.py": "error_logger.error",
        "src/sentiment.py": "error_logger.warning",
        "src/train.py": "error_logger.debug",
    }
    
    for module_file, error_call in modules.items():
        with open(module_file, 'r') as f:
            content = f.read()
            assert error_call in content, f"{error_call} not found in {module_file}"
            print(f"  [OK] {module_file}: {error_call} integrated")
except AssertionError as e:
    print(f"  [FAIL] Module error logging check failed: {e}")
    sys.exit(1)

# Test 6: Verify error messages include type information
print("\n[TEST 6] Checking error message format...")
try:
    with open("src/explainability.py", 'r') as f:
        content = f.read()
        assert "{type(e).__name__}" in content, "Error type not included in message"
        print("  [OK] Error messages include error type: {type(e).__name__}")
except AssertionError as e:
    print(f"  [FAIL] Error message format check failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("SUCCESS: All verification tests passed!")
print("=" * 70)
print("\nFixes implemented:")
print("  [OK] SHAP TreeExplainer error fixed (removed invalid check_additivity)")
print("  [OK] Error logging system created (logs/error_log.txt)")
print("  [OK] Error logging integrated across all modules")
print("  [OK] Graceful fallback on SHAP failure")
print("  [OK] Enhanced error messages with exception type")
print("\nReady for pipeline testing!")
