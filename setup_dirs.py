"""Setup script to initialize project structure."""
from pathlib import Path

# Create directories
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

print("✓ Project structure created")
