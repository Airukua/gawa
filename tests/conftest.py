import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path for imports like `model.*` and `training.*`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Optional: make tests more deterministic across environments.
os.environ.setdefault("PYTHONHASHSEED", "0")
