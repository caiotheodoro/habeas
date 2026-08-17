import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "model" / "src"))
sys.path.insert(0, str(_root / "forge" / "src"))
