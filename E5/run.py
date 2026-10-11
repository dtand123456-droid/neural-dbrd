"""Run the configured retrieval experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.launcher import main

if __name__ == "__main__":
    main('E5')
