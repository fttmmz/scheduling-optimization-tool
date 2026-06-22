import os
import sys

# Vercel's Python runtime invokes this file directly, so `src/` (which
# contains the `backend` package) needs to be added to the import path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.main import app  # noqa: E402
