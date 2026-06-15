import os
import sys

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the FastAPI app from the api entrypoint
    from api.index import app  # type: ignore
except Exception as e:
    # Provide a helpful fallback if import fails
    raise RuntimeError(f"Failed to import API app: {e}")
