"""
Entry point for the Marketing Video Pipeline API.

Usage:
  python run_marketing_api.py

Environment variables: see marketing/config.py
"""

import os
import sys
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(
        "marketing.api:app",
        host=host,
        port=port,
        workers=1,  # Single worker (GPU can only handle 1 job at a time)
    )
