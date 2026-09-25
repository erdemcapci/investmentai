from __future__ import annotations

import os
from pathlib import Path

SCORING_MODEL_VERSION = "3.0.1"
TOP_N = int(os.getenv("TOP_N", "100"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
FORCE_REFRESH = os.getenv("FORCE_REFRESH", "false").lower() in {"1", "true", "yes", "y"}
EXPORT_RESULTS = os.getenv("EXPORT_RESULTS", "false").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DATA_DIR = Path(os.getenv("INVESTMENT_AI_DATA_DIR", "investment_ai_data"))
CACHE_DIR = DATA_DIR / "cache"
HISTORY_DB = DATA_DIR / "history.sqlite"
ANALYST_TTL_HOURS = 18
VALUATION_TTL_HOURS = 24
FUNDAMENTALS_TTL_HOURS = 24 * 7
PRICE_PERIOD = "2y"
