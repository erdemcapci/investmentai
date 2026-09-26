from __future__ import annotations

import os
from pathlib import Path

APPLICATION_VERSION = "1.1.0"
SCORING_MODEL_VERSION = "3.1.1"
DATABASE_SCHEMA_VERSION = 5
CACHE_SCHEMA_VERSION = 3
OUTPUT_SCHEMA_VERSION = 1
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
ANALYST_TTL_HOURS = float(os.getenv("ANALYST_TTL_HOURS", "18"))
VALUATION_TTL_HOURS = float(os.getenv("VALUATION_TTL_HOURS", "24"))
FUNDAMENTALS_TTL_HOURS = float(os.getenv("FUNDAMENTALS_TTL_HOURS", str(24 * 7)))
PRICE_PERIOD = "2y"
MIN_PEERS = int(os.getenv("MIN_PEERS", "15"))
PRICE_BATCH_SIZE = int(os.getenv("PRICE_BATCH_SIZE", "150"))
PRICE_DOWNLOAD_ATTEMPTS = 3
RUNS_DIR = Path(os.getenv("INVESTMENT_AI_RUNS_DIR", "investment_ai_runs"))
SP500_MIN_CONSTITUENTS = int(os.getenv("SP500_MIN_CONSTITUENTS", "480"))
STOXX600_MIN_CONSTITUENTS = int(os.getenv("STOXX600_MIN_CONSTITUENTS", "560"))
MAPPING_DEGRADED_PCT = float(os.getenv("MAPPING_DEGRADED_PCT", "95"))
MAPPING_INVALID_PCT = float(os.getenv("MAPPING_INVALID_PCT", "90"))
CONSTITUENT_FRESH_HOURS = float(os.getenv("CONSTITUENT_FRESH_HOURS", "72"))
CONSTITUENT_MAX_AGE_HOURS = float(os.getenv("CONSTITUENT_MAX_AGE_HOURS", "336"))
PRICE_FRESH_VALID_PCT = float(os.getenv("PRICE_FRESH_VALID_PCT", "97"))
PRICE_FRESH_INVALID_PCT = float(os.getenv("PRICE_FRESH_INVALID_PCT", "90"))


def validate_config() -> None:
    errors = []
    if TOP_N <= 0:
        errors.append("TOP_N must be > 0")
    if not 1 <= MAX_WORKERS <= 32:
        errors.append("MAX_WORKERS must be between 1 and 32")
    if not 1 <= PRICE_BATCH_SIZE <= 500:
        errors.append("PRICE_BATCH_SIZE must be between 1 and 500")
    if MIN_PEERS < 2:
        errors.append("MIN_PEERS must be >= 2")
    if any(
        value < 0
        for value in (ANALYST_TTL_HOURS, VALUATION_TTL_HOURS, FUNDAMENTALS_TTL_HOURS)
    ):
        errors.append("TTL values must be >= 0")
    if errors:
        raise ValueError("Invalid configuration: " + "; ".join(errors))
