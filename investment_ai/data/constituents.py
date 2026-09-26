"""Yahoo parsing and download helpers extracted from the original notebook."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
STOXX600_URL = "https://en.wikipedia.org/wiki/STOXX_Europe_600"
SUPPORTED_INDEXES = ("sp500", "stoxx600")

# Yahoo's European symbols use the primary exchange suffix.  The public STOXX
# table exposes country and local ticker rather than a vendor-specific symbol.
YAHOO_SUFFIX_BY_COUNTRY = {
    "austria": ".VI",
    "belgium": ".BR",
    "denmark": ".CO",
    "finland": ".HE",
    "france": ".PA",
    "germany": ".DE",
    "ireland": ".IR",
    "italy": ".MI",
    "netherlands": ".AS",
    "norway": ".OL",
    "poland": ".WA",
    "portugal": ".LS",
    "spain": ".MC",
    "sweden": ".ST",
    "switzerland": ".SW",
    "united kingdom": ".L",
    "uk": ".L",
}

RATE_LIMIT_MARKERS = ("rate limit", "too many requests", "429", "yfratelimit")


@dataclass
class ApiResult:
    value: Any
    error: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def safe_float(value: Any) -> float:
    if is_missing(value):
        return np.nan
    try:
        number = float(value)
        return number if math.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def safe_int(value: Any) -> int | float:
    number = safe_float(value)
    return int(number) if not np.isnan(number) else np.nan


def first_present(
    mapping: dict[str, Any] | None, keys: Iterable[str], default: Any = np.nan
) -> Any:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        value = mapping.get(key)
        if not is_missing(value):
            return value
    return default


def normalize_symbol_for_yahoo(symbol: str) -> str:
    """Yahoo uses '-' for class shares that S&P/Wikipedia writes with '.'."""
    return str(symbol).strip().upper().replace(".", "-")


def truncate_text(value: Any, max_length: int = 500) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= max_length else text[: max_length - 3] + "..."


def clean_column_name(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def safe_divide(numerator: Any, denominator: Any) -> float:
    n = safe_float(numerator)
    d = safe_float(denominator)
    if np.isnan(n) or np.isnan(d) or d == 0:
        return np.nan
    return n / d


def make_unique_columns(columns: Iterable[Any]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for column in columns:
        base = clean_column_name(column)
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


def fetch_sp500_constituents(cache_path: Path) -> pd.DataFrame:
    """Fetch current constituents; fall back to the last cached list."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/150 Safari/537.36"
        )
    }

    try:
        logging.info("Downloading current S&P 500 constituent list...")
        response = requests.get(SP500_URL, headers=headers, timeout=45)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text), match="Symbol")
        if not tables:
            raise ValueError("No matching constituent table found")
        frame = tables[0].copy()
        frame.columns = make_unique_columns(frame.columns)

        required_aliases = {
            "symbol": ["symbol", "ticker"],
            "security": ["security", "company", "company_name"],
            "gics_sector": ["gics_sector", "sector"],
            "gics_sub_industry": ["gics_sub_industry", "sub_industry", "industry"],
            "headquarters_location": ["headquarters_location", "headquarters"],
            "date_added": ["date_added", "added"],
            "cik": ["cik"],
            "founded": ["founded"],
        }

        normalized = pd.DataFrame(index=frame.index)
        for output_name, aliases in required_aliases.items():
            source = next((name for name in aliases if name in frame.columns), None)
            normalized[output_name] = frame[source] if source else pd.NA

        normalized["sp500_ticker"] = (
            normalized["symbol"].astype(str).str.strip().str.upper()
        )
        normalized["symbol"] = normalized["sp500_ticker"].map(
            normalize_symbol_for_yahoo
        )
        normalized["index_name"] = "S&P 500"
        normalized["constituent_list_fetched_at_utc"] = utc_now_iso()
        normalized["constituent_source_cache_used"] = False
        normalized = normalized.drop_duplicates("symbol").reset_index(drop=True)

        normalized.to_csv(cache_path, index=False)
        logging.info("Loaded %s S&P 500 listings.", len(normalized))
        return normalized

    except Exception as exc:
        logging.warning("Could not refresh constituent list: %s", exc)
        if cache_path.exists():
            cached = pd.read_csv(cache_path)
            cached["constituent_source_cache_used"] = True
            logging.warning("Using cached constituent list: %s", cache_path)
            return cached
        raise RuntimeError(
            "S&P 500 list could not be downloaded and no cached list exists."
        ) from exc


def yahoo_symbol_for_europe(local_ticker: Any, country: Any) -> str:
    """Translate the STOXX table's local ticker to Yahoo's exchange symbol."""
    ticker = str(local_ticker).strip().upper().replace(" ", "-")
    if not ticker or ticker == "NAN":
        raise ValueError("Missing STOXX ticker")
    suffix = YAHOO_SUFFIX_BY_COUNTRY.get(str(country).strip().lower())
    if not suffix:
        raise ValueError(f"Unsupported STOXX listing country: {country}")
    # Preserve an already vendor-qualified ticker supplied by a future table.
    if ticker.endswith(suffix):
        return ticker
    return ticker.replace(".", "-") + suffix


def _find_stoxx_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        columns = {clean_column_name(c) for c in table.columns}
        if {"company", "ticker", "country"} <= columns or {
            "company_name",
            "ticker",
            "country",
        } <= columns:
            result = table.copy()
            result.columns = [clean_column_name(c) for c in result.columns]
            return result
    raise ValueError("No STOXX Europe 600 constituent table was found")


def fetch_stoxx600_constituents(cache_path: Path) -> pd.DataFrame:
    """Fetch raw STOXX members without pre-empting authoritative resolution."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; InvestmentAI/1.0)"}
    try:
        response = requests.get(STOXX600_URL, headers=headers, timeout=45)
        response.raise_for_status()
        raw = _find_stoxx_table(pd.read_html(StringIO(response.text)))
        company_column = "company" if "company" in raw else "company_name"
        industry_column = next(
            (c for c in ("industry", "icb_sector", "sector") if c in raw), None
        )
        rows = []
        for _, item in raw.iterrows():
            source_symbol = str(item.get("ticker", "")).strip()
            rows.append(
                {
                    # This is deliberately the source ticker. SymbolResolver is
                    # the only authority allowed to produce a Yahoo symbol.
                    "symbol": source_symbol,
                    "source_symbol": source_symbol,
                    "stoxx_ticker": source_symbol,
                    "security": item[company_column],
                    "gics_sector": item[industry_column] if industry_column else pd.NA,
                    "country": item["country"],
                    "exchange": item.get("exchange", pd.NA),
                    "isin": item.get("isin", pd.NA),
                    "index_name": "STOXX Europe 600",
                    "constituent_list_fetched_at_utc": utc_now_iso(),
                    "constituent_source_cache_used": False,
                }
            )
        if not rows:
            raise ValueError("No STOXX Europe 600 constituents were found")
        normalized = pd.DataFrame(rows).drop_duplicates("source_symbol").reset_index(drop=True)
        normalized.to_csv(cache_path, index=False)
        logging.info("Loaded %s STOXX Europe 600 listings.", len(normalized))
        return normalized
    except Exception as exc:
        logging.warning("Could not refresh STOXX Europe 600 list: %s", exc)
        if cache_path.exists():
            cached = pd.read_csv(cache_path)
            cached["constituent_source_cache_used"] = True
            return cached
        raise RuntimeError(
            "STOXX Europe 600 list could not be downloaded and no cached list exists."
        ) from exc


def combine_index_constituents(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine source rows without vendor-symbol deduplication before resolution."""
    available = [
        frame.copy() for frame in frames if frame is not None and not frame.empty
    ]
    if not available:
        raise ValueError("No index constituent rows were supplied")

    combined = pd.concat(available, ignore_index=True, sort=False)
    required = {"symbol", "index_name"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(
            f"Constituent data is missing required columns: {sorted(missing)}"
        )

    combined["symbol"] = combined["symbol"].astype(str).str.strip()
    combined = combined.loc[combined["symbol"].ne("") & combined["symbol"].ne("NAN")]
    combined["index_name"] = combined["index_name"].fillna("").astype(str)

    result = combined.reset_index(drop=True)

    # Expose stable aliases used by the v3 pipeline.
    alias_pairs = {
        "security": "company_name",
        "gics_sector": "sector",
        "gics_sub_industry": "sub_industry",
        "sp500_ticker": "source_symbol",
    }
    for source, destination in alias_pairs.items():
        if source in result:
            if destination not in result:
                result[destination] = result[source]
            else:
                result[destination] = result[destination].combine_first(result[source])
    return result


def fetch_index_constituents(
    cache_dir: Path, indexes: Iterable[str] = SUPPORTED_INDEXES
) -> pd.DataFrame:
    """Fetch and combine requested indexes while retaining overlapping membership."""
    requested = tuple(dict.fromkeys(str(name).lower() for name in indexes))
    unknown = set(requested) - set(SUPPORTED_INDEXES)
    if unknown:
        raise ValueError(f"Unsupported indexes: {sorted(unknown)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    if "sp500" in requested:
        frames.append(fetch_sp500_constituents(cache_dir / "sp500_constituents.csv"))
    if "stoxx600" in requested:
        frames.append(
            fetch_stoxx600_constituents(cache_dir / "stoxx600_constituents.csv")
        )
    return combine_index_constituents(frames)
