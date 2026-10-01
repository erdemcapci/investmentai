"""Yahoo parsing and download helpers extracted from the original notebook."""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from investment_ai.config import (
    APPLICATION_VERSION,
    SP500_MIN_CONSTITUENTS,
    STOXX600_MIN_CONSTITUENTS,
)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
STOXX600_ISHARES_URL = (
    "https://www.ishares.com/ch/individual/en/products/251931/"
    "ishares-stoxx-europe-600-ucits-etf-de-fund/1495092304805.ajax"
    "?fileType=csv&fileName=EXSA_holdings&dataType=fund"
)
STOXX600_WIKIMEDIA_URL = (
    "https://en.wikipedia.org/w/rest.php/v1/page/STOXX_Europe_600/html"
)
SUPPORTED_INDEXES = ("sp500", "stoxx600")
WIKIMEDIA_HEADERS = {
    "User-Agent": (
        f"InvestmentAI/{APPLICATION_VERSION} "
        "(https://github.com/erdemcapci/investmentai)"
    ),
    "Accept": "text/html",
}
DOWNLOAD_HEADERS = {
    "User-Agent": (
        f"InvestmentAI/{APPLICATION_VERSION} "
        "(https://github.com/erdemcapci/investmentai)"
    )
}

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

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


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


def normalize_symbol_for_yahoo(symbol: str) -> str:
    """Yahoo uses '-' for class shares that S&P/Wikipedia writes with '.'."""
    return str(symbol).strip().upper().replace(".", "-")


def clean_column_name(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def make_unique_columns(columns: Iterable[Any]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for column in columns:
        base = clean_column_name(column)
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


def _validated_constituents(
    frame: pd.DataFrame, index_name: str, minimum_rows: int
) -> pd.DataFrame:
    """Reject partial/malformed downloads before they can replace a good cache."""
    required = {"symbol", "security", "index_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{index_name} data is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str).str.strip()
    result = result[
        result["symbol"].ne("")
        & result["symbol"].str.upper().ne("NAN")
        & result["security"].notna()
    ].reset_index(drop=True)
    if len(result) < minimum_rows:
        raise ValueError(
            f"{index_name} returned {len(result)} rows; minimum is {minimum_rows}"
        )
    return result


def _write_constituent_cache(frame: pd.DataFrame, cache_path: Path) -> None:
    """Atomically replace a constituent cache only after caller validation."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, cache_path)


def _read_valid_cache(
    cache_path: Path, index_name: str, minimum_rows: int
) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    try:
        cached = _validated_constituents(
            pd.read_csv(cache_path), index_name, minimum_rows
        )
    except Exception as exc:
        logging.warning("Ignoring invalid constituent cache %s: %s", cache_path, exc)
        return None
    cached["constituent_source_cache_used"] = True
    logging.warning("Using cached constituent list: %s", cache_path)
    return cached


def fetch_sp500_constituents(cache_path: Path) -> pd.DataFrame:
    """Fetch current constituents; fall back to the last cached list."""
    try:
        logging.info("Downloading current S&P 500 constituent list...")
        response = requests.get(SP500_URL, headers=WIKIMEDIA_HEADERS, timeout=45)
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

        normalized = _validated_constituents(
            normalized, "S&P 500", SP500_MIN_CONSTITUENTS
        )
        _write_constituent_cache(normalized, cache_path)
        logging.info("Loaded %s S&P 500 listings.", len(normalized))
        return normalized

    except Exception as exc:
        logging.warning("Could not refresh constituent list: %s", exc)
        cached = _read_valid_cache(
            cache_path, "S&P 500", SP500_MIN_CONSTITUENTS
        )
        if cached is not None:
            return cached
        raise RuntimeError(
            "S&P 500 list could not be downloaded and no cached list exists."
        ) from exc


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


def _normalize_stoxx(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [clean_column_name(c) for c in raw.columns]
    company_column = next(
        (c for c in ("company", "company_name", "name") if c in raw), None
    )
    ticker_column = next(
        (c for c in ("ticker", "issuer_ticker") if c in raw), None
    )
    country_column = next(
        (c for c in ("country", "location") if c in raw), None
    )
    if not company_column or not ticker_column or not country_column:
        raise ValueError("STOXX source is missing company, ticker, or country")
    if "asset_class" in raw:
        raw = raw[raw["asset_class"].astype(str).str.casefold().eq("equity")]
    industry_column = next(
        (c for c in ("industry", "icb_sector", "sector") if c in raw), None
    )
    fetched_at = utc_now_iso()
    normalized = pd.DataFrame(
        {
            "symbol": raw[ticker_column].astype(str).str.strip(),
            "source_symbol": raw[ticker_column].astype(str).str.strip(),
            "stoxx_ticker": raw[ticker_column].astype(str).str.strip(),
            "security": raw[company_column],
            "gics_sector": raw[industry_column] if industry_column else pd.NA,
            "country": raw[country_column],
            "exchange": raw["exchange"] if "exchange" in raw else pd.NA,
            "isin": raw["isin"] if "isin" in raw else pd.NA,
            "index_name": "STOXX Europe 600",
            "constituent_list_fetched_at_utc": fetched_at,
            "constituent_source_cache_used": False,
        }
    )
    return normalized.drop_duplicates(
        ["source_symbol", "country", "security"]
    ).reset_index(drop=True)


def _parse_ishares_stoxx(text: str) -> pd.DataFrame:
    # The first line is the holdings date and the second is blank/non-breaking space.
    return _normalize_stoxx(pd.read_csv(StringIO(text), skiprows=2))


def _parse_wikimedia_stoxx(text: str) -> pd.DataFrame:
    return _normalize_stoxx(_find_stoxx_table(pd.read_html(StringIO(text))))


def fetch_stoxx600_constituents(cache_path: Path) -> pd.DataFrame:
    """Fetch, validate and atomically cache the complete STOXX universe."""
    failures = []
    sources = (
        (STOXX600_ISHARES_URL, DOWNLOAD_HEADERS, _parse_ishares_stoxx, "iShares"),
        (
            STOXX600_WIKIMEDIA_URL,
            WIKIMEDIA_HEADERS,
            _parse_wikimedia_stoxx,
            "Wikimedia",
        ),
    )
    for url, headers, parser, source_name in sources:
        try:
            response = requests.get(url, headers=headers, timeout=45)
            response.raise_for_status()
            normalized = _validated_constituents(
                parser(response.text),
                "STOXX Europe 600",
                STOXX600_MIN_CONSTITUENTS,
            )
            _write_constituent_cache(normalized, cache_path)
            logging.info(
                "Loaded %s STOXX Europe 600 listings from %s.",
                len(normalized),
                source_name,
            )
            return normalized
        except Exception as exc:
            failures.append(f"{source_name}: {exc}")
            logging.warning(
                "Could not refresh STOXX Europe 600 list from %s: %s",
                source_name,
                exc,
            )
    cached = _read_valid_cache(
        cache_path, "STOXX Europe 600", STOXX600_MIN_CONSTITUENTS
    )
    if cached is not None:
        return cached
    raise RuntimeError(
        "STOXX Europe 600 list could not be downloaded and no valid cached "
        f"list exists ({'; '.join(failures)})."
    )


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
