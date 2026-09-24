"""Yahoo parsing and download helpers extracted from the original notebook."""
from __future__ import annotations

import json
import logging
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf



SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
STOXX600_URL = "https://en.wikipedia.org/wiki/STOXX_Europe_600"
SUPPORTED_INDEXES = ("sp500", "stoxx600")

# Yahoo's European symbols use the primary exchange suffix.  The public STOXX
# table exposes country and local ticker rather than a vendor-specific symbol.
YAHOO_SUFFIX_BY_COUNTRY = {
    "austria": ".VI", "belgium": ".BR", "denmark": ".CO",
    "finland": ".HE", "france": ".PA", "germany": ".DE",
    "ireland": ".IR", "italy": ".MI", "netherlands": ".AS",
    "norway": ".OL", "poland": ".WA", "portugal": ".LS",
    "spain": ".MC", "sweden": ".ST", "switzerland": ".SW",
    "united kingdom": ".L", "uk": ".L",
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


def first_present(mapping: dict[str, Any] | None, keys: Iterable[str], default: Any = np.nan) -> Any:
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

        normalized["sp500_ticker"] = normalized["symbol"].astype(str).str.strip().str.upper()
        normalized["symbol"] = normalized["sp500_ticker"].map(normalize_symbol_for_yahoo)
        normalized["index_name"] = "S&P 500"
        normalized["constituent_list_fetched_at_utc"] = utc_now_iso()
        normalized = normalized.drop_duplicates("symbol").reset_index(drop=True)

        normalized.to_csv(cache_path, index=False)
        logging.info("Loaded %s S&P 500 listings.", len(normalized))
        return normalized

    except Exception as exc:
        logging.warning("Could not refresh constituent list: %s", exc)
        if cache_path.exists():
            cached = pd.read_csv(cache_path)
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
        if ({"company", "ticker", "country"} <= columns or
                {"company_name", "ticker", "country"} <= columns):
            result = table.copy()
            result.columns = [clean_column_name(c) for c in result.columns]
            return result
    raise ValueError("No STOXX Europe 600 constituent table was found")


def fetch_stoxx600_constituents(cache_path: Path) -> pd.DataFrame:
    """Fetch current STOXX Europe 600 members, falling back to its own cache."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; InvestmentAI/1.0)"}
    try:
        response = requests.get(STOXX600_URL, headers=headers, timeout=45)
        response.raise_for_status()
        raw = _find_stoxx_table(pd.read_html(StringIO(response.text)))
        company_column = "company" if "company" in raw else "company_name"
        industry_column = next((c for c in ("industry", "icb_sector", "sector") if c in raw), None)
        rows = []
        for _, item in raw.iterrows():
            try:
                symbol = yahoo_symbol_for_europe(item["ticker"], item["country"])
            except ValueError as exc:
                logging.warning(
                    "Skipping unmappable STOXX constituent %r (%r): %s",
                    item.get(company_column), item.get("ticker"), exc,
                )
                continue
            rows.append({
                "symbol": symbol,
                "stoxx_ticker": str(item["ticker"]).strip().upper(),
                "security": item[company_column],
                "gics_sector": item[industry_column] if industry_column else pd.NA,
                "country": item["country"],
                "index_name": "STOXX Europe 600",
                "constituent_list_fetched_at_utc": utc_now_iso(),
            })
        if not rows:
            raise ValueError("No mappable STOXX Europe 600 constituents were found")
        normalized = pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)
        normalized.to_csv(cache_path, index=False)
        logging.info("Loaded %s STOXX Europe 600 listings.", len(normalized))
        return normalized
    except Exception as exc:
        logging.warning("Could not refresh STOXX Europe 600 list: %s", exc)
        if cache_path.exists():
            return pd.read_csv(cache_path)
        raise RuntimeError(
            "STOXX Europe 600 list could not be downloaded and no cached list exists."
        ) from exc


def combine_index_constituents(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Normalize and deduplicate constituent frames by their Yahoo symbol.

    Memberships are combined in source order.  Metadata uses the first non-null
    value, which avoids losing useful fields when an overlapping listing has a
    sparse row in one of the source tables.
    """
    available = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not available:
        raise ValueError("No index constituent rows were supplied")

    combined = pd.concat(available, ignore_index=True, sort=False)
    required = {"symbol", "index_name"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Constituent data is missing required columns: {sorted(missing)}")

    combined["symbol"] = combined["symbol"].astype(str).str.strip().str.upper()
    combined = combined.loc[combined["symbol"].ne("") & combined["symbol"].ne("NAN")]
    combined["index_name"] = combined["index_name"].fillna("").astype(str)

    def first_present_value(values: pd.Series) -> Any:
        present = values.loc[values.notna()]
        return present.iloc[0] if not present.empty else pd.NA

    metadata_columns = [
        column for column in combined.columns
        if column not in {"symbol", "index_name"}
    ]
    metadata = combined.groupby("symbol", sort=False)[metadata_columns].agg(
        first_present_value
    ) if metadata_columns else pd.DataFrame(index=combined["symbol"].drop_duplicates())
    memberships = combined.groupby("symbol", sort=False)["index_name"].agg(
        lambda values: " | ".join(dict.fromkeys(v for v in values if v))
    )
    result = metadata.join(memberships).reset_index()

    # Expose canonical aliases used by main.py without removing the legacy
    # names consumed by investment_scanner.py and existing caches.
    alias_pairs = {
        "security": "company_name",
        "gics_sector": "sector",
        "gics_sub_industry": "sub_industry",
        "sp500_ticker": "source_symbol",
    }
    for source, destination in alias_pairs.items():
        if destination not in result and source in result:
            result[destination] = result[source]
    return result


def fetch_index_constituents(cache_dir: Path, indexes: Iterable[str] = SUPPORTED_INDEXES) -> pd.DataFrame:
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
        frames.append(fetch_stoxx600_constituents(cache_dir / "stoxx600_constituents.csv"))
    return combine_index_constituents(frames)


def looks_rate_limited(exc: Exception | str) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def call_with_retry(
    function: Callable[[], Any],
    label: str,
    max_attempts: int = 4,
    base_wait_seconds: float = 4.0,
) -> ApiResult:
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return ApiResult(function())
        except Exception as exc:  # yfinance can raise multiple exception classes
            last_error = exc
            if attempt == max_attempts:
                break

            multiplier = 4 if looks_rate_limited(exc) else 1
            wait = min(base_wait_seconds * (2 ** (attempt - 1)) * multiplier, 180)
            wait += random.uniform(0.2, 1.2)
            logging.warning(
                "%s failed (%s/%s): %s. Retrying after %.1f seconds.",
                label,
                attempt,
                max_attempts,
                truncate_text(exc, 240),
                wait,
            )
            time.sleep(wait)

    error_text = truncate_text(last_error, 800) if last_error else "Unknown error"
    logging.error("%s failed permanently: %s", label, error_text)
    return ApiResult(None, error_text)




def extract_ticker_frame(downloaded: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()

    if not isinstance(downloaded.columns, pd.MultiIndex):
        return downloaded.copy()

    level_0 = set(map(str, downloaded.columns.get_level_values(0)))
    level_1 = set(map(str, downloaded.columns.get_level_values(1)))

    if symbol in level_0:
        result = downloaded[symbol].copy()
    elif symbol in level_1:
        result = downloaded.xs(symbol, axis=1, level=1).copy()
    else:
        return pd.DataFrame()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(-1)
    return result


def series_return(series: pd.Series, trading_days: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= trading_days:
        return np.nan
    return safe_divide(clean.iloc[-1], clean.iloc[-(trading_days + 1)]) - 1


def annualized_volatility(series: pd.Series, window: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    returns = clean.pct_change(fill_method=None).dropna()
    if len(returns) < max(10, window // 2):
        return np.nan
    sample = returns.tail(window)
    return float(sample.std(ddof=1) * math.sqrt(252))


def calculate_price_metrics(symbol: str, history: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {"symbol": symbol}
    if history.empty:
        row["price_error"] = "No price history returned"
        return row

    history = history.sort_index()
    close_column = next((c for c in ["Close", "Adj Close", "close", "adj close"] if c in history), None)
    volume_column = next((c for c in ["Volume", "volume"] if c in history), None)

    if close_column is None:
        row["price_error"] = f"Close column not found: {list(history.columns)}"
        return row

    close = pd.to_numeric(history[close_column], errors="coerce").dropna()
    volume = (
        pd.to_numeric(history[volume_column], errors="coerce")
        if volume_column is not None
        else pd.Series(index=history.index, dtype=float)
    )

    if close.empty:
        row["price_error"] = "Close series is empty"
        return row

    current = safe_float(close.iloc[-1])
    ma_50 = safe_float(close.tail(50).mean()) if len(close) >= 20 else np.nan
    ma_200 = safe_float(close.tail(200).mean()) if len(close) >= 100 else np.nan
    high_52w = safe_float(close.tail(252).max())
    low_52w = safe_float(close.tail(252).min())
    avg_volume_20 = safe_float(volume.tail(20).mean()) if volume.notna().any() else np.nan
    avg_volume_60 = safe_float(volume.tail(60).mean()) if volume.notna().any() else np.nan

    last_index = close.index[-1]
    try:
        price_as_of = pd.Timestamp(last_index).isoformat()
    except Exception:
        price_as_of = str(last_index)

    row.update(
        {
            "price_close": current,
            "price_as_of": price_as_of,
            "return_1m_pct": 100 * series_return(close, 21),
            "return_3m_pct": 100 * series_return(close, 63),
            "return_6m_pct": 100 * series_return(close, 126),
            "return_1y_pct": 100 * series_return(close, 252),
            "moving_average_50d": ma_50,
            "moving_average_200d": ma_200,
            "distance_to_50d_ma_pct": 100 * (safe_divide(current, ma_50) - 1),
            "distance_to_200d_ma_pct": 100 * (safe_divide(current, ma_200) - 1),
            "high_52w": high_52w,
            "low_52w": low_52w,
            "distance_to_52w_high_pct": 100 * (safe_divide(current, high_52w) - 1),
            "distance_from_52w_low_pct": 100 * (safe_divide(current, low_52w) - 1),
            "volatility_20d_pct": 100 * annualized_volatility(close, 20),
            "volatility_60d_pct": 100 * annualized_volatility(close, 60),
            "average_volume_20d": avg_volume_20,
            "average_volume_60d": avg_volume_60,
            "average_daily_turnover_20d": current * avg_volume_20,
            "price_error": "",
        }
    )
    return row


def download_batch_history(symbols: list[str], period: str, timeout: int) -> pd.DataFrame:
    return yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        actions=False,
        threads=True,
        progress=False,
        timeout=timeout,
        repair=False,
        keepna=False,
        multi_level_index=True,
    )

# %%
def dataframe_row(frame: Any, labels: Iterable[str]) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(dtype=object)

    working = frame.copy()
    if "period" in working.columns:
        period_text = working["period"].astype(str).str.lower()
        for label in labels:
            match = working.loc[period_text == label.lower()]
            if not match.empty:
                return match.iloc[0]

    index_text = pd.Index(working.index).astype(str).str.lower()
    for label in labels:
        positions = np.where(index_text == label.lower())[0]
        if len(positions):
            return working.iloc[int(positions[0])]

    return working.iloc[0]


def series_value(series: pd.Series, names: Iterable[str], default: Any = np.nan) -> Any:
    if not isinstance(series, pd.Series) or series.empty:
        return default

    lower_map = {str(index).lower(): index for index in series.index}
    for name in names:
        actual = lower_map.get(name.lower())
        if actual is not None:
            value = series.get(actual)
            if not is_missing(value):
                return value
    return default


def parse_earnings_timestamp(info: dict[str, Any]) -> tuple[str | None, float]:
    timestamp_value = first_present(
        info,
        [
            "earningsTimestamp",
            "earningsTimestampStart",
            "earningsTimestampEnd",
        ],
        default=np.nan,
    )
    timestamp_number = safe_float(timestamp_value)
    if np.isnan(timestamp_number):
        return None, np.nan

    try:
        date = datetime.fromtimestamp(timestamp_number, tz=timezone.utc)
        days = (date - utc_now()).total_seconds() / 86_400
        return date.isoformat(timespec="seconds"), days
    except (OverflowError, OSError, ValueError):
        return None, np.nan




def price_target_metrics(targets: Any) -> dict[str, Any]:
    targets = targets if isinstance(targets, dict) else {}
    return {
        "target_current_reference": safe_float(targets.get("current")),
        "target_low": safe_float(targets.get("low")),
        "target_high": safe_float(targets.get("high")),
        "target_mean": safe_float(targets.get("mean")),
        "target_median": safe_float(targets.get("median")),
        # Yahoo/yfinance does not expose a reliable explicit expiry date.
        "target_horizon_date": None,
        "target_horizon_note": "Explicit target expiry not provided by Yahoo/yfinance",
    }


def eps_metrics(
    earnings_estimate: Any,
    eps_trend: Any,
    eps_revisions: Any,
) -> dict[str, Any]:
    estimate_row = dataframe_row(earnings_estimate, ["0q", "+1q", "0y"])
    trend_row = dataframe_row(eps_trend, ["0q", "+1q", "0y"])
    revisions_row = dataframe_row(eps_revisions, ["0q", "+1q", "0y"])

    current = safe_float(series_value(trend_row, ["current"]))
    ago_7 = safe_float(series_value(trend_row, ["7daysAgo", "7_days_ago"]))
    ago_30 = safe_float(series_value(trend_row, ["30daysAgo", "30_days_ago"]))

    return {
        "earnings_estimate_analysts": safe_int(
            series_value(estimate_row, ["numberOfAnalysts", "number_of_analysts"])
        ),
        "earnings_estimate_avg": safe_float(series_value(estimate_row, ["avg", "average"])),
        "earnings_estimate_low": safe_float(series_value(estimate_row, ["low"])),
        "earnings_estimate_high": safe_float(series_value(estimate_row, ["high"])),
        "earnings_estimate_growth_pct": 100
        * safe_float(series_value(estimate_row, ["growth"])),
        "eps_estimate_current": current,
        "eps_estimate_7d_ago": ago_7,
        "eps_estimate_30d_ago": ago_30,
        "eps_estimate_change_7d_pct": 100 * safe_divide(current - ago_7, abs(ago_7)),
        "eps_estimate_change_30d_pct": 100 * safe_divide(current - ago_30, abs(ago_30)),
        "eps_revisions_up_7d": safe_int(
            series_value(revisions_row, ["upLast7days", "up_last_7_days"])
        ),
        "eps_revisions_up_30d": safe_int(
            series_value(revisions_row, ["upLast30days", "up_last_30_days"])
        ),
        "eps_revisions_down_7d": safe_int(
            series_value(revisions_row, ["downLast7days", "down_last_7_days"])
        ),
        "eps_revisions_down_30d": safe_int(
            series_value(revisions_row, ["downLast30days", "down_last_30_days"])
        ),
    }


def rating_action_metrics(actions: Any) -> dict[str, Any]:
    output = {
        "rating_actions_30d": np.nan,
        "upgrades_30d": np.nan,
        "downgrades_30d": np.nan,
        "rating_actions_90d": np.nan,
        "upgrades_90d": np.nan,
        "downgrades_90d": np.nan,
        "latest_rating_date": None,
        "latest_rating_firm": None,
        "latest_rating_action": None,
        "latest_rating_from": None,
        "latest_rating_to": None,
    }

    if not isinstance(actions, pd.DataFrame) or actions.empty:
        return output

    frame = actions.copy()
    dates = pd.to_datetime(frame.index, utc=True, errors="coerce")
    if dates.isna().all() and "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.assign(_date=dates).dropna(subset=["_date"]).sort_values("_date")
    if frame.empty:
        return output

    now = pd.Timestamp.now(tz="UTC")
    action_column = next((c for c in ["action", "Action"] if c in frame.columns), None)
    action_text = (
        frame[action_column].astype(str).str.lower()
        if action_column
        else pd.Series("", index=frame.index)
    )

    for days in (30, 90):
        mask = frame["_date"] >= now - pd.Timedelta(days=days)
        subset_actions = action_text.loc[mask]
        output[f"rating_actions_{days}d"] = int(mask.sum())
        output[f"upgrades_{days}d"] = int(subset_actions.str.startswith("up").sum())
        output[f"downgrades_{days}d"] = int(subset_actions.str.startswith("down").sum())

    latest = frame.iloc[-1]
    output["latest_rating_date"] = latest["_date"].isoformat()
    output["latest_rating_firm"] = first_present(latest.to_dict(), ["firm", "Firm"], None)
    output["latest_rating_action"] = first_present(latest.to_dict(), ["action", "Action"], None)
    output["latest_rating_from"] = first_present(
        latest.to_dict(), ["fromGrade", "from_grade", "From Grade"], None
    )
    output["latest_rating_to"] = first_present(
        latest.to_dict(), ["toGrade", "to_grade", "To Grade"], None
    )
    return output


def info_metrics(info: Any) -> dict[str, Any]:
    info = info if isinstance(info, dict) else {}
    next_earnings_date, days_to_earnings = parse_earnings_timestamp(info)

    return {
        "company_name_yahoo": first_present(info, ["longName", "shortName"], None),
        "country": first_present(info, ["country"], None),
        "city": first_present(info, ["city"], None),
        "exchange": first_present(info, ["exchange", "fullExchangeName"], None),
        "exchange_timezone": first_present(
            info, ["exchangeTimezoneName", "timeZoneFullName"], None
        ),
        "currency": first_present(info, ["currency", "financialCurrency"], None),
        "quote_type": first_present(info, ["quoteType"], None),
        "sector_yahoo": first_present(info, ["sector", "sectorDisp"], None),
        "industry_yahoo": first_present(info, ["industry", "industryDisp"], None),
        "website": first_present(info, ["website"], None),
        "regular_market_price": safe_float(
            first_present(info, ["currentPrice", "regularMarketPrice"])
        ),
        "market_cap": safe_float(first_present(info, ["marketCap"])),
        "enterprise_value": safe_float(first_present(info, ["enterpriseValue"])),
        "beta": safe_float(first_present(info, ["beta"])),
        "trailing_pe": safe_float(first_present(info, ["trailingPE"])),
        "forward_pe": safe_float(first_present(info, ["forwardPE"])),
        "price_to_book": safe_float(first_present(info, ["priceToBook"])),
        "price_to_sales_ttm": safe_float(first_present(info, ["priceToSalesTrailing12Months"])),
        "enterprise_to_ebitda": safe_float(first_present(info, ["enterpriseToEbitda"])),
        "peg_ratio": safe_float(first_present(info, ["pegRatio", "trailingPegRatio"])),
        "profit_margin_pct": 100 * safe_float(first_present(info, ["profitMargins"])),
        "operating_margin_pct": 100 * safe_float(first_present(info, ["operatingMargins"])),
        "return_on_equity_pct": 100 * safe_float(first_present(info, ["returnOnEquity"])),
        "return_on_assets_pct": 100 * safe_float(first_present(info, ["returnOnAssets"])),
        "earnings_growth_pct": 100 * safe_float(first_present(info, ["earningsGrowth"])),
        "revenue_growth_pct": 100 * safe_float(first_present(info, ["revenueGrowth"])),
        "debt_to_equity": safe_float(first_present(info, ["debtToEquity"])),
        "free_cashflow": safe_float(first_present(info, ["freeCashflow"])),
        "operating_cashflow": safe_float(first_present(info, ["operatingCashflow"])),
        "total_cash": safe_float(first_present(info, ["totalCash"])),
        "total_debt": safe_float(first_present(info, ["totalDebt"])),
        "dividend_yield_pct": 100 * safe_float(first_present(info, ["dividendYield"])),
        "payout_ratio_pct": 100 * safe_float(first_present(info, ["payoutRatio"])),
        "shares_outstanding": safe_float(first_present(info, ["sharesOutstanding"])),
        "float_shares": safe_float(first_present(info, ["floatShares"])),
        "short_percent_float_pct": 100 * safe_float(first_present(info, ["shortPercentOfFloat"])),
        "average_volume_yahoo": safe_float(first_present(info, ["averageVolume", "averageDailyVolume10Day"])),
        "next_earnings_date_utc": next_earnings_date,
        "days_to_earnings": days_to_earnings,
    }

# %%
def component_to_dict(result: ApiResult) -> dict[str, Any]:
    """Convert an ApiResult to a pickle-friendly component record."""
    return {"value": result.value, "error": result.error}




def normalize_saved_yahoo_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Turn one saved raw Yahoo bundle into the flat row used by the analysis."""
    component_names = [
        "info", "analyst_price_targets", "recommendations",
        "earnings_estimate", "eps_trend", "eps_revisions",
        "upgrades_downgrades",
    ]
    component_errors = []
    for name in component_names:
        component = bundle.get(name, {})
        error = component.get("error") if isinstance(component, dict) else None
        if error:
            component_errors.append(f"{name}: {error}")

    def value(name: str) -> Any:
        component = bundle.get(name, {})
        return component.get("value") if isinstance(component, dict) else None

    row: dict[str, Any] = {
        "symbol": bundle.get("symbol"),
        "fetched_at_utc": bundle.get("fetched_at_utc"),
        "fetch_success": bundle.get("fetch_success", False),
        "component_errors": " | ".join(component_errors),
    }
    row.update(info_metrics(value("info")))
    row.update(price_target_metrics(value("analyst_price_targets")))
    row.update(recommendation_metrics(value("recommendations")))
    row.update(
        eps_metrics(
            value("earnings_estimate"),
            value("eps_trend"),
            value("eps_revisions"),
        )
    )
    row.update(rating_action_metrics(value("upgrades_downgrades")))
    return row


# %%

def recommendation_metrics(recommendations: Any) -> dict[str, Any]:
    """Only accept a complete, current distribution; unknown is not zero."""
    names = {'strong_buy': 'strongBuy', 'buy': 'buy', 'hold': 'hold',
             'sell': 'sell', 'strong_sell': 'strongSell'}
    empty = {k: np.nan for k in names}
    empty.update(recommendation_total=np.nan, ratings_valid=False)
    if not isinstance(recommendations, pd.DataFrame) or recommendations.empty:
        return empty
    periods = (recommendations['period'].astype(str).str.lower()
               if 'period' in recommendations else
               pd.Series(recommendations.index.astype(str).str.lower(), index=recommendations.index))
    current = recommendations.loc[periods.isin(['0m', 'current'])]
    if current.empty:
        return empty
    row = current.iloc[0]
    counts = {key: safe_float(series_value(row, [source, key])) for key, source in names.items()}
    valid = all(np.isfinite(v) and v >= 0 and v.is_integer() for v in counts.values())
    total = sum(counts.values()) if valid else np.nan
    return {**counts, 'recommendation_total': total, 'ratings_valid': bool(valid and total > 0)}
