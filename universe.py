"""Large-cap universe discovery and neutral Top-100 analysis-universe selection.

The investment model should not be limited by index membership.  This module
builds a broad NYSE/Nasdaq equity universe and then chooses the stocks that the
existing scanner can analyse most reliably.  Importantly, the universe score
never uses analyst direction (Buy/Sell), target upside, or recent returns; those
belong to the investment scores that run *after* universe selection.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery


DEFAULT_EXCHANGES = ("NYQ", "NMS", "NGM", "NCM")
SPAC_NAME_PATTERN = re.compile(
    r"\b(blank\s+check|acquisition\s+(?:corp(?:oration)?|co(?:mpany)?))\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class UniverseConfig:
    """Eligibility and neutral selection rules for the analysis universe."""

    min_market_cap: float = 10_000_000_000
    min_average_dollar_volume: float = 20_000_000
    min_analyst_count: int = 8
    min_data_quality_score: float = 60.0
    analysis_universe_size: int = 100
    exchanges: tuple[str, ...] = DEFAULT_EXCHANGES
    page_size: int = 250
    max_pages: int = 20


UNIVERSE_SCORE_WEIGHTS = {
    "analyst_coverage": 0.35,
    "market_cap": 0.25,
    "liquidity": 0.20,
    "data_quality": 0.20,
}


def _safe_float(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("raw", value.get("value"))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def _first_value(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name not in mapping:
            continue
        value = mapping.get(name)
        if isinstance(value, dict):
            value = value.get("raw", value.get("fmt", value.get("value")))
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return default


def analyst_coverage_band(rating_count: Any) -> str:
    """Human-readable coverage band agreed for Investment AI."""
    count = _safe_float(rating_count)
    if pd.isna(count) or count < 5:
        return "INSUFFICIENT"
    if count < 8:
        return "LOW"
    if count < 15:
        return "ACCEPTABLE"
    if count < 25:
        return "STRONG"
    return "VERY_STRONG"


def analyst_coverage_confidence(rating_count: Any) -> float:
    """Coverage confidence factor used by the long-term analyst score.

    The broad universe requires at least eight current recommendations, so the
    0.50/0.70 bands mainly remain useful for diagnostics and direct function use.
    """
    count = _safe_float(rating_count)
    if pd.isna(count):
        return 0.50
    if count >= 25:
        return 1.00
    if count >= 15:
        return 0.95
    if count >= 8:
        return 0.85
    if count >= 5:
        return 0.70
    return 0.50


def _extract_quotes(response: Any) -> list[dict[str, Any]]:
    """Handle both current yfinance screen output and defensive nested shapes."""
    if not isinstance(response, dict):
        return []
    quotes = response.get("quotes")
    if isinstance(quotes, list):
        return [row for row in quotes if isinstance(row, dict)]
    finance = response.get("finance")
    if isinstance(finance, dict):
        result = finance.get("result")
        if isinstance(result, list) and result:
            quotes = result[0].get("quotes") if isinstance(result[0], dict) else None
            if isinstance(quotes, list):
                return [row for row in quotes if isinstance(row, dict)]
    return []


def _response_total(response: Any) -> int | None:
    if not isinstance(response, dict):
        return None
    for key in ("total", "count"):
        value = response.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    finance = response.get("finance")
    if isinstance(finance, dict):
        result = finance.get("result")
        if isinstance(result, list) and result and isinstance(result[0], dict):
            for key in ("total", "count"):
                try:
                    value = result[0].get(key)
                    if value is not None:
                        return int(value)
                except (TypeError, ValueError):
                    pass
    return None


def _normalize_screen_quote(quote: dict[str, Any]) -> dict[str, Any]:
    symbol = str(_first_value(quote, ["symbol"], "")).strip().upper().replace(".", "-")
    company_name = _first_value(quote, ["longName", "shortName", "displayName"], symbol)
    exchange = str(_first_value(quote, ["exchange"], "")).strip().upper()
    quote_type = str(_first_value(quote, ["quoteType", "typeDisp"], "")).strip().upper()
    sector = _first_value(quote, ["sector", "sectorDisp"], "Unknown")
    industry = _first_value(quote, ["industry", "industryDisp"], "")
    market_cap = _safe_float(
        _first_value(
            quote,
            [
                "marketCap",
                "intradaymarketcap",
                "lastclosemarketcap.lasttwelvemonths",
            ],
        )
    )
    price = _safe_float(
        _first_value(quote, ["regularMarketPrice", "intradayprice", "eodprice"])
    )
    avg_volume_3m = _safe_float(
        _first_value(quote, ["averageDailyVolume3Month", "avgdailyvol3m"])
    )
    name_text = str(company_name or "")
    industry_text = str(industry or "")
    is_spac = bool(
        industry_text.strip().lower() == "shell companies"
        or SPAC_NAME_PATTERN.search(name_text)
    )
    is_adr_guess = bool(
        re.search(r"\bADR\b|depositary", name_text, flags=re.IGNORECASE)
        or re.search(r"depositary", str(_first_value(quote, ["typeDisp"], "")), flags=re.IGNORECASE)
    )
    return {
        "symbol": symbol,
        "source_symbol": symbol,
        "company_name": company_name,
        "sector": sector if sector else "Unknown",
        "industry": industry,
        "sub_industry": industry,
        "exchange": exchange,
        "quote_type": quote_type,
        "market_cap": market_cap,
        "screener_price": price,
        "screener_average_volume_3m": avg_volume_3m,
        "screener_average_dollar_volume_3m": (
            price * avg_volume_3m
            if pd.notna(price) and pd.notna(avg_volume_3m)
            else np.nan
        ),
        "is_adr_guess": is_adr_guess,
        "is_spac": is_spac,
    }


def fetch_large_cap_universe(config: UniverseConfig = UniverseConfig()) -> pd.DataFrame:
    """Discover NYSE/Nasdaq equities above the configured market-cap floor.

    yfinance custom screen requests are capped at 250 rows, therefore the
    function paginates until Yahoo reports no further matches.
    """
    query = EquityQuery(
        "and",
        [
            EquityQuery("is-in", ["exchange", *config.exchanges]),
            EquityQuery("gte", ["intradaymarketcap", config.min_market_cap]),
        ],
    )

    rows: list[dict[str, Any]] = []
    offset = 0
    for _ in range(config.max_pages):
        response = yf.screen(
            query,
            offset=offset,
            size=min(config.page_size, 250),
            sortField="intradaymarketcap",
            sortAsc=False,
        )
        quotes = _extract_quotes(response)
        if not quotes:
            break
        rows.extend(_normalize_screen_quote(quote) for quote in quotes)
        offset += len(quotes)
        total = _response_total(response)
        if len(quotes) < min(config.page_size, 250):
            break
        if total is not None and offset >= total:
            break

    if not rows:
        raise RuntimeError("Yahoo screener returned no large-cap equity candidates.")

    universe = pd.DataFrame(rows)
    universe = universe.loc[universe["symbol"].ne("")].copy()
    universe = universe.loc[universe["exchange"].isin(config.exchanges)].copy()
    universe = universe.loc[
        universe["quote_type"].isin(["", "EQUITY", "STOCK"])
    ].copy()
    universe = universe.loc[~universe["is_spac"]].copy()
    universe = universe.loc[
        pd.to_numeric(universe["market_cap"], errors="coerce").ge(config.min_market_cap)
    ].copy()
    universe = (
        universe.sort_values("market_cap", ascending=False)
        .drop_duplicates("symbol", keep="first")
        .reset_index(drop=True)
    )
    return universe


def annotate_sp500_membership(
    universe: pd.DataFrame,
    sp500_constituents: pd.DataFrame | None,
) -> pd.DataFrame:
    """Keep S&P 500 membership as metadata, never as an eligibility rule."""
    result = universe.copy()
    if sp500_constituents is None or sp500_constituents.empty:
        result["is_sp500"] = False
        return result
    symbols = set(
        sp500_constituents["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )
    result["is_sp500"] = result["symbol"].isin(symbols)
    return result


def parse_rating_activity(raw_actions: Any, now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Summarise analyst rating activity without using its bullish/bearish direction."""
    output = {
        "rating_actions_30d": np.nan,
        "rating_actions_90d": np.nan,
        "latest_rating_date": pd.NaT,
        "analyst_activity_freshness_score": 40.0,
    }
    if not isinstance(raw_actions, pd.DataFrame) or raw_actions.empty:
        return output

    frame = raw_actions.copy()
    dates = pd.to_datetime(frame.index, errors="coerce", utc=True)
    if dates.isna().all():
        for candidate in ("GradeDate", "date", "Date"):
            if candidate in frame.columns:
                dates = pd.to_datetime(frame[candidate], errors="coerce", utc=True)
                break
    valid_dates = pd.Series(dates).dropna()
    if valid_dates.empty:
        return output

    latest = pd.Timestamp(valid_dates.max())
    current = now if now is not None else pd.Timestamp.now(tz="UTC")
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    age_days = max((current - latest).total_seconds() / 86_400, 0)
    output["rating_actions_30d"] = int((valid_dates >= current - pd.Timedelta(days=30)).sum())
    output["rating_actions_90d"] = int((valid_dates >= current - pd.Timedelta(days=90)).sum())
    output["latest_rating_date"] = latest
    if age_days <= 30:
        freshness = 100.0
    elif age_days <= 90:
        freshness = 85.0
    elif age_days <= 180:
        freshness = 70.0
    elif age_days <= 365:
        freshness = 50.0
    else:
        freshness = 30.0
    output["analyst_activity_freshness_score"] = freshness
    return output


def _rating_count(frame: pd.DataFrame) -> pd.Series:
    columns = ["strong_buy", "buy", "hold", "sell", "strong_sell"]
    values = pd.DataFrame(index=frame.index)
    for column in columns:
        values[column] = pd.to_numeric(frame.get(column), errors="coerce").fillna(0)
    return values.sum(axis=1)


def _row_data_completeness(row: pd.Series) -> float:
    """Availability-only score: no bullish/bearish values are rewarded."""
    target_available = pd.notna(row.get("target_median")) or pd.notna(row.get("target_mean"))
    target_range_available = pd.notna(row.get("target_low")) and pd.notna(row.get("target_high"))
    recommendations_available = _safe_float(row.get("rating_count")) > 0
    revisions_available = any(
        pd.notna(row.get(column))
        for column in ("eps_up_7d", "eps_up_30d", "eps_down_7d", "eps_down_30d")
    )
    price_available = pd.notna(row.get("latest_quote")) or pd.notna(row.get("history_price"))
    ma200_available = pd.notna(row.get("ma_200"))
    earnings_available = pd.notna(row.get("next_earnings_date"))
    activity_available = pd.notna(row.get("rating_actions_90d"))

    points = (
        20 * target_available
        + 10 * target_range_available
        + 20 * recommendations_available
        + 15 * revisions_available
        + 10 * price_available
        + 10 * ma200_available
        + 5 * earnings_available
        + 10 * activity_available
    )
    return float(points)


def _percentile_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    transformed = np.log1p(numeric.clip(lower=0))
    return transformed.rank(method="average", pct=True).mul(100)


def select_analysis_universe(
    universe: pd.DataFrame,
    price_metrics: pd.DataFrame,
    analyst_data: pd.DataFrame,
    config: UniverseConfig = UniverseConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (diagnostics, eligible universe, selected Top-N constituents).

    Selection is intentionally neutral.  It rewards the *amount, freshness and
    completeness* of available analyst/market data, company size and liquidity;
    it does not reward Buy percentages, target upside, EPS revision direction,
    momentum or any other signal that the downstream investment model ranks.
    """
    frame = (
        universe.merge(price_metrics, on="symbol", how="left", validate="one_to_one")
        .merge(analyst_data, on="symbol", how="left", validate="one_to_one")
    )

    frame["rating_count"] = _rating_count(frame)
    frame["analyst_coverage_band"] = frame["rating_count"].apply(analyst_coverage_band)
    frame["analyst_coverage_confidence"] = frame["rating_count"].apply(
        analyst_coverage_confidence
    )
    frame["current_price_for_universe"] = pd.to_numeric(
        frame.get("latest_quote"), errors="coerce"
    ).combine_first(pd.to_numeric(frame.get("history_price"), errors="coerce"))
    target_median = pd.to_numeric(frame.get("target_median"), errors="coerce")
    target_mean = pd.to_numeric(frame.get("target_mean"), errors="coerce")
    frame["selected_target_for_universe"] = target_median.combine_first(target_mean)
    frame["data_completeness_score"] = frame.apply(_row_data_completeness, axis=1)
    freshness = pd.to_numeric(
        frame.get("analyst_activity_freshness_score"), errors="coerce"
    ).fillna(40.0)
    frame["universe_data_quality_score"] = (
        0.80 * frame["data_completeness_score"] + 0.20 * freshness
    ).clip(0, 100)

    market_cap = pd.to_numeric(frame["market_cap"], errors="coerce")
    liquidity = pd.to_numeric(frame["average_dollar_volume_20d"], errors="coerce")

    failures = pd.Series("", index=frame.index, dtype="object")

    def add_failure(mask: pd.Series, label: str) -> None:
        nonlocal failures
        failures = failures.mask(
            mask,
            failures.where(failures.eq(""), failures + "|") + label,
        )

    add_failure(market_cap.lt(config.min_market_cap) | market_cap.isna(), "MARKET_CAP")
    add_failure(liquidity.lt(config.min_average_dollar_volume) | liquidity.isna(), "LIQUIDITY")
    add_failure(frame["rating_count"].lt(config.min_analyst_count), "ANALYST_COVERAGE")
    add_failure(frame["current_price_for_universe"].isna(), "CURRENT_PRICE")
    add_failure(frame["selected_target_for_universe"].isna(), "ANALYST_TARGET")
    add_failure(
        frame["universe_data_quality_score"].lt(config.min_data_quality_score),
        "DATA_QUALITY",
    )
    if "is_spac" in frame.columns:
        add_failure(frame["is_spac"].fillna(False).astype(bool), "SPAC")

    frame["universe_eligibility_failures"] = failures
    frame["universe_eligible"] = failures.eq("")

    eligible = frame.loc[frame["universe_eligible"]].copy()
    if eligible.empty:
        raise RuntimeError(
            "No stocks passed the large-cap universe rules. Check Yahoo data availability "
            "or relax the configured thresholds."
        )

    eligible["universe_score_analyst_coverage"] = (
        eligible["analyst_coverage_confidence"] * 100
    )
    eligible["universe_score_market_cap"] = _percentile_score(eligible["market_cap"])
    eligible["universe_score_liquidity"] = _percentile_score(
        eligible["average_dollar_volume_20d"]
    )
    eligible["universe_score_data_quality"] = eligible["universe_data_quality_score"]
    eligible["universe_quality_score"] = (
        UNIVERSE_SCORE_WEIGHTS["analyst_coverage"]
        * eligible["universe_score_analyst_coverage"]
        + UNIVERSE_SCORE_WEIGHTS["market_cap"]
        * eligible["universe_score_market_cap"]
        + UNIVERSE_SCORE_WEIGHTS["liquidity"]
        * eligible["universe_score_liquidity"]
        + UNIVERSE_SCORE_WEIGHTS["data_quality"]
        * eligible["universe_score_data_quality"]
    )
    eligible = eligible.sort_values(
        [
            "universe_quality_score",
            "rating_count",
            "market_cap",
            "average_dollar_volume_20d",
            "symbol",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    eligible["analysis_universe_rank"] = np.arange(1, len(eligible) + 1)
    eligible["selected_for_analysis"] = (
        eligible["analysis_universe_rank"] <= config.analysis_universe_size
    )

    score_columns = [
        "symbol",
        "rating_count",
        "analyst_coverage_band",
        "analyst_coverage_confidence",
        "data_completeness_score",
        "universe_data_quality_score",
        "universe_score_analyst_coverage",
        "universe_score_market_cap",
        "universe_score_liquidity",
        "universe_score_data_quality",
        "universe_quality_score",
        "analysis_universe_rank",
        "selected_for_analysis",
    ]
    frame = frame.merge(
        eligible[score_columns],
        on="symbol",
        how="left",
        suffixes=("", "_selected"),
    )
    # Prefer the values computed on the eligible frame if a diagnostic column
    # existed before the merge.
    for column in score_columns[1:]:
        selected_column = f"{column}_selected"
        if selected_column in frame.columns:
            frame[column] = frame[selected_column].combine_first(frame.get(column))
            frame = frame.drop(columns=[selected_column])
    frame["selected_for_analysis"] = frame["selected_for_analysis"].fillna(False).astype(bool)

    selected_symbols = eligible.loc[
        eligible["selected_for_analysis"], "symbol"
    ].tolist()
    base = universe.loc[universe["symbol"].isin(selected_symbols)].copy()
    metadata_columns = [
        "symbol",
        "rating_count",
        "analyst_coverage_band",
        "analyst_coverage_confidence",
        "data_completeness_score",
        "universe_data_quality_score",
        "universe_quality_score",
        "analysis_universe_rank",
        "selected_for_analysis",
    ]
    selected = base.merge(
        eligible[metadata_columns],
        on="symbol",
        how="left",
        validate="one_to_one",
    )
    selected = selected.sort_values("analysis_universe_rank").reset_index(drop=True)

    # Keep the shape expected by the existing analysis pipeline while retaining
    # the richer universe metadata.
    for column in ("headquarters", "date_added", "CIK", "founded"):
        if column not in selected.columns:
            selected[column] = pd.NA

    return frame, eligible, selected
