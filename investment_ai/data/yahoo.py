"""Cached, rate-limit-aware yfinance provider adapter."""

from __future__ import annotations
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
import numpy as np
import pandas as pd
import yfinance as yf
from investment_ai.config import (
    ANALYST_TTL_HOURS,
    FORCE_REFRESH,
    FUNDAMENTALS_TTL_HOURS,
    MAX_WORKERS,
    VALUATION_TTL_HOURS,
)
from investment_ai.data.cache import JsonCache
from investment_ai.features.analyst import (
    parse_actions,
    parse_eps_trend,
    parse_estimates,
    parse_recommendations,
    parse_revisions,
    parse_surprises,
    parse_targets,
)
from investment_ai.features.fundamentals import derive_fundamentals
from investment_ai.features.valuation import parse_valuation

RATE_MARKERS = ("rate limit", "too many requests", "429", "yfratelimit")


def call_with_retry(function: Callable[[], Any], attempts: int = 3) -> Any:
    error = None
    for attempt in range(attempts):
        try:
            return function()
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                rate_limited = any(
                    marker in str(exc).lower() for marker in RATE_MARKERS
                )
                delays = (5, 15, 30) if rate_limited else (0.5, 1, 2)
                time.sleep(delays[attempt] + random.uniform(0, 0.25 * delays[attempt]))
    raise error


def _meaningful(data: dict[str, Any]) -> bool:
    for value in data.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, str) and value:
            return True
        try:
            if pd.notna(float(value)):
                return True
        except (TypeError, ValueError):
            continue
    return False


class YahooClient:
    ANALYST_COMPONENTS = {
        "targets": ("get_analyst_price_targets", parse_targets),
        "recommendations": ("get_recommendations_summary", parse_recommendations),
        "eps_trend": ("get_eps_trend", parse_eps_trend),
        "eps_revisions": ("get_eps_revisions", parse_revisions),
        "earnings_estimate": (
            "get_earnings_estimate",
            lambda value: parse_estimates(value, "eps"),
        ),
        "revenue_estimate": (
            "get_revenue_estimate",
            lambda value: parse_estimates(value, "revenue"),
        ),
        "rating_actions": ("get_upgrades_downgrades", parse_actions),
        "earnings_history": ("get_earnings_history", parse_surprises),
    }

    def __init__(self, cache: JsonCache):
        self.cache = cache

    def _component(
        self,
        ticker: Any,
        symbol: str,
        name: str,
        endpoint: str,
        parser: Callable[[Any], dict[str, Any]],
    ):
        def fetch() -> dict[str, Any]:
            return parser(call_with_retry(getattr(ticker, endpoint)))

        return self.cache.get_or_fetch(
            f"analyst_{name}",
            symbol,
            ANALYST_TTL_HOURS,
            fetch,
            FORCE_REFRESH,
            _meaningful,
        )

    def _earnings_dates(self, ticker: Any) -> dict[str, Any]:
        dates = call_with_retry(lambda: ticker.get_earnings_dates(limit=12))
        if not isinstance(dates, pd.DataFrame) or dates.empty:
            return {}
        now = pd.Timestamp.now(tz="UTC")
        index = pd.to_datetime(dates.index, utc=True, errors="coerce").dropna()
        future, past = index[index >= now], index[index < now]
        return {
            "next_earnings_date": str(future.min()) if len(future) else None,
            "days_to_next_earnings": (
                (future.min() - now).total_seconds() / 86400 if len(future) else np.nan
            ),
            "days_since_last_earnings": (
                (now - past.max()).total_seconds() / 86400 if len(past) else np.nan
            ),
        }

    def fetch_symbol(self, symbol: str, sector: str = "") -> dict[str, Any]:
        ticker = yf.Ticker(symbol)
        result: dict[str, Any] = {"symbol": symbol}
        component_statuses = []
        fetched_times = []
        errors = []
        for name, (endpoint, parser) in self.ANALYST_COMPONENTS.items():
            item, status = self._component(ticker, symbol, name, endpoint, parser)
            result.update(item.get("data", {}))
            result[f"{name}_success"] = status in {"provider", "cache"}
            result[f"{name}_cache_status"] = status
            component_statuses.append(status)
            if item.get("fetched_at_utc"):
                fetched_times.append(item["fetched_at_utc"])
            if "error" in status:
                errors.append(f"{name}:{status}")
        dates, date_status = self.cache.get_or_fetch(
            "analyst_earnings_dates",
            symbol,
            ANALYST_TTL_HOURS,
            lambda: self._earnings_dates(ticker),
            FORCE_REFRESH,
            _meaningful,
        )
        result.update(dates.get("data", {}))
        result["earnings_dates_success"] = date_status in {"provider", "cache"}
        component_statuses.append(date_status)
        if dates.get("fetched_at_utc"):
            fetched_times.append(dates["fetched_at_utc"])
        info, info_status = self.cache.get_or_fetch(
            "analyst_info",
            symbol,
            ANALYST_TTL_HOURS,
            lambda: {
                key: value
                for key, value in (call_with_retry(ticker.get_info) or {}).items()
                if key in {"sector", "currentPrice", "regularMarketPrice"}
            },
            FORCE_REFRESH,
            _meaningful,
        )
        info_data = info.get("data", {})
        result["sector_raw_yahoo"] = info_data.get("sector")
        result["current_price_provider"] = info_data.get(
            "currentPrice", info_data.get("regularMarketPrice")
        )
        component_statuses.append(info_status)
        result["analyst_fetched_at_utc"] = max(fetched_times) if fetched_times else None
        result["analyst_cache_status"] = (
            "STALE_FALLBACK"
            if any(status.startswith("stale") for status in component_statuses)
            else (
                "PARTIAL"
                if any(status == "error" for status in component_statuses)
                else (
                    "FRESH"
                    if any(status == "provider" for status in component_statuses)
                    else "CACHE"
                )
            )
        )

        valuation, valuation_status = self.cache.get_or_fetch(
            "valuation",
            symbol,
            VALUATION_TTL_HOURS,
            lambda: parse_valuation(
                call_with_retry(
                    lambda: ticker.get_valuation_measures(freq="quarterly", periods=12)
                )
            ),
            FORCE_REFRESH,
            _meaningful,
        )
        fundamentals, fundamental_status = self.cache.get_or_fetch(
            "fundamentals",
            symbol,
            FUNDAMENTALS_TTL_HOURS,
            lambda: self._fundamentals(ticker, sector),
            FORCE_REFRESH,
            _meaningful,
        )
        result.update(valuation.get("data", {}))
        result.update(fundamentals.get("data", {}))
        result.update(
            {
                "valuation_cache_status": valuation_status,
                "fundamental_cache_status": fundamental_status,
                "valuation_fetched_at_utc": valuation.get("fetched_at_utc"),
                "fundamentals_fetched_at_utc": fundamentals.get("fetched_at_utc"),
            }
        )
        errors.extend(
            status
            for status in (valuation_status, fundamental_status)
            if "error" in status
        )
        result["analyst_component_error_count"] = sum(
            "error" in status for status in component_statuses
        )
        result["provider_success"] = np.mean(
            [
                "error" not in status
                for status in component_statuses
                + [valuation_status, fundamental_status]
            ]
        )
        result["data_errors"] = " | ".join(errors)
        return result

    @staticmethod
    def _fundamentals(ticker: Any, sector: str) -> dict[str, Any]:
        # Annual statements only: quarterly calls were previously discarded.
        income = call_with_retry(lambda: ticker.get_income_stmt(freq="yearly"))
        balance = call_with_retry(lambda: ticker.get_balance_sheet(freq="yearly"))
        cashflow = call_with_retry(lambda: ticker.get_cash_flow(freq="yearly"))
        return derive_fundamentals(income, balance, cashflow, sector)

    def fetch_many(self, universe: pd.DataFrame) -> pd.DataFrame:
        rows = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            jobs = {
                pool.submit(
                    self.fetch_symbol, row.symbol, getattr(row, "sector", "")
                ): row.symbol
                for row in universe.itertuples()
            }
            for future in as_completed(jobs):
                try:
                    rows.append(future.result())
                except Exception as exc:
                    rows.append(
                        {
                            "symbol": jobs[future],
                            "data_errors": str(exc),
                            "provider_success": 0,
                        }
                    )
        return pd.DataFrame(rows)
