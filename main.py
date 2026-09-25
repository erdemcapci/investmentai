"""Investment AI v3.1 entry point. Run only: ``python main.py``."""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import yfinance as yf
from investment_ai.config import (
    CACHE_DIR,
    EXPORT_RESULTS,
    HISTORY_DB,
    PRICE_PERIOD,
    SCORING_MODEL_VERSION,
    TOP_N,
)
from investment_ai.data.cache import JsonCache
from investment_ai.data.constituents import fetch_index_constituents
from investment_ai.data.history_store import HistoryStore
from investment_ai.data.prices import build_price_features, download_prices
from investment_ai.data.yahoo import YahooClient
from investment_ai.pipeline import build_analysis
from investment_ai.reporting.export import export_run
from investment_ai.reporting.tables import print_rankings
from investment_ai.status import ERROR, FRESH_CACHE, FRESH_PROVIDER, STALE_FALLBACK


def download_combined_constituents() -> pd.DataFrame:
    frame = fetch_index_constituents(CACHE_DIR / "constituents")
    present = {
        membership.strip()
        for value in frame.index_name.dropna()
        for membership in str(value).split("|")
        if membership.strip()
    }
    if not {"S&P 500", "STOXX Europe 600"} <= present:
        raise RuntimeError(
            "Both constituent sources are required for the combined universe"
        )
    return frame


def methodology() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "score": "Long term",
                "formula": "25% Quality + 20% Growth + 20% Valuation + 20% Long-Term Expectations + 10% Long Trend + 5% Financial Safety",
            },
            {
                "score": "Short term",
                "formula": "20% Relative Strength + 25% Setup + 20% Short-Term Expectations + 10% Volume + 15% Technical Trend + 10% Event Timing",
            },
            {
                "score": "Long Expectations",
                "formula": "25% EPS momentum + 15% EPS breadth + 15% revenue momentum + 10% recommendations + 10% rating actions + 10% target momentum + 5% target signal + 5% execution + 5% consistency",
            },
            {"score":"Short Expectations", "formula":"35% short EPS momentum + 20% near-term breadth + 15% short revenue revisions + 10% recommendations + 10% rating actions + 5% target momentum + 5% execution"},
            {
                "score": "Risk",
                "formula": "30% market + 25% applicable balance sheet + 15% event + 20% analyst disagreement + 10% liquidity",
            },
        ]
    )


def _rank_changes(
    frame: pd.DataFrame, store: HistoryStore, started: datetime
) -> pd.DataFrame:
    result = frame.copy()
    values = []
    for row in result.to_dict("records"):
        old = store.changes(row["symbol"], 7, started)
        if not old:
            values.append((np.nan, np.nan, np.nan, np.nan, "NEW"))
            continue
        values.append(
            (
                (
                    row.get("long_term_score") - old["long_term_score"]
                    if pd.notna(row.get("long_term_score"))
                    and old["long_term_score"] is not None
                    else np.nan
                ),
                (
                    row.get("short_term_score") - old["short_term_score"]
                    if pd.notna(row.get("short_term_score"))
                    and old["short_term_score"] is not None
                    else np.nan
                ),
                (
                    old["long_term_rank"] - row.get("long_term_rank")
                    if pd.notna(row.get("long_term_rank"))
                    and old["long_term_rank"] is not None
                    else np.nan
                ),
                (
                    old["short_term_rank"] - row.get("short_term_rank")
                    if pd.notna(row.get("short_term_rank"))
                    and old["short_term_rank"] is not None
                    else np.nan
                ),
                "AVAILABLE",
            )
        )
    result[
        [
            "long_term_score_change_7d",
            "short_term_score_change_7d",
            "long_term_rank_change_7d",
            "short_term_rank_change_7d",
            "rank_history_status",
        ]
    ] = values
    return result


def main() -> None:
    started = datetime.now(timezone.utc)
    run_id = f"{started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    print("Loading combined S&P 500 + STOXX Europe 600 universe...")
    universe = download_combined_constituents()
    symbols = universe.symbol.tolist()
    print(
        f"Downloading two years of daily OHLCV for {len(symbols)} symbols in resilient batches..."
    )
    prices = build_price_features(download_prices(symbols, PRICE_PERIOD), symbols)
    print("Loading cached/fresh Yahoo expectations, valuation, and fundamentals...")
    provider = YahooClient(JsonCache(CACHE_DIR)).fetch_many(universe)
    store = HistoryStore(HISTORY_DB)
    history_rows = []
    for row in provider.to_dict("records"):
        row = store.add_analyst_history_features(row["symbol"], row, started)
        # One component, one authoritative timestamp: stale/cache data never advances history.
        for name in YahooClient.ANALYST_COMPONENTS:
            if row.get(f"{name}_cache_status") == FRESH_PROVIDER:
                fetched_at = row.get(f"{name}_fetched_at_utc")
                if fetched_at:
                    store.upsert_component(row["symbol"], name, row, datetime.fromisoformat(fetched_at))
        history_rows.append(row)
    provider = pd.DataFrame(history_rows)
    full, _, _ = build_analysis(universe, prices, provider)
    full = _rank_changes(full, store, started)
    if "expectations_long_score" not in full and "expectations_score" in full:
        full["expectations_long_score"] = full["expectations_score"]
    if "expectations_short_score" not in full and "expectations_score" in full:
        full["expectations_short_score"] = full["expectations_score"]
    long_term = full[full.long_term_rank.notna()].sort_values("long_term_rank")
    short_term = full[full.short_term_rank.notna()].sort_values("short_term_rank")
    for label, column in (
        ("Expectations", "expectations_long_score"),
        ("Fundamental", "quality_score"),
        ("Valuation", "valuation_score"),
    ):
        coverage = full[column].notna().mean() * 100
        if coverage < 60:
            print(
                f"WARNING: {label} data coverage is only {coverage:.0f}%. Rankings are incomplete and should be treated cautiously."
            )
    price_coverage = prices.get("current_price", pd.Series(dtype=float)).notna().mean() * 100
    warning = " STRONG WARNING" if price_coverage < 90 else " WARNING" if price_coverage < 95 else ""
    print("\nRUN HEALTH")
    print(f"Universe: {len(full):,}")
    print(f"Price coverage: {price_coverage:.1f}%{warning}")
    print(f"Analyst coverage: {full.expectations_long_score.notna().mean()*100:.1f}%")
    print(f"Fundamental Quality coverage: {full.quality_score.notna().mean()*100:.1f}%")
    print(f"Valuation coverage: {full.valuation_score.notna().mean()*100:.1f}%")
    print(f"LT ranked: {len(long_term):,}\nST ranked: {len(short_term):,}")
    print(f"Stale fallbacks: {provider.astype(str).apply(lambda c: c.eq(STALE_FALLBACK)).sum().sum():,}")
    print(f"Provider errors: {provider.astype(str).apply(lambda c: c.eq(ERROR)).sum().sum():,}")
    print_rankings(long_term, short_term, TOP_N)
    store.save_rankings(
        run_id,
        started.isoformat(),
        full[full.long_term_rank.notna() | full.short_term_rank.notna()].to_dict(
            "records"
        ),
    )
    store.close()
    metadata = {
        "run_id": run_id,
        "run_timestamp_utc": started.isoformat(),
        "scoring_model_version": SCORING_MODEL_VERSION,
        "yfinance_version": yf.__version__,
        "universe_count": len(universe),
        "sp500_count": universe.index_name.str.contains("S&P 500", regex=False).sum(),
        "stoxx600_count": universe.index_name.str.contains(
            "STOXX Europe 600", regex=False
        ).sum(),
        "price_coverage": prices.current_price.notna().mean(),
        "ranked_long_term_count": len(long_term),
        "ranked_short_term_count": len(short_term),
    }
    for tier, column in (
        ("analyst", "analyst_cache_status"),
        ("valuation", "valuation_cache_status"),
        ("fundamental", "fundamental_cache_status"),
    ):
        statuses = provider.get(column, pd.Series("", index=provider.index)).astype(str)
        metadata[f"{tier}_cache_hit_count"] = statuses.eq(FRESH_CACHE).sum()
        metadata[f"{tier}_fresh_fetch_count"] = (
            statuses.eq(FRESH_PROVIDER).sum()
            if tier != "analyst"
            else sum(
                provider.get(
                    f"{name}_cache_status", pd.Series("", index=provider.index)
                )
                .eq(FRESH_PROVIDER)
                .sum()
                for name in YahooClient.ANALYST_COMPONENTS
            )
        )
        metadata[f"{tier}_stale_fallback_count"] = statuses.eq(STALE_FALLBACK).sum()
    metadata.update(
        {
            "analyst_component_error_count": provider.get(
                "analyst_component_error_count", pd.Series(0, index=provider.index)
            ).sum(),
            "fundamental_error_count": provider.get(
                "fundamental_cache_status", pd.Series("", index=provider.index)
            )
            .astype(str)
            .str.contains("error")
            .sum(),
            "valuation_error_count": provider.get(
                "valuation_cache_status", pd.Series("", index=provider.index)
            )
            .astype(str)
            .str.contains("error")
            .sum(),
        }
    )
    for pillar in ("quality", "growth", "valuation", "expectations"):
        metadata[f"{pillar}_score_coverage_pct"] = (
            full[f"{pillar}_score"].notna().mean() * 100
        )
    if EXPORT_RESULTS:
        directory = Path("investment_ai_runs") / run_id
        export_run(directory, full, long_term, short_term, metadata, methodology())
        print(f"Exported {directory}")
    print(
        "\nResearch support only: no orders, guarantees, or automated buy/sell advice."
    )


if __name__ == "__main__":
    main()
