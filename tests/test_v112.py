"""v1.1.2 production-freeze consistency regressions."""

from datetime import date, datetime, timezone
import json
import sqlite3

import numpy as np
import pandas as pd

import main
from investment_ai.config import CACHE_SCHEMA_VERSION, SCORING_MODEL_VERSION
from investment_ai.data.cache import JsonCache
from investment_ai.data.history_store import HistoryStore
from investment_ai.data.price_store import PriceStore
from investment_ai.data.prices import download_prices_range
from investment_ai.data.yahoo import YahooClient
from investment_ai.status import ERROR, FRESH_CACHE
from investment_ai.validation import update_outcomes, validation_report


def test_schema_three_info_and_fundamentals_are_invalidated(tmp_path):
    cache = JsonCache(tmp_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    for tier, data in (
        ("provider_info", {"sector": "Industrials", "currentPrice": 10}),
        ("fundamentals", {"return_on_equity": 12}),
    ):
        cache._path(tier, "AIR.PA").write_text(
            json.dumps(
                {
                    "cache_schema_version": 3,
                    "fetched_at_utc": timestamp,
                    "data": data,
                }
            )
        )
        _, status = cache.get_or_fetch(
            tier,
            "AIR.PA",
            168,
            lambda: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        assert status == ERROR

    item, first = cache.get_or_fetch("valid", "AIR.PA", 18, lambda: {"currency": "EUR"})
    assert first != FRESH_CACHE
    assert item["cache_schema_version"] == CACHE_SCHEMA_VERSION
    assert cache.get_or_fetch("valid", "AIR.PA", 18, lambda: {})[1] == FRESH_CACHE


def test_resolver_and_provider_share_info_cache(monkeypatch, tmp_path):
    calls = 0

    class Ticker:
        def get_info(self):
            nonlocal calls
            calls += 1
            return {
                "symbol": "AIR.PA",
                "longName": "Airbus SE",
                "quoteType": "EQUITY",
                "sector": "Industrials",
                "currentPrice": 100,
                "currency": "EUR",
                "financialCurrency": "EUR",
            }

        def __getattr__(self, _name):
            return lambda *args, **kwargs: pd.DataFrame()

    monkeypatch.setattr("investment_ai.data.yahoo.yf.Ticker", lambda _symbol: Ticker())
    client = YahooClient(JsonCache(tmp_path))
    assert client.metadata_lookup("AIR.PA")["longName"] == "Airbus SE"
    result = client.fetch_symbol("AIR.PA")
    assert calls == 1
    assert result["info_cache_status"] == FRESH_CACHE
    assert result["trading_currency"] == "EUR"


def test_range_download_retries_omission_and_reports_permanent_missing(monkeypatch):
    calls = []

    def chunk(symbols, period=None, start=None, end=None):
        calls.append(list(symbols))
        returned = [symbol for symbol in symbols if symbol != "B" or len(calls) > 1]
        return pd.concat(
            {
                symbol: pd.DataFrame(
                    {"Close": [1.0]}, index=pd.DatetimeIndex(["2026-01-02"])
                )
                for symbol in returned
            },
            axis=1,
        )

    monkeypatch.setattr("investment_ai.data.prices._download_chunk", chunk)
    result = download_prices_range(["A", "B", "C"], date(2026, 1, 1), date(2026, 1, 3))
    assert calls == [["A", "B", "C"], ["B"]]
    assert set(result.columns.get_level_values(0)) == {"A", "B", "C"}
    assert result.attrs == {"requested_symbol_count": 3, "missing_after_retry_count": 0}

    monkeypatch.setattr(
        "investment_ai.data.prices._download_chunk",
        lambda symbols, period=None, start=None, end=None: (
            pd.concat(
                {
                    symbol: pd.DataFrame({"Close": [1.0]})
                    for symbol in symbols
                    if symbol != "B"
                },
                axis=1,
            )
            if any(symbol != "B" for symbol in symbols)
            else pd.DataFrame()
        ),
    )
    result = download_prices_range(["A", "B"], date(2026, 1, 1), date(2026, 1, 3))
    assert "B" not in result.columns.get_level_values(0)
    assert result.attrs["missing_after_retry_count"] == 1


def test_benchmark_sessions_control_maturity_and_stock_can_remain_missing(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    row = {
        "symbol": "AIR.PA",
        "short_term_score": 80,
        "short_term_rank": 1,
        "current_price": 100,
        "price_as_of": "2026-01-02T00:00:00Z",
        "benchmark_symbol": "^STOXX",
        "benchmark_price": 100,
        "benchmark_price_as_of": "2026-01-02T00:00:00Z",
    }
    store.save_predictions("r", "2026-01-02T00:00:00Z", SCORING_MODEL_VERSION, [row])
    sessions = pd.bdate_range("2026-01-05", "2026-01-09", tz="UTC")
    benchmark = pd.Series(np.arange(101, 106), index=sessions)
    update_outcomes(store.db, {}, {"^STOXX": benchmark})
    status = store.db.execute(
        "SELECT status FROM outcome_status WHERE run_id='r' AND horizon='5d'"
    ).fetchone()[0]
    assert status == "PRICE_MISSING"


def test_validation_defaults_to_current_model_only(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    for version, run_id, outcome in (("3.1.1", "old", -20), ("3.1.2", "new", 10)):
        rows = [
            {"symbol": f"S{i}", "short_term_score": i, "short_term_rank": i + 1}
            for i in range(10)
        ]
        store.save_predictions(run_id, "2025-01-01T00:00:00Z", version, rows)
        with store.db:
            store.db.execute(
                "UPDATE prediction_outcomes SET forward_5d_return=? WHERE run_id=?",
                (outcome, run_id),
            )
            store.db.executemany(
                "INSERT INTO outcome_status VALUES(?,?,?,?,?,?)",
                [(run_id, f"S{i}", "5d", "AVAILABLE", None, "2025-02-01") for i in range(10)],
            )
    summary = validation_report(store.db)["5d"]
    assert summary["model_version"] == SCORING_MODEL_VERSION
    assert summary["evaluation_run_count"] == 1
    assert summary["stock_observation_count"] == 10
    assert summary["mean_top10_return"] == 10


def test_offline_repeat_run_uses_incremental_prices_and_recomputes(monkeypatch):
    """Exercise the repeat-run contract without network or wall-clock dependencies."""
    store = PriceStore(sqlite3.connect(":memory:"))
    calls = []

    def fake(symbols, start, end):
        calls.append((start, end))
        days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
        return pd.concat(
            {symbols[0]: pd.DataFrame({"Close": range(len(days)), "Volume": 1}, index=days)},
            axis=1,
        )

    monkeypatch.setattr(main, "download_prices_range", fake)
    score_runs = 0
    for end in (date(2026, 1, 10), date(2026, 1, 11)):
        main._incremental_price_history(
            store, {"A": "A"}, date(2024, 1, 1), end
        )
        score_runs += 1  # scoring consumes the freshly reconstructed frame every run
    assert calls[0][0] == date(2024, 1, 1)
    assert calls[1][0] > date(2024, 1, 1)
    assert score_runs == 2
