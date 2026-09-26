from datetime import datetime, timezone
import sqlite3
import pandas as pd
import pytest
from investment_ai.data.symbol_resolver import SymbolResolver
from investment_ai.data.price_store import PriceStore, price_quality
from investment_ai.data.history_store import HistoryStore
from investment_ai.features.analyst import parse_estimates
from investment_ai.features.risk import risk_and_confidence
from investment_ai.health import constituent_source_health, overall_status
import main


def test_symbol_resolution_exceptions_and_unresolved():
    r = SymbolResolver(sqlite3.connect(":memory:"))
    assert (
        r.resolve(
            "STOXX Europe 600", "ATCOa", "Atlas Copco", "Sweden"
        ).canonical_yahoo_symbol
        == "ATCO-A.ST"
    )
    assert (
        r.resolve("STOXX Europe 600", "ERICb", "Ericsson", "Sweden").mapping_status
        == "VERIFIED"
    )
    assert (
        r.resolve("STOXX Europe 600", "HMB", "H&M", "Sweden").canonical_yahoo_symbol
        == "HM-B.ST"
    )
    assert (
        r.resolve(
            "STOXX Europe 600", "SRENH", "Swiss Re", "Switzerland"
        ).canonical_yahoo_symbol
        == "SREN.SW"
    )
    assert (
        r.resolve("STOXX Europe 600", "VOW3", "Volkswagen", "Germany").mapping_status
        == "HEURISTIC"
    )
    assert (
        r.resolve("STOXX Europe 600", "X", "Unknown", "Atlantis").mapping_status
        == "UNRESOLVED"
    )


def test_provider_verification_ambiguity_and_persisted_reuse():
    db = sqlite3.connect(":memory:")
    r = SymbolResolver(db, lambda _: {"name": "Acme Plc", "country": "United Kingdom"})
    first = r.resolve("STOXX Europe 600", "ACME", "Acme Plc", "United Kingdom")
    assert first.mapping_status == "VERIFIED"
    assert (
        r.resolve(
            "STOXX Europe 600", "ACME", "Acme Plc", "United Kingdom"
        ).mapping_method
        == "PERSISTED_VERIFIED"
    )
    ambiguous = SymbolResolver(sqlite3.connect(":memory:"), lambda _: [{}, {}])
    assert (
        ambiguous.resolve("STOXX Europe 600", "ABC", "ABC", "France").mapping_status
        == "AMBIGUOUS"
    )


def test_health_and_partial_policy():
    assert constituent_source_health("S&P 500", 500, 490, 2)["source_status"] == "VALID"
    assert (
        constituent_source_health("STOXX Europe 600", 400, 400, 2)["source_status"]
        == "INVALID"
    )
    assert (
        constituent_source_health("STOXX Europe 600", 600, 560, 80)["source_status"]
        == "DEGRADED"
    )
    assert (
        constituent_source_health("STOXX Europe 600", 600, 530, 2)["source_status"]
        == "INVALID"
    )
    assert overall_status("VALID", "INVALID", "VALID") == "PARTIAL"


def test_price_store_idempotence_incremental_and_quality(tmp_path):
    db = sqlite3.connect(tmp_path / "p.db")
    store = PriceStore(db)
    row = {
        "security_id": "S",
        "symbol": "OLD",
        "date": "2026-01-01",
        "adjusted_close": 10,
    }
    store.upsert([row])
    store.upsert([row])
    assert len(store.history("S")) == 1
    assert (
        store.missing_ranges(
            {"S": "OLD"}, datetime(2025, 1, 1).date(), datetime(2026, 1, 10).date()
        )["S"][0].isoformat()
        == "2025-12-27"
    )
    status, flags = price_quality(pd.DataFrame({"Close": [1, 100]}))
    assert status == "SUSPECT" and "UNIT_JUMP" in flags


def test_exact_timestamp_and_independent_rank_history(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    store.upsert_component(
        "A",
        "targets",
        {"target_mean": 100},
        datetime(2026, 9, 19, 7, 59, tzinfo=timezone.utc),
    )
    store.upsert_component(
        "A",
        "targets",
        {"target_mean": 200},
        datetime(2026, 9, 19, 8, 1, tzinfo=timezone.utc),
    )
    change, _ = store.historical_change(
        "A", "target_mean", 7, 110, datetime(2026, 9, 26, 8, tzinfo=timezone.utc)
    )
    assert change == pytest.approx(10)
    store.save_rankings(
        "a",
        "2026-09-01T00:00:00+00:00",
        [{"symbol": "A", "long_term_rank": 10, "short_term_rank": 20}],
    )
    store.save_rankings(
        "b",
        "2026-09-02T00:00:00+00:00",
        [{"symbol": "A", "long_term_rank": None, "short_term_rank": 15}],
    )
    old = store.changes_by_field("A", 7, datetime(2026, 9, 10, tzinfo=timezone.utc))
    assert old["long_term_rank"] == 10 and old["short_term_rank"] == 15


def test_growth_fallback_confidence_separation_and_replay_diff():
    estimates = pd.DataFrame({"growth": [0.2, 20]}, index=["0y", "+1y"])
    parsed = parse_estimates(estimates, "eps")
    assert (
        parsed["forward_eps_growth"] == 0.2
        and parsed["forward_eps_growth_source"] == "0y"
    )
    stale = {
        f"{name}_cache_status": "STALE_FALLBACK"
        for name in (
            "targets",
            "recommendations",
            "eps_trend",
            "eps_revisions",
            "revenue_estimate",
            "earnings_estimate",
            "rating_actions",
            "earnings_history",
            "earnings_dates",
        )
    }
    stale.update(
        {
            "valuation_cache_status": "STALE_FALLBACK",
            "fundamental_cache_status": "STALE_FALLBACK",
            "price_data_status": "STALE_FALLBACK",
        }
    )
    result = risk_and_confidence(stale)
    assert result["freshness_score"] < result["provider_completeness_score"]
    original = pd.DataFrame(
        {"symbol": ["A"], "long_term_score": [1.0], "short_term_setup": ["PULLBACK"]}
    )
    changed = original.copy()
    changed.loc[0, "long_term_score"] += 1e-4
    assert (
        main.verify_replay_output(original, changed)["replay_verification_status"]
        == "REPLAY_MISMATCH"
    )
