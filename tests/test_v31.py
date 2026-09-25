from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from investment_ai.data.cache import CURRENT_CACHE_SCHEMA_VERSION, JsonCache
from investment_ai.data.history_store import HistoryStore
from investment_ai.features.analyst import expectations_score, parse_estimates
from investment_ai.features.fundamentals import derive_fundamentals
from investment_ai.features.risk import risk_and_confidence
from investment_ai.features.technical import price_features
from investment_ai.features.valuation import add_peer_percentiles
from investment_ai.status import ERROR, FRESH_CACHE, FRESH_PROVIDER, STALE_FALLBACK


def test_component_history_uses_exact_timestamp_and_never_stale(tmp_path):
    store = HistoryStore(tmp_path / "h.db")
    old = datetime(2026, 9, 1, 8, tzinfo=timezone.utc)
    fresh = datetime(2026, 9, 25, 11, 12, tzinfo=timezone.utc)
    store.upsert_component("A", "targets", {"target_median": 100}, old)
    store.upsert_component("A", "eps_trend", {"eps_0q_current": 2}, fresh)
    # Simulating stale target fallback intentionally performs no target write.
    rows = store.db.execute(
        "select component,observed_at_utc,field from analyst_observations order by observed_at_utc"
    ).fetchall()
    assert [(r["component"], r["observed_at_utc"], r["field"]) for r in rows] == [
        ("targets", old.isoformat(), "target_median"),
        ("eps_trend", fresh.isoformat(), "eps_0q_current"),
    ]


def test_cache_schema_and_semantics_gate_fallback(tmp_path):
    cache = JsonCache(tmp_path)
    valid = lambda data: pd.notna(data.get("value"))
    item, status = cache.get_or_fetch("x", "A", 10, lambda: {"value": 2}, validator=valid)
    assert status == FRESH_PROVIDER and item["cache_schema_version"] == CURRENT_CACHE_SCHEMA_VERSION
    assert cache.get_or_fetch("x", "A", 10, lambda: {}, validator=valid)[1] == FRESH_CACHE
    path = cache._path("x", "A")
    path.write_text('{"cache_schema_version":2,"fetched_at_utc":"2026-09-25T00:00:00+00:00","data":{"value":null}}')
    assert cache.get_or_fetch("x", "A", 10, lambda: (_ for _ in ()).throw(RuntimeError()), validator=valid)[1] == ERROR


def test_valid_stale_cache_is_only_allowed_fallback(tmp_path):
    cache = JsonCache(tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    cache._path("x", "A").write_text(
        f'{{"cache_schema_version":2,"fetched_at_utc":"{old}","data":{{"value":1}}}}'
    )
    _, status = cache.get_or_fetch("x", "A", 1, lambda: (_ for _ in ()).throw(RuntimeError()), validator=lambda d: d.get("value") == 1)
    assert status == STALE_FALLBACK


def test_eps_recency_weights_short_more_than_long():
    row = {}
    for period in ("0q", "plus_1q", "0y", "plus_1y"):
        for days, value in ((7, -15), (30, 0), (60, 10), (90, 15)):
            row[f"eps_{period}_change_{days}d_pct"] = value
        for days in (7, 30):
            row[f"eps_revision_breadth_{period}_{days}d"] = 0
    scores = expectations_score(row)
    assert scores["eps_short_horizon_momentum_score"] < scores["eps_long_horizon_momentum_score"]
    assert scores["eps_revision_acceleration"] < 0


def _statements(equity=100, ebitda=50, ebit=20, debt=100, cash=0):
    dates = [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31")]
    income = pd.DataFrame({dates[0]: [ebitda, ebit, 10, 100], dates[1]: [40, 15, 8, 90]}, index=["EBITDA", "EBIT", "NetIncome", "TotalRevenue"])
    balance = pd.DataFrame({dates[0]: [equity, debt, cash, 200], dates[1]: [90, 90, 0, 180]}, index=["StockholdersEquity", "TotalDebt", "CashAndCashEquivalents", "TotalAssets"])
    return income, balance, pd.DataFrame()


def test_negative_financial_denominators_are_never_favorable():
    income, balance, cashflow = _statements(equity=-100, ebitda=-50, ebit=-10)
    result = derive_fundamentals(income, balance, cashflow)
    assert result["negative_equity_flag"] and np.isnan(result["debt_to_equity"])
    assert np.isnan(result["return_on_equity"])
    assert result["negative_ebitda_flag"] and np.isnan(result["net_debt_to_ebitda"])
    assert result["negative_operating_profit_flag"] and np.isnan(result["interest_coverage"])


def test_net_cash_with_positive_ebitda_remains_valid():
    income, balance, cashflow = _statements(ebitda=50, debt=10, cash=100)
    assert derive_fundamentals(income, balance, cashflow)["net_debt_to_ebitda"] < 0


def test_metric_specific_peer_minimum():
    frame = pd.DataFrame({"sector_normalized": ["T"] * 30, "index_name": ["I"] * 30,
                          "forward_pe": list(range(8)) + [np.nan] * 22})
    result = add_peer_percentiles(frame, {"forward_pe": False})
    assert result.forward_pe_peer_method.eq("universe").all()
    assert result.forward_pe_peer_count.eq(8).all()
    frame.loc[:17, "forward_pe"] = range(18)
    result = add_peer_percentiles(frame, {"forward_pe": False})
    assert result.forward_pe_peer_method.eq("sector").all()
    assert result.forward_pe_peer_count.eq(18).all()


def test_today_bar_excluded_and_price_freshness_is_real():
    now = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)
    history = pd.DataFrame({"Close": [100, 101, 999], "Volume": [1, 1, 100]},
                           index=pd.to_datetime(["2026-09-23", "2026-09-24", "2026-09-25"], utc=True))
    result = price_features(history, now)
    assert result["current_price"] == 101
    assert result["technical_price_as_of"].startswith("2026-09-24")
    assert result["price_data_status"] == "FRESH"
    old = history.iloc[:1].copy()
    old.index = pd.to_datetime(["2026-09-01"], utc=True)
    assert price_features(old, now)["price_data_status"] == "INSUFFICIENT"


def test_forward_growth_nan_fallback():
    frame = pd.DataFrame({"growth": [np.nan, .2]}, index=["+1y", "0y"])
    assert parse_estimates(frame, "eps")["forward_eps_growth"] == .2
    frame.loc["+1y", "growth"] = .3
    assert parse_estimates(frame, "eps")["forward_eps_growth"] == .3


def test_risk_exposes_coverage_and_financial_balance_is_unknown():
    result = risk_and_confidence({"is_financial": True, "volatility_60d": 20,
                                  "downside_volatility": 15, "max_drawdown_1y": -20,
                                  "average_dollar_volume_20d": 1e8})
    assert np.isnan(result["balance_sheet_risk_score"])
    assert "risk_coverage" in result and "risk_status" in result
