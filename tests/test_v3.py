import math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from investment_ai.scoring.common import change_pct, weighted, INSUFFICIENT_DATA
from investment_ai.features.analyst import (
    parse_eps_trend,
    parse_recommendations,
    parse_revisions,
    parse_actions,
    parse_surprises,
)
from investment_ai.features.fundamentals import derive_fundamentals
from investment_ai.features.valuation import parse_valuation, add_peer_percentiles
from investment_ai.features.technical import (
    price_features,
    setup_scores,
    event_timing_score,
    add_relative_strength,
)
from investment_ai.data.cache import JsonCache
from investment_ai.data.history_store import HistoryStore
from investment_ai.scoring.ranking import rank_results


def test_change_denominators():
    assert (
        change_pct(-1, -2) == 50
        and math.isnan(change_pct(1, 0))
        and math.isnan(change_pct(1, np.nan))
    )


def test_eps_trend_all_windows():
    f = pd.DataFrame(
        {
            "current": [1.1],
            "7daysAgo": [1],
            "30daysAgo": [0.9],
            "60daysAgo": [0.8],
            "90daysAgo": [0.7],
        },
        index=["0q"],
    )
    r = parse_eps_trend(f)
    assert (
        r["eps_0q_change_7d_pct"] > 0
        and r["eps_0q_change_90d_pct"] > r["eps_0q_change_7d_pct"]
    )


def test_eps_deterioration():
    assert (
        parse_eps_trend(
            pd.DataFrame({"current": [0.8], "30daysAgo": [1]}, index=["0q"])
        )["eps_0q_change_30d_pct"]
        < 0
    )


def test_recommendation_trend():
    f = pd.DataFrame(
        {
            "strongBuy": [8, 4],
            "buy": [2, 2],
            "hold": [0, 4],
            "sell": [0, 0],
            "strongSell": [0, 0],
        },
        index=["0m", "-1m"],
    )
    assert parse_recommendations(f)["positive_rating_change_1m_pp"] > 0


def test_revision_breadth_unknown_zero():
    f = pd.DataFrame({"upLast7days": [0], "downLast7days": [0]}, index=["0q"])
    assert math.isnan(parse_revisions(f)["eps_revision_breadth_0q_7d"])


def test_actions_balance():
    now = pd.Timestamp.now(tz="UTC")
    f = pd.DataFrame(
        {"Action": ["up", "down", "up"]},
        index=[now - pd.Timedelta(days=x) for x in [1, 2, 3]],
    )
    assert parse_actions(f, now)["rating_action_balance_7d"] == 1 / 3


def test_surprises_need_coverage():
    assert parse_surprises(pd.DataFrame({"surprisePercent": [0.1]}, index=[0])) == {}


def test_fcf_and_cash_conversion():
    i = pd.DataFrame(
        {0: [100, 10, -5]}, index=["Total Revenue", "Operating Income", "Net Income"]
    )
    b = pd.DataFrame(
        {0: [20, 10, 30, 100]},
        index=[
            "Cash And Cash Equivalents",
            "Total Debt",
            "Stockholders Equity",
            "Total Assets",
        ],
    )
    c = pd.DataFrame(
        {0: [15, -5]}, index=["Operating Cash Flow", "Capital Expenditure"]
    )
    r = derive_fundamentals(i, b, c)
    assert (
        r["free_cash_flow"] == 10
        and r["fcf_margin_pct"] == 10
        and math.isnan(r["cash_conversion"])
    )


def test_margin_trend_and_missing_rows():
    i = pd.DataFrame(
        {0: [100, 20], 1: [100, 10]}, index=["Total Revenue", "Operating Income"]
    )
    r = derive_fundamentals(i, pd.DataFrame(), pd.DataFrame())
    assert r["operating_margin_change_1y"] == 10 and math.isnan(r["free_cash_flow"])


def test_financial_applicability():
    r = derive_fundamentals(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Financials"
    )
    assert (
        r["is_financial"]
        and not r["roic_applicable"]
        and math.isnan(r["net_debt_to_ebitda"])
    )


def test_valuation_parser_invalid_pe_and_history():
    f = pd.DataFrame(
        {
            "Current": [-2, 10],
            "Q1": [8, 12],
            "Q2": [9, 13],
            "Q3": [10, 14],
            "Q4": [11, 15],
        },
        index=["Forward P/E", "Price/Book"],
    )
    r = parse_valuation(f)
    assert r["forward_pe"] == 8 and pd.notna(r["price_book_own_history_percentile"])


def test_percentile_direction_and_outlier():
    f = pd.DataFrame({"sector": ["x"] * 15, "forward_pe": list(range(1, 15)) + [1000]})
    r = add_peer_percentiles(f, {"forward_pe": False})
    assert r.iloc[0].forward_pe_peer_percentile > r.iloc[-1].forward_pe_peer_percentile


def _history(n=260):
    idx = pd.date_range("2025-01-01", periods=n)
    close = pd.Series(np.linspace(80, 100, n), index=idx)
    return pd.DataFrame(
        {"Close": close, "Volume": np.r_[np.ones(n - 1) * 100, 200]}, index=idx
    )


def test_technical_features():
    r = price_features(_history())
    assert (
        r["relative_volume_1d"] > 1
        and -1 <= r["max_drawdown_1y"] <= 0
        and pd.notna(r["return_252d_pct"])
    )


def test_relative_strength():
    f = pd.DataFrame(
        {
            "sector": ["x"] * 15,
            **{f"return_{n}d_pct": range(15) for n in (20, 60, 126, 252)},
        }
    )
    r = add_relative_strength(f)
    assert r.iloc[-1].rs_20d_percentile > r.iloc[0].rs_20d_percentile


def test_pullback_breakout_and_overextension():
    base = {
        "drawdown_from_20d_high_pct": -5,
        "drawdown_from_60d_high_pct": -5,
        "return_1d_pct": 0,
        "return_5d_pct": -4,
        "return_20d_pct": 5,
        "return_60d_pct": 15,
        "price_vs_ma20_pct": 2,
        "price_vs_ma50_pct": 5,
        "price_vs_ma200_pct": 10,
        "relative_volume_5d": 0.7,
        "relative_volume_1d": 1.5,
        "rs_60d_percentile": 80,
    }
    assert setup_scores(base)["short_term_setup"] in {"PULLBACK", "MIXED"}
    over = {**base, "return_5d_pct": 10, "return_20d_pct": 50, "price_vs_ma20_pct": 40}
    assert (
        setup_scores(over)["breakout_momentum_setup_score"]
        < setup_scores(base)["breakout_momentum_setup_score"]
    )


def test_earnings_timing():
    assert event_timing_score(2) < event_timing_score(30)


def test_minimum_coverage_and_missing_not_zero():
    s, c, status = weighted({"a": 80, "b": np.nan}, {"a": 0.4, "b": 0.6}, 0.6)
    assert math.isnan(s) and status == INSUFFICIENT_DATA


def test_scores_bounded():
    assert weighted({"a": 200, "b": -10}, {"a": 0.5, "b": 0.5})[0] == 95


def test_numeric_ranking_and_ties():
    f = pd.DataFrame(
        [
            {
                "symbol": "a",
                "long_term_score": 80,
                "short_term_score": 70,
                "confidence_score": 50,
                "expectations_score": 50,
                "quality_score": 50,
                "short_rs_score": 50,
                "short_term_setup": "PULLBACK",
            },
            {
                "symbol": "b",
                "long_term_score": 81,
                "short_term_score": 70,
                "confidence_score": 40,
                "expectations_score": 50,
                "quality_score": 50,
                "short_rs_score": 50,
                "short_term_setup": "BREAKOUT",
            },
        ]
    )
    lt, st = rank_results(f)
    assert lt.symbol.tolist() == ["b", "a"] and st.symbol.tolist() == ["a", "b"]


def test_cache_preserves_success(tmp_path):
    c = JsonCache(tmp_path)
    item, _ = c.get_or_fetch("a", "X", 0, lambda: {"x": 1})
    old, _ = c.get_or_fetch(
        "a", "X", 0, lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    )
    assert old["data"]["x"] == 1


def test_snapshot_upsert_and_missing_history(tmp_path):
    h = HistoryStore(tmp_path / "h.db")
    now = datetime.now(timezone.utc)
    h.upsert_analyst("X", {"target_mean": 10}, now)
    h.upsert_analyst("X", {"target_mean": 11}, now)
    assert h.db.execute("select count(*) from analyst_snapshots").fetchone()[0] == 1
    v, s = h.historical_change("X", "target_mean", 30, 12)
    assert math.isnan(v) and s == "HISTORY_NOT_YET_AVAILABLE"
    h.close()


def test_history_available(tmp_path):
    h = HistoryStore(tmp_path / "h.db")
    h.upsert_component(
        "X", "targets", {"target_mean": 10}, datetime.now(timezone.utc) - timedelta(days=31)
    )
    v, s = h.historical_change("X", "target_mean", 30, 12)
    assert v == 20 and s == "AVAILABLE"
