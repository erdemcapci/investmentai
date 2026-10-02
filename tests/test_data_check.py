import numpy as np
import pandas as pd

from investment_ai.runtime import horizon_run_statuses
from investment_ai.scoring.action import build_action_list
from investment_ai.scoring.data_check import CHECK, INCOMPLETE, OK, stock_data_check

BASE = {
    "current_price": 100.0,
    "price_data_status": "FRESH",
    "stop_loss_price": 90.0,
    "short_term_rank": 3,
    "long_term_rank": 10,
    "eps_short_horizon_momentum_score": 60.0,
    "days_to_next_earnings": 40.0,
    "market_regime": "RISK_ON",
    "eps_trend_cache_status": "FRESH_PROVIDER",
    "fundamental_cache_status": "FRESH_CACHE",
}


def test_complete_stock_is_ok():
    assert stock_data_check(BASE) == {"data_check_status": OK, "data_check_notes": ""}


def test_outdated_price_or_missing_stop_is_incomplete():
    stale = stock_data_check({**BASE, "price_data_status": "STALE"})
    assert stale["data_check_status"] == INCOMPLETE and "price is not current" in stale["data_check_notes"]
    no_stop = stock_data_check({**BASE, "stop_loss_price": np.nan})
    assert no_stop["data_check_status"] == INCOMPLETE and "no stop level" in no_stop["data_check_notes"]
    assert stock_data_check({**BASE, "current_price": np.nan})["data_check_status"] == INCOMPLETE


def test_stock_specific_gaps_ask_for_a_check():
    for change, text in (
        ({"days_to_next_earnings": np.nan}, "no next earnings date"),
        ({"days_to_next_earnings": 6.0}, "earnings in ~6 days"),
        ({"eps_trend_cache_status": "STALE_FALLBACK"}, "analyst data not refreshed today: EPS estimate trend"),
        ({"eps_short_horizon_momentum_score": np.nan}, "no EPS estimate trend"),
        ({"fundamental_cache_status": "ERROR"}, "fundamentals not refreshed"),
        ({"market_regime": "UNKNOWN"}, "market trend unknown"),
    ):
        result = stock_data_check({**BASE, **change})
        assert result["data_check_status"] == CHECK, change
        assert text in result["data_check_notes"], change


def test_gaps_shared_by_every_stock_are_not_flagged():
    # Revenue revisions and target changes need stored history; short interest
    # is absent for European listings. None of these is stock-specific.
    row = {**BASE, "revenue_revision_short_score": np.nan, "target_momentum_score": np.nan,
           "positioning_score": np.nan, "earnings_drift_score": np.nan,
           "expectations_short_coverage": 0.7}
    assert stock_data_check(row)["data_check_status"] == OK


def test_incomplete_stocks_never_reach_the_action_list():
    frame = pd.DataFrame(
        {
            "symbol": ["GOOD", "STALE"] + [f"F{i}" for i in range(4)],
            "sector": ["A", "B", "C", "C", "C", "C"],
            "long_term_rank": [1, 2, 3, 4, 5, 6],
            "short_term_rank": [1, 2, np.nan, np.nan, np.nan, np.nan],
            "short_term_risk_adjusted_score": [70, 90, np.nan, np.nan, np.nan, np.nan],
            "data_check_status": [OK, INCOMPLETE, OK, OK, OK, OK],
        }
    )
    assert build_action_list(frame).symbol.tolist() == ["GOOD"]


def test_run_status_ignores_coverage_below_target_but_keeps_floors():
    metrics = {
        "price_fresh_coverage_pct": 100,
        "quality_coverage_pct": 96, "growth_coverage_pct": 96, "valuation_coverage_pct": 92,
        "long_expectations_coverage_pct": 71.5, "lt_score_coverage_pct": 65.3,
        "short_expectations_coverage_pct": 72.2, "short_rs_coverage_pct": 99,
        "setup_data_coverage_pct": 99, "technical_data_coverage_pct": 99,
    }
    health = horizon_run_statuses(metrics, universe_valid="DEGRADED")
    assert health["overall_run_status"] == "VALID"
    assert any("lt_score_coverage_pct" in note for note in health["coverage_notes"])
    assert "constituent universe health degraded" in health["coverage_notes"]
    collapsed = horizon_run_statuses({**metrics, "short_expectations_coverage_pct": 40})
    assert collapsed["st_run_status"] == "INVALID"


def test_run_status_degrades_when_prices_are_outdated():
    metrics = {key: 99 for key in (
        "quality_coverage_pct", "growth_coverage_pct", "valuation_coverage_pct",
        "long_expectations_coverage_pct", "lt_score_coverage_pct", "short_expectations_coverage_pct",
        "short_rs_coverage_pct", "setup_data_coverage_pct", "technical_data_coverage_pct")}
    health = horizon_run_statuses({**metrics, "price_fresh_coverage_pct": 93})
    assert health["overall_run_status"] == "DEGRADED"
    assert "price_fresh_coverage_pct" in health["common_run_status_reasons"][0]
