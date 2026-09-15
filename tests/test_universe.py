import math

import numpy as np
import pandas as pd

from universe import (
    UniverseConfig,
    analyst_coverage_band,
    analyst_coverage_confidence,
    annotate_sp500_membership,
    parse_rating_activity,
    select_analysis_universe,
)


def test_analyst_coverage_bands_and_confidence_match_policy():
    cases = [
        (4, "INSUFFICIENT", 0.50),
        (5, "LOW", 0.70),
        (7, "LOW", 0.70),
        (8, "ACCEPTABLE", 0.85),
        (14, "ACCEPTABLE", 0.85),
        (15, "STRONG", 0.95),
        (24, "STRONG", 0.95),
        (25, "VERY_STRONG", 1.00),
        (40, "VERY_STRONG", 1.00),
    ]
    for count, band, confidence in cases:
        assert analyst_coverage_band(count) == band
        assert math.isclose(analyst_coverage_confidence(count), confidence)


def test_sp500_membership_is_metadata_not_filter():
    universe = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "company_name": ["A", "B"],
        }
    )
    sp500 = pd.DataFrame({"symbol": ["AAA"]})
    annotated = annotate_sp500_membership(universe, sp500)
    assert annotated["symbol"].tolist() == ["AAA", "BBB"]
    assert annotated.set_index("symbol").loc["AAA", "is_sp500"]
    assert not annotated.set_index("symbol").loc["BBB", "is_sp500"]


def _universe_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "source_symbol": ["AAA", "BBB", "CCC", "DDD"],
            "company_name": ["A", "B", "C", "D"],
            "sector": ["Technology"] * 4,
            "industry": ["Software"] * 4,
            "sub_industry": ["Software"] * 4,
            "exchange": ["NMS"] * 4,
            "quote_type": ["EQUITY"] * 4,
            "market_cap": [50e9, 50e9, 40e9, 60e9],
            "screener_price": [100.0, 100.0, 80.0, 120.0],
            "screener_average_volume_3m": [1e6] * 4,
            "screener_average_dollar_volume_3m": [100e6, 100e6, 80e6, 120e6],
            "is_adr_guess": [False] * 4,
            "is_spac": [False] * 4,
            "is_sp500": [True, False, False, True],
        }
    )


def _price_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "history_price": [100.0, 100.0, 80.0, 120.0],
            "ma_200": [np.nan] * 4,
            "average_dollar_volume_20d": [100e6, 100e6, 80e6, 120e6],
        }
    )


def _analyst_fixture() -> pd.DataFrame:
    # AAA and BBB deliberately have opposite recommendation mixes while keeping
    # the same analyst count and data availability.  Universe selection must not
    # reward the bullish mix because direction belongs to downstream scoring.
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "latest_quote": [100.0, 100.0, 80.0, 120.0],
            "target_low": [90.0, 90.0, 70.0, 110.0],
            "target_mean": [110.0, 110.0, 90.0, 130.0],
            "target_median": [108.0, 108.0, 88.0, 128.0],
            "target_high": [130.0, 130.0, 110.0, 150.0],
            "strong_buy": [10, 0, 4, 5],
            "buy": [10, 0, 3, 5],
            "hold": [0, 0, 0, 10],
            "sell": [0, 10, 0, 5],
            "strong_sell": [0, 10, 0, 5],
            "eps_up_7d": [1, 0, 1, 1],
            "eps_up_30d": [2, 0, 1, 2],
            "eps_down_7d": [0, 1, 0, 1],
            "eps_down_30d": [0, 2, 0, 1],
            "next_earnings_date": pd.to_datetime(
                ["2026-10-01", "2026-10-01", "2026-10-01", "2026-10-01"],
                utc=True,
            ),
            "rating_actions_90d": [4, 4, 3, 5],
            "analyst_activity_freshness_score": [100.0, 100.0, 85.0, 100.0],
        }
    )


def test_universe_selection_uses_coverage_not_recommendation_direction():
    config = UniverseConfig(
        min_market_cap=10e9,
        min_average_dollar_volume=20e6,
        min_analyst_count=8,
        min_data_quality_score=50,
        analysis_universe_size=4,
    )
    _, eligible, _ = select_analysis_universe(
        _universe_fixture(),
        _price_fixture(),
        _analyst_fixture(),
        config,
    )
    scores = eligible.set_index("symbol")["universe_quality_score"]
    assert math.isclose(scores["AAA"], scores["BBB"])


def test_less_than_eight_analysts_is_not_eligible():
    config = UniverseConfig(
        min_market_cap=10e9,
        min_average_dollar_volume=20e6,
        min_analyst_count=8,
        min_data_quality_score=50,
        analysis_universe_size=3,
    )
    diagnostics, eligible, selected = select_analysis_universe(
        _universe_fixture(),
        _price_fixture(),
        _analyst_fixture(),
        config,
    )
    assert "CCC" not in set(eligible["symbol"])
    assert "CCC" not in set(selected["symbol"])
    failure = diagnostics.set_index("symbol").loc["CCC", "universe_eligibility_failures"]
    assert "ANALYST_COVERAGE" in failure


def test_top_n_selection_is_dynamic_and_exact_when_enough_eligible():
    config = UniverseConfig(
        min_market_cap=10e9,
        min_average_dollar_volume=20e6,
        min_analyst_count=8,
        min_data_quality_score=50,
        analysis_universe_size=2,
    )
    _, eligible, selected = select_analysis_universe(
        _universe_fixture(),
        _price_fixture(),
        _analyst_fixture(),
        config,
    )
    assert len(eligible) == 3
    assert len(selected) == 2
    assert selected["analysis_universe_rank"].tolist() == [1, 2]


def test_rating_activity_freshness_is_availability_only():
    now = pd.Timestamp("2026-09-15T12:00:00Z")
    actions = pd.DataFrame(
        {"action": ["up", "down", "main"]},
        index=pd.to_datetime(
            ["2026-09-10", "2026-08-15", "2026-05-01"], utc=True
        ),
    )
    metrics = parse_rating_activity(actions, now=now)
    assert metrics["rating_actions_30d"] == 1
    assert metrics["rating_actions_90d"] == 2
    assert metrics["analyst_activity_freshness_score"] == 100.0
