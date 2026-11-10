import numpy as np
import pandas as pd

from investment_ai.backtest import strategy_definitions


def _scored():
    return pd.DataFrame({
        "symbol": ["DIP", "HIGH", "BROKEN", "WEAK"],
        "momentum_pillar_raw": [80.0, 80.0, 90.0, 40.0],
        "price_vs_ma200_pct": [20.0, 25.0, -5.0, 10.0],
        "drawdown_from_20d_high_pct": [-9.0, -0.5, -12.0, -9.0],
        "credible_setup": [0.0] * 4,
        "st_price_composite_raw": [50.0] * 4,
        "overextended": [False, True, False, False],
    })


def test_production_strategy_includes_the_pullback_bonus():
    definitions = strategy_definitions(_scored())
    bonus = definitions["production_price_model"] - definitions["production_3_3_0"]
    # HIGH is at its high, BROKEN is below its 200-day average, WEAK lacks momentum.
    assert bonus.round(6).tolist() == [9.0, 0.5, 0.0, 0.0]


def test_skip_overextended_excludes_flagged_stocks():
    skip = strategy_definitions(_scored())["production_skip_overextended"]
    assert np.isnan(skip.iloc[1]) and skip.drop(index=1).notna().all()


def test_pullback_bonus_in_the_short_term_risk_adjusted_score():
    from investment_ai.scoring.action import risk_adjusted_scores
    base = {"short_term_score": 60.0, "long_term_score": 60.0, "risk_score": 50.0,
            "short_rs_score": 85.0, "price_vs_ma200_pct": 15.0}
    dip = risk_adjusted_scores({**base, "drawdown_from_20d_high_pct": -17.0})
    assert dip["pullback_bonus_points"] == 11.25  # capped at 15, live scale 0.75
    assert dip["short_term_risk_adjusted_score"] == 71.25
    crash = risk_adjusted_scores({**base, "drawdown_from_20d_high_pct": -22.0})
    assert crash["pullback_bonus_points"] == 0.0
    assert dip["long_term_risk_adjusted_score"] == 60.0  # long-term is unaffected
    broken = risk_adjusted_scores({**base, "price_vs_ma200_pct": -1.0, "drawdown_from_20d_high_pct": -10.0})
    assert broken["pullback_bonus_points"] == 0.0
    weak = risk_adjusted_scores({**base, "short_rs_score": 65.0, "drawdown_from_20d_high_pct": -10.0})
    assert weak["pullback_bonus_points"] == 0.0
