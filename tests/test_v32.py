"""Scoring model 3.2.0: signal fixes, earnings drift, TTM, risk layer, backtest."""

import math

import numpy as np
import pandas as pd
import pytest

from investment_ai.backtest import run_backtest
from investment_ai.features.benchmark import (
    REBOUND_RISK,
    RISK_OFF,
    RISK_ON,
    REGIME_UNKNOWN,
    market_regime,
)
from investment_ai.features.fundamentals import (
    derive_ttm_fundamentals,
    latest_quarter_revenue_growth,
    merge_fundamentals,
)
from investment_ai.features.peers import fallback_percentile
from investment_ai.features.technical import (
    add_model_relative_strength,
    earnings_drift_score,
    earnings_reaction,
    model_rs,
    price_features,
    setup_scores,
)
from investment_ai.pipeline import add_earnings_drift, add_positioning_features
from investment_ai.scoring.action import (
    build_action_list,
    risk_adjusted_scores,
    trade_plan,
)
from investment_ai.scoring.common import weighted
from investment_ai.scoring.ranking import rank_results
from investment_ai.scoring.short_term import momentum_pillar, volume_pillar

PULLBACK = {
    "drawdown_from_20d_high_pct": -6,
    "drawdown_from_60d_high_pct": -8,
    "return_1d_pct": 0,
    "return_5d_pct": -4,
    "return_20d_pct": 2,
    "return_60d_pct": 15,
    "price_vs_ma20_pct": -1,
    "price_vs_ma50_pct": 5,
    "price_vs_ma200_pct": 12,
    "relative_volume_5d": 0.7,
    "relative_volume_1d": 0.8,
    "rs_60d_percentile": 80,
}


def test_neutral_fill_shrinks_missing_components_toward_fifty():
    renormalised = weighted({"a": 90, "b": np.nan}, {"a": 0.5, "b": 0.5})[0]
    filled, coverage, _ = weighted({"a": 90, "b": np.nan}, {"a": 0.5, "b": 0.5}, neutral_fill=50)
    assert renormalised == 90
    assert filled == pytest.approx(70)
    assert coverage == 0.5


def test_one_day_return_is_counted_once_in_pullback():
    crash = setup_scores({**PULLBACK, "return_1d_pct": -8})["pullback_setup_score"]
    calm = setup_scores({**PULLBACK, "return_1d_pct": 0.5})["pullback_setup_score"]
    # Only the 15% stability weight may move with the one-day return.
    assert calm - crash == pytest.approx(15)


def test_pullback_requires_an_established_uptrend():
    assert setup_scores(PULLBACK)["short_term_setup"] in {"PULLBACK", "MIXED"}
    downtrend = setup_scores({**PULLBACK, "price_vs_ma200_pct": -5})
    assert downtrend["short_term_setup"] not in {"PULLBACK", "MIXED"}
    weak = setup_scores({**PULLBACK, "rs_60d_percentile": 30})
    assert weak["short_term_setup"] not in {"PULLBACK", "MIXED"}


def test_momentum_pillar_is_led_by_skip_month_momentum():
    base = {"rs_60d_percentile": 50, "rs_mom_6_1_percentile": 50}
    strong = momentum_pillar({**base, "rs_mom_12_1_percentile": 90})[0]
    weak = momentum_pillar({**base, "rs_mom_12_1_percentile": 10})[0]
    assert strong - weak == pytest.approx(40)
    # The one-month return no longer moves the pillar in either direction.
    assert momentum_pillar({**base, "rs_20d_percentile": 99}) == momentum_pillar(base)
    # A short listing history still scores from the 60-day term alone.
    assert momentum_pillar({"rs_60d_percentile": 80})[0] == pytest.approx(80)


def test_skip_month_momentum_ignores_the_latest_month():
    dates = pd.bdate_range("2024-01-01", periods=300, tz="UTC")
    close = pd.Series(np.linspace(100, 130, 300), index=dates)
    spiked = close.copy()
    spiked.iloc[-21:] *= 1.5
    now = dates[-1] + pd.Timedelta(days=1)
    plain = price_features(pd.DataFrame({"Close": close, "Volume": 1e6}), now=now)
    jumped = price_features(pd.DataFrame({"Close": spiked, "Volume": 1e6}), now=now)
    assert plain["momentum_12_1_pct"] == pytest.approx(jumped["momentum_12_1_pct"])
    assert plain["momentum_6_1_pct"] == pytest.approx(jumped["momentum_6_1_pct"])
    assert jumped["return_252d_pct"] > plain["return_252d_pct"]


def test_model_relative_strength_basis_and_fallback():
    frame = pd.DataFrame(
        {
            "rs_60d_percentile": [10.0, 20.0],
            "benchmark_rs_60d_percentile": [90.0, np.nan],
        }
    )
    benchmark = add_model_relative_strength(frame, "benchmark")
    assert benchmark.model_rs_60d_percentile.tolist() == [90.0, 20.0]
    assert benchmark.rs_basis.tolist() == ["BENCHMARK_EXCESS", "RAW_FALLBACK"]
    raw = add_model_relative_strength(frame, "raw")
    assert raw.model_rs_60d_percentile.tolist() == [10.0, 20.0]
    assert model_rs({"rs_60d_percentile": 40}, "60d") == 40


def test_fallback_percentile_uses_first_level_with_enough_peers():
    values = pd.Series([1, 2, 3, 10, 20])
    sector = pd.Series(["A", "A", "A", "B", "B"])
    result = fallback_percentile(values, [sector], minimum_peers=3)
    assert result.iloc[:3].tolist() == [0, 50, 100]
    # Sector B is too small, so its members are ranked against the universe.
    assert result.iloc[3] == 75 and result.iloc[4] == 100


def test_earnings_reaction_covers_pre_open_and_post_close_reports():
    dates = pd.bdate_range("2026-01-05", periods=6, tz="UTC")
    stock = pd.Series([100, 100, 110, 121, 121, 121], index=dates, dtype=float)
    bench = pd.Series([100, 100, 101, 102, 102, 102], index=dates, dtype=float)
    result = earnings_reaction(stock, bench, "2026-01-07 21:00:00+00:00")
    # Window: close before the report day (100) to the next session (121).
    assert result["earnings_reaction_pct"] == pytest.approx(21)
    assert result["earnings_reaction_excess_pct"] == pytest.approx(19)
    assert math.isnan(earnings_reaction(stock, bench, None)["earnings_reaction_pct"])


def test_earnings_drift_decays_to_neutral_and_expires():
    fresh = earnings_drift_score(15, 8, 10)
    fading = earnings_drift_score(15, 8, 67.5)
    assert fresh > fading > 50
    assert earnings_drift_score(15, 8, 90) == pytest.approx(50)
    assert math.isnan(earnings_drift_score(15, 8, 120))
    assert earnings_drift_score(-15, -8, 10) < 50


def test_earnings_drift_is_measured_from_the_last_completed_close():
    frame = pd.DataFrame(
        {
            "price_as_of": ["2026-02-20T00:00:00+00:00"],
            "last_earnings_date": ["2026-02-10 21:00:00+00:00"],
            "last_earnings_surprise_pct": [12.0],
            "earnings_reaction_excess_pct": [6.0],
        }
    )
    result = add_earnings_drift(frame).iloc[0]
    assert result.days_since_earnings_asof == pytest.approx(9.125)
    assert result.earnings_drift_score > 70


def test_positioning_features_convert_fraction_and_change():
    frame = pd.DataFrame(
        {
            "provider_info_shortPercentOfFloat": [0.05],
            "provider_info_sharesShort": [120.0],
            "provider_info_sharesShortPriorMonth": [100.0],
        }
    )
    result = add_positioning_features(frame).iloc[0]
    assert result.short_interest_pct_float == pytest.approx(5)
    assert result.short_interest_change_pct == pytest.approx(20)


def test_volume_pillar_is_signed_by_direction():
    accumulation = volume_pillar({"relative_volume_5d": 2, "return_5d_pct": 3})
    distribution = volume_pillar({"relative_volume_5d": 2, "return_5d_pct": -3})
    quiet_pullback = volume_pillar({"relative_volume_5d": 0.7, "return_5d_pct": -3})
    assert accumulation > 50 > distribution
    assert quiet_pullback > 50


def _quarterly(values_by_label, quarters):
    dates = pd.date_range("2024-03-31", periods=quarters, freq="QE")[::-1]
    return pd.DataFrame(
        {date: {label: values[i] for label, values in values_by_label.items()}
         for i, date in enumerate(dates)}
    )


def test_ttm_fundamentals_sum_four_quarters_and_compare_windows():
    revenue = [130, 120, 110, 100, 90, 90, 90, 90]
    income = _quarterly(
        {"TotalRevenue": revenue, "OperatingIncome": [r * 0.2 for r in revenue],
         "NetIncome": [r * 0.1 for r in revenue]},
        8,
    )
    cash = _quarterly({"OperatingCashFlow": [20] * 8, "CapitalExpenditure": [-5] * 8}, 8)
    balance = _quarterly({"TotalDebt": [50] * 8, "StockholdersEquity": [200] * 8}, 8)
    ttm = derive_ttm_fundamentals(income, balance, cash, "Technology")
    assert ttm["ttm_windows"] == 2
    assert ttm["revenue"] == 460
    assert ttm["revenue_growth_yoy"] == pytest.approx((460 / 360 - 1) * 100)
    assert ttm["free_cash_flow"] == 60
    assert latest_quarter_revenue_growth(income) == pytest.approx((130 / 90 - 1) * 100)


def test_ttm_refuses_a_gap_and_merge_prefers_fresh_levels():
    income = _quarterly({"TotalRevenue": [100, np.nan, 100, 100]}, 4)
    assert derive_ttm_fundamentals(income, pd.DataFrame(), pd.DataFrame()) == {}
    merged = merge_fundamentals(
        {"operating_margin_pct": 10, "revenue_growth_yoy": 5},
        {"operating_margin_pct": 20, "ttm_windows": 1},
    )
    assert merged["operating_margin_pct"] == 20
    assert merged["revenue_growth_yoy"] == 5
    assert merged["fundamentals_basis"] == "TTM_LEVELS_ANNUAL_GROWTH"
    assert merge_fundamentals({}, {})["fundamentals_basis"] == "ANNUAL"


def test_market_regime_flags_rebounds_below_the_trend():
    assert market_regime(3, -2) == RISK_ON
    assert market_regime(-4, 1) == RISK_OFF
    assert market_regime(-4, 8) == REBOUND_RISK
    assert market_regime(np.nan, 1) == REGIME_UNKNOWN


def test_risk_adjustment_and_imminent_earnings_penalty():
    result = risk_adjusted_scores(
        {"long_term_score": 70, "short_term_score": 70, "risk_score": 70,
         "days_to_next_earnings": 1}
    )
    assert result["long_term_risk_adjusted_score"] == pytest.approx(66)
    assert result["short_term_risk_adjusted_score"] == pytest.approx(61)
    assert result["earnings_imminent_flag"]
    calm = risk_adjusted_scores({"long_term_score": 70, "risk_score": 30})
    assert calm["long_term_risk_adjusted_score"] == 70


def test_ranking_follows_the_risk_adjusted_score():
    frame = pd.DataFrame(
        {
            "symbol": ["RISKY", "STEADY"],
            "long_term_score": [72.0, 70.0],
            "long_term_risk_adjusted_score": [64.0, 70.0],
            "short_term_score": [np.nan, np.nan],
            "confidence_score": [80, 80],
            "expectations_long_score": [60, 60],
            "expectations_short_score": [60, 60],
            "quality_score": [60, 60],
            "short_rs_score": [60, 60],
        }
    )
    lt, _ = rank_results(frame)
    assert lt.symbol.tolist() == ["STEADY", "RISKY"]


def test_trade_plan_scales_with_volatility_and_regime():
    plan = trade_plan({"current_price": 100, "volatility_20d": 30, "market_regime": RISK_ON})
    expected_stop = 30 / np.sqrt(252) * np.sqrt(10)
    assert plan["stop_loss_pct"] == pytest.approx(expected_stop)
    assert plan["stop_loss_price"] == pytest.approx(100 - expected_stop)
    assert plan["target_price"] == pytest.approx(100 + 1.5 * expected_stop)
    # 1% portfolio risk at a ~6% stop would be ~17%, so the 10% cap binds.
    assert plan["suggested_position_pct"] == pytest.approx(10)
    volatile = trade_plan({"current_price": 100, "volatility_20d": 60, "market_regime": RISK_ON})
    volatile_stop = 60 / np.sqrt(252) * np.sqrt(10)
    assert volatile["suggested_position_pct"] == pytest.approx(100 / volatile_stop)
    defensive = trade_plan({"current_price": 100, "volatility_20d": 60, "market_regime": RISK_OFF})
    assert defensive["suggested_position_pct"] == pytest.approx(volatile["suggested_position_pct"] / 2)
    assert math.isnan(trade_plan({"current_price": 100})["stop_loss_pct"])


def test_action_list_requires_long_term_top_half_and_caps_sectors():
    rows = []
    for i in range(8):
        rows.append(
            {
                "symbol": f"S{i}",
                "sector": "Tech" if i < 5 else "Energy",
                "long_term_rank": i + 1,
                "short_term_rank": i + 1,
                "short_term_risk_adjusted_score": 90 - i,
            }
        )
    rows.append({"symbol": "LOW", "sector": "Energy", "long_term_rank": 20,
                 "short_term_rank": 9, "short_term_risk_adjusted_score": 99})
    rows += [{"symbol": f"F{i}", "sector": "X", "long_term_rank": 21 + i,
              "short_term_rank": np.nan, "short_term_risk_adjusted_score": np.nan}
             for i in range(12)]
    action = build_action_list(pd.DataFrame(rows))
    assert "LOW" not in action.symbol.tolist()
    assert (action.sector == "Tech").sum() == 3
    assert action.action_rank.tolist() == list(range(1, len(action) + 1))


def test_action_list_weights_sum_to_at_most_the_portfolio():
    from investment_ai.scoring.action import portfolio_weights

    frame = pd.DataFrame(
        {"stop_loss_pct": [4.0, 8.0] + [5.0] * 18, "market_regime": [RISK_ON] * 20}
    )
    weights = portfolio_weights(frame)
    assert weights.sum() <= 100 + 1e-9
    assert weights.iloc[0] == pytest.approx(2 * weights.iloc[1])
    assert weights.max() <= 10


def _synthetic_histories(symbols, periods=160, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=periods)
    frames = {}
    for symbol in symbols:
        drift = rng.normal(0.0005, 0.0005)
        close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, periods)))
        frames[symbol] = pd.DataFrame(
            {"Close": close, "Volume": rng.integers(1e5, 1e6, periods).astype(float)},
            index=dates,
        )
    return pd.concat(frames, axis=1)


def test_backtest_is_point_in_time():
    symbols = [f"S{i}" for i in range(40)]
    universe = pd.DataFrame(
        {
            "symbol": symbols,
            "index_name": ["S&P 500"] * 20 + ["STOXX Europe 600"] * 20,
            "sector": ["Tech", "Energy"] * 20,
        }
    )
    histories = _synthetic_histories(symbols + ["^GSPC", "^STOXX"])
    panel, summary = run_backtest(histories, universe, step=20, horizons=(5,))
    assert not summary.empty and panel.as_of.nunique() >= 2
    first = panel.as_of.min()
    # Rewriting everything after the first rebalance date must not change the
    # signals observed on that date.
    tampered = histories.copy()
    tampered.loc[tampered.index > first.tz_localize(None)] *= 3
    replay, _ = run_backtest(tampered, universe, step=20, horizons=(5,))
    signals = [c for c in panel if c.startswith(("rs_", "momentum_pillar", "setup_quality"))]
    original = panel[panel.as_of.eq(first)].set_index("symbol")[signals]
    altered = replay[replay.as_of.eq(first)].set_index("symbol")[signals]
    pd.testing.assert_frame_equal(original, altered)
    # Forward returns start at the next session's close, not the signal close.
    closes = histories.xs("Close", axis=1, level=1)
    position = closes.index.get_loc(first.tz_localize(None))
    expected = (closes["S0"].iloc[position + 6] / closes["S0"].iloc[position + 1] - 1) * 100
    observed = panel[(panel.as_of == first) & (panel.symbol == "S0")].forward_5d_return.iloc[0]
    assert observed == pytest.approx(expected)
