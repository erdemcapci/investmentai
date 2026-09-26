"""v3.1.1 correctness and model-semantics regression tests."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from investment_ai.data.history_store import HistoryStore
from investment_ai.features.analyst import parse_estimates
from investment_ai.features.risk import risk_and_confidence
from investment_ai.features.technical import add_relative_strength, price_features
from investment_ai.features.valuation import add_peer_percentiles
from investment_ai.pipeline import LT_DRIVER_SPECS, ST_DRIVER_SPECS, _drivers
from investment_ai.scoring import short_term as short_term_module
from investment_ai.status import ERROR, FRESH_PROVIDER, STALE_FALLBACK


def test_legacy_only_analyst_history_is_ignored(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    old = datetime.now(timezone.utc) - timedelta(days=31)
    store.upsert_analyst("A", {"target_mean": 100}, old)
    value, status = store.historical_change("A", "target_mean", 30, 120)
    assert np.isnan(value)
    assert status == "HISTORY_NOT_YET_AVAILABLE"


def test_normalized_history_is_used_and_wins_over_legacy(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    old = datetime.now(timezone.utc) - timedelta(days=31)
    store.upsert_analyst("A", {"target_mean": 50}, old)
    store.upsert_component("A", "targets", {"target_mean": 100}, old)
    assert store.historical_change("A", "target_mean", 30, 120) == (20, "AVAILABLE")


def test_missing_normalized_30d_target_and_revenue_remain_nan(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    old = datetime.now(timezone.utc) - timedelta(days=31)
    store.upsert_analyst("A", {"target_mean": 50, "revenue_0y_avg": 50}, old)
    result = store.add_analyst_history_features(
        "A", {"target_mean": 100, "revenue_0y_avg": 100}
    )
    assert np.isnan(result["target_mean_change_30d_pct"])
    assert np.isnan(result["revenue_0y_change_30d_pct"])


def _mixed_region_frame():
    rows = []
    for index_name, prefix in (("S&P 500", "US"), ("STOXX Europe 600", "EU")):
        for number in range(15):
            rows.append(
                {
                    "symbol": f"{prefix}{number}",
                    "index_name": index_name,
                    "sector_normalized": "Technology" if number == 0 else f"Sector {number}",
                    "forward_pe": number + 1,
                    "return_20d_pct": number,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[0, "symbol"] = "AAPL"
    frame.loc[15, "symbol"] = "SAP.DE"
    return frame


def test_security_specific_index_fallback_for_valuation_and_rs():
    frame = _mixed_region_frame()
    valuation = add_peer_percentiles(frame, {"forward_pe": False})
    strength = add_relative_strength(frame)
    for result, prefix in ((valuation, "forward_pe"), (strength, "rs_20d")):
        aapl = result[result.symbol.eq("AAPL")].iloc[0]
        sap = result[result.symbol.eq("SAP.DE")].iloc[0]
        assert aapl[f"{prefix}_peer_index"] == "S&P 500"
        assert sap[f"{prefix}_peer_index"] == "STOXX Europe 600"
        assert aapl[f"{prefix}_peer_count"] == sap[f"{prefix}_peer_count"] == 15


def test_sector_and_metric_specific_peer_resolution():
    frame = pd.DataFrame(
        {
            "sector_normalized": ["Technology"] * 18,
            "index_name": ["S&P 500"] * 18,
            "forward_pe": range(18),
            "ev_ebitda": list(range(7)) + [np.nan] * 11,
        }
    )
    result = add_peer_percentiles(frame, {"forward_pe": False, "ev_ebitda": False})
    assert result.forward_pe_peer_method.eq("sector").all()
    assert result.forward_pe_peer_count.eq(18).all()
    assert result.ev_ebitda_peer_method.eq("universe").all()
    assert result.ev_ebitda_peer_count.eq(7).all()


def test_dual_membership_uses_largest_count_then_stable_name():
    frame = _mixed_region_frame()
    dual = frame.iloc[[0]].copy()
    dual["symbol"] = "DUAL"
    dual["index_name"] = "STOXX Europe 600 | S&P 500"
    dual["sector_normalized"] = "Other"
    frame = pd.concat([frame, dual], ignore_index=True)
    result = add_peer_percentiles(frame, {"forward_pe": False})
    selected = result[result.symbol.eq("DUAL")].iloc[0]
    # DUAL belongs to both, so both counts tie at 16; lexical name is stable.
    assert selected.forward_pe_peer_index == "S&P 500"
    assert selected.forward_pe_peer_count == 16


@pytest.mark.parametrize("setup", ["PULLBACK", "BREAKOUT", "MOMENTUM_CONTINUATION", "MIXED"])
def test_credible_setup_is_eligible(monkeypatch, setup):
    monkeypatch.setattr(short_term_module, "setup_scores", lambda _: {
        "pullback_setup_score": 70, "breakout_momentum_setup_score": 70,
        "setup_coverage": 1, "short_term_setup": setup,
        "setup_quality_score": 70, "setup_status": "RANKED",
    })
    row = {"rs_20d_percentile": 70, "rs_60d_percentile": 70,
           "rs_126d_percentile": 70, "expectations_short_score": 70,
           "expectations_short_coverage": 1, "relative_volume_1d": 1,
           "price_vs_ma20_pct": 1, "price_vs_ma50_pct": 1,
           "days_to_next_earnings": 20}
    assert pd.notna(short_term_module.score_short_term(row)["short_term_score"])


def test_none_setup_is_never_eligible_even_with_numeric_quality(monkeypatch):
    monkeypatch.setattr(short_term_module, "setup_scores", lambda _: {
        "pullback_setup_score": 50, "breakout_momentum_setup_score": 50,
        "setup_coverage": 1, "short_term_setup": "NONE",
        "setup_quality_score": 50, "setup_status": "RANKED",
    })
    result = short_term_module.score_short_term({
        "rs_20d_percentile": 70, "rs_60d_percentile": 70,
        "rs_126d_percentile": 70, "expectations_short_score": 70,
        "expectations_short_coverage": 1,
    })
    assert np.isnan(result["short_term_score"])
    assert result["short_term_status"] == "NO_CREDIBLE_SETUP"


def test_lt_and_st_drivers_use_their_own_pillars():
    row = {key: 80 for _, _, key, _ in LT_DRIVER_SPECS}
    row.update({key: 20 for _, _, key, _ in ST_DRIVER_SPECS})
    lt = _drivers(row, LT_DRIVER_SPECS)
    st = _drivers(row, ST_DRIVER_SPECS)
    assert "Business quality" in lt[0] and "Relative strength" not in lt[0]
    assert "Relative strength" in st[1] and "Business quality" not in st[1]
    assert sum(item["effective_weight"] for item in lt[2]) == 1
    assert sum(item["effective_weight"] for item in st[2]) == 1


@pytest.mark.parametrize(
    "row,minimum,maximum,reason",
    [
        ({"negative_equity_flag": True}, 100, 100, "NEGATIVE_EQUITY"),
        ({"negative_ebitda_flag": True, "net_debt": 100}, 95, 100, "POSITIVE_NET_DEBT"),
        ({"negative_ebitda_flag": True, "net_debt": -100, "net_cash_flag": True,
          "debt_to_equity": .1}, 0, 99, "NET_CASH"),
        ({"negative_operating_profit_flag": True, "net_debt": -100,
          "net_cash_flag": True, "debt_to_equity": .1}, 0, 99, "METRIC_BASED"),
    ],
)
def test_balance_sheet_loss_cases(row, minimum, maximum, reason):
    result = risk_and_confidence(row)
    assert minimum <= result["balance_sheet_risk_score"] <= maximum
    assert reason in result["balance_sheet_risk_reason"]


def _confidence(statuses):
    row = {"long_term_coverage": 1, "short_term_coverage": 1,
           "expectations_coverage": 1, "rating_count": 15}
    names = ("targets", "recommendations", "eps_trend", "eps_revisions",
             "revenue_estimate", "earnings_estimate", "rating_actions",
             "earnings_history", "earnings_dates")
    row.update({f"{name}_cache_status": status for name, status in zip(names, statuses)})
    row.update({"valuation_cache_status": statuses[-1],
                "fundamental_cache_status": statuses[-1],
                "price_data_status": statuses[-1]})
    return risk_and_confidence(row)


def test_component_freshness_orders_confidence_and_cannot_be_masked():
    fresh = _confidence([FRESH_PROVIDER] * 9)
    mixed = _confidence([FRESH_PROVIDER] * 5 + [STALE_FALLBACK] * 4)
    mostly_stale = _confidence([FRESH_PROVIDER] + [STALE_FALLBACK] * 8)
    errors = _confidence([ERROR] * 9)
    assert fresh["confidence_score"] > mixed["confidence_score"] > mostly_stale["confidence_score"] > errors["confidence_score"]
    assert mostly_stale["analyst_component_fresh_count"] == 1
    assert mostly_stale["analyst_component_stale_count"] == 8


def test_relative_volume_excludes_signal_windows_from_baselines():
    volumes = [10] * 55 + [20] * 4 + [40]
    history = pd.DataFrame(
        {"Close": np.arange(1, 61), "Volume": volumes},
        index=pd.date_range("2026-01-01", periods=60, tz="UTC"),
    )
    result = price_features(history, datetime(2026, 4, 1, tzinfo=timezone.utc))
    assert result["relative_volume_1d"] == pytest.approx(40 / (10 * 16 + 20 * 4) * 20)
    assert result["relative_volume_5d"] == pytest.approx(24 / 10)


@pytest.mark.parametrize("kind", ["eps", "revenue"])
def test_suspicious_forward_growth_scale_is_rejected(kind):
    result = parse_estimates(pd.DataFrame({"growth": [20]}, index=["+1y"]), kind)
    assert np.isnan(result[f"forward_{kind}_growth"])
    assert result[f"forward_{kind}_growth_scale_warning"] is True
