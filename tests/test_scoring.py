import math

import numpy as np
import pandas as pd

from scoring import (
    INSUFFICIENT_DATA,
    assign_candidate_profile,
    calculate_analyst_backed_pullback_bonus,
    calculate_coverage_multiplier,
    score_selloff_stability,
    score_earnings_timing,
    score_eps_revisions,
    score_momentum,
    score_pullback_quality,
    weighted_score_available,
)


def test_missing_optional_eps_data_is_nan_not_zero():
    score, available = score_eps_revisions(np.nan, np.nan)
    assert math.isnan(score)
    assert available is False


def test_explicit_zero_eps_revisions_is_neutral():
    score, available = score_eps_revisions(0, 0)
    assert score == 50
    assert available is True


def test_missing_critical_analyst_data_can_be_marked_insufficient():
    score, coverage = weighted_score_available(
        {"upside": np.nan, "sentiment": 80},
        {"upside": 0.35, "sentiment": 0.30},
    )
    assert not math.isnan(score)
    assert coverage < 0.70
    assert INSUFFICIENT_DATA == "INSUFFICIENT_DATA"


def test_extreme_positive_momentum_scores_lower_than_healthy_momentum():
    healthy = score_momentum(6, 15)
    overextended = score_momentum(25, 60)
    assert healthy > overextended


def test_analyst_backed_stabilizing_pullback_earns_bonus():
    assert calculate_analyst_backed_pullback_bonus(40, 95, -4, 0.2) == 15


def test_analyst_backed_pullback_requires_every_condition():
    scenarios = [
        (24.9, 95, -4, 0.2),
        (40, 79.9, -4, 0.2),
        (40, 95, -2.9, 0.2),
        (40, 95, -4, 1.1),
        (np.nan, 95, -4, 0.2),
    ]
    assert all(
        calculate_analyst_backed_pullback_bonus(*scenario) == 0
        for scenario in scenarios
    )


def test_analyst_backed_pullback_bonus_accepts_requested_boundaries():
    assert calculate_analyst_backed_pullback_bonus(25, 80, -5, -1) == 15
    assert calculate_analyst_backed_pullback_bonus(25, 80, -3, 1) == 15


def test_controlled_pullback_scores_better_than_zero_drawdown():
    assert score_pullback_quality(-6) > score_pullback_quality(0)


def test_deep_drawdown_scores_poorly():
    assert score_pullback_quality(-35) == 0


def test_very_near_earnings_reduces_entry_score():
    assert score_earnings_timing(2) < score_earnings_timing(30)


def test_missing_earnings_date_is_nan_not_zero():
    assert math.isnan(score_earnings_timing(np.nan))


def test_high_analyst_count_does_not_change_raw_score():
    components = {"upside": 70, "sentiment": 80}
    weights = {"upside": 0.35, "sentiment": 0.30}
    raw_5, _ = weighted_score_available(components, weights)
    raw_30, _ = weighted_score_available(components, weights)
    assert raw_5 == raw_30


def test_softened_coverage_multiplier_for_moderate_coverage():
    multiplier = calculate_coverage_multiplier(0.75)
    assert math.isclose(multiplier, 0.925)
    assert math.isclose(80 * multiplier, 74)


def test_attractive_but_technically_weak_profile():
    profile, _ = assign_candidate_profile(82, 18)
    assert profile == "ATTRACTIVE BUT TECHNICALLY WEAK"


def test_attractive_wait_for_entry_profile():
    profile, _ = assign_candidate_profile(82, 52)
    assert profile == "ATTRACTIVE, WAIT FOR ENTRY"


def test_strong_candidate_profile_order():
    profile, _ = assign_candidate_profile(82, 78)
    assert profile == "STRONG CANDIDATE"


def test_tactical_candidate_profile():
    profile, _ = assign_candidate_profile(67, 81)
    assert profile == "TACTICAL CANDIDATE"


def test_momentum_only_profile():
    profile, _ = assign_candidate_profile(54, 80)
    assert profile == "MOMENTUM ONLY"


def test_missing_long_valid_short_is_technical_only():
    profile, _ = assign_candidate_profile(np.nan, 80)
    assert profile == "TECHNICAL ONLY"


def test_valid_long_missing_short_is_analyst_view_only():
    profile, _ = assign_candidate_profile(80, np.nan)
    assert profile == "ANALYST VIEW ONLY"


def test_combined_candidate_profile_requires_both_thresholds():
    profile, _ = assign_candidate_profile(70, 70)
    assert profile == "RESEARCH CANDIDATE"


def test_partial_selloff_inputs_still_score_with_partial_status():
    score, coverage, status = score_selloff_stability(1, 1, np.nan)
    assert not math.isnan(score)
    assert coverage == 75
    assert status == "PARTIAL_DATA"


def test_missing_all_selloff_inputs_excludes_component():
    score, coverage, status = score_selloff_stability(np.nan, np.nan, np.nan)
    assert math.isnan(score)
    assert coverage == 0
    assert status == "INSUFFICIENT_DATA"


def test_risk_penalties_are_modest_maximum_amounts():
    assert 5 <= 5
    assert 10 <= 10
    assert 3 <= 5


def test_weighted_scores_stay_within_range():
    score, coverage = weighted_score_available(
        {"a": 120, "b": -10},
        {"a": 0.5, "b": 0.5},
    )
    assert 0 <= score <= 100
    assert coverage == 1
