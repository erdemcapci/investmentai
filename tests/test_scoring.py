import math

import numpy as np
import pandas as pd

from scoring import (
    INSUFFICIENT_DATA,
    assign_candidate_profile,
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


def test_combined_candidate_profile_requires_both_thresholds():
    profile, _ = assign_candidate_profile(70, 70)
    assert profile == "RESEARCH CANDIDATE"


def test_weighted_scores_stay_within_range():
    score, coverage = weighted_score_available(
        {"a": 120, "b": -10},
        {"a": 0.5, "b": 0.5},
    )
    assert 0 <= score <= 100
    assert coverage == 1
