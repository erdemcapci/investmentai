import numpy as np
import pandas as pd

from investment_ai import notify
from investment_ai.scoring.overextension import overextension_check


def test_flags_large_one_year_gain_or_distance_above_trend():
    assert overextension_check({"return_252d_pct": 557.0, "price_vs_ma200_pct": 20.0})["overextended"]
    assert overextension_check({"return_252d_pct": 120.0, "price_vs_ma200_pct": 62.0})["overextended"]
    calm = overextension_check({"return_252d_pct": 80.0, "price_vs_ma200_pct": 30.0})
    assert not calm["overextended"] and calm["overextension_note"] == ""


def test_missing_inputs_are_not_flagged():
    assert not overextension_check({"return_252d_pct": np.nan})["overextended"]


def test_note_names_the_reason():
    note = overextension_check({"return_252d_pct": 1493.0, "price_vs_ma200_pct": 57.0})["overextension_note"]
    assert "up 1,493% in a year" in note and "57% above its 200-day average" in note
    pulled_back = overextension_check({"return_252d_pct": 208.0, "price_vs_ma200_pct": -3.0})["overextension_note"]
    assert "after a huge run" in pulled_back and "waiting for a pullback" not in pulled_back


def test_slack_marks_overextended_rows():
    row = pd.Series({"data_check_status": "OK", "overextended": np.bool_(True),
                     "overextension_note": "overextended (up 300% in a year)"})
    assert notify._flag(row).strip() == "🔥"
    assert any("overextended" in note for note in notify._warnings(row))
    assert notify._flag(pd.Series({"data_check_status": "CHECK", "overextended": False})).strip() == "⚠"
