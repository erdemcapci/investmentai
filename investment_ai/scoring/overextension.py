"""Per-stock overextension check: has the price run far ahead of its trend?

A stock up several-fold in a year, or far above its 200-day average, is not a
worse pick on average, but it falls hard much more often soon after buying.
On the stored price history (Sep 2025 - Sep 2026, ~57k stock-weeks) the share
that fell 20% or more within the next 40 sessions was:

* up 200%+ in a year: about 30%, even after it had already fallen back
  towards its 200-day average (the swings stay large);
* only 50%+ above the 200-day average: about 16%;
* neither: about 6%.

The check is a warning only and does not change any score or rank.
"""

from __future__ import annotations

import pandas as pd

ABOVE_MA200_PCT = 50.0
ONE_YEAR_GAIN_PCT = 200.0


def overextension_check(row: dict) -> dict:
    above = pd.to_numeric(row.get("price_vs_ma200_pct"), errors="coerce")
    gain = pd.to_numeric(row.get("return_252d_pct"), errors="coerce")
    big_run = pd.notna(gain) and gain >= ONE_YEAR_GAIN_PCT
    stretched = pd.notna(above) and above >= ABOVE_MA200_PCT
    if big_run and stretched:
        note = (f"overextended (up {gain:,.0f}% in a year, {above:.0f}% above its 200-day average) — "
                "about 1 in 3 such stocks fell 20%+ within 2 months; consider half size")
    elif big_run:
        note = (f"after a huge run (up {gain:,.0f}% in a year) — even after a pullback about 1 in 3 "
                "such stocks fell another 20%+ within 2 months; consider half size")
    elif stretched:
        note = (f"overextended ({above:.0f}% above its 200-day average) — about 1 in 6 such stocks "
                "fell 20%+ within 2 months; consider half size or waiting for a pullback")
    else:
        note = ""
    return {"overextended": bool(note), "overextension_note": note}
