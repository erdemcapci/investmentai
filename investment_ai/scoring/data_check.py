"""Per-stock data check: is any information behind this stock's ranking missing?

A run can be complete enough while one recommended stock still rests on an
outdated price or a stale analyst snapshot.  This check looks only at the
inputs that would change what you do with that stock:

* ``INCOMPLETE`` — the price or trade plan cannot be trusted; the stock is
  kept out of the action list.
* ``CHECK`` — tradable, but verify the listed items first.
* ``OK`` — nothing material missing.

Gaps that the model deliberately scores as neutral (short interest outside
the US, earnings drift before the first report) and inputs missing for every
stock alike (history still accumulating, quiet weeks without upgrades) are
not stock-specific and are not flagged.  Only cache
statuses are used, never wall-clock ages, so replays reproduce the result.
"""

from __future__ import annotations

import pandas as pd

OK, CHECK, INCOMPLETE = "OK", "CHECK", "INCOMPLETE"
# Analyst inputs behind the short- and long-term expectations pillars.
ANALYST_COMPONENTS = {
    "eps_trend": "EPS estimate trend",
    "eps_revisions": "EPS revisions",
    "earnings_estimate": "earnings estimates",
    "revenue_estimate": "revenue estimates",
    "recommendations": "analyst ratings",
    "targets": "price targets",
}
HOLDING_WINDOW_DAYS = 16


def _number(value) -> float:
    return pd.to_numeric(value, errors="coerce")


def stock_data_check(row: dict) -> dict:
    critical, check = [], []
    price_status = row.get("price_data_status")
    if pd.isna(_number(row.get("current_price"))):
        critical.append("no price")
    elif price_status not in (None, "FRESH") and not pd.isna(price_status):
        critical.append(f"price is not current ({str(price_status).lower()})")
    ranked = pd.notna(_number(row.get("short_term_rank"))) or pd.notna(_number(row.get("long_term_rank")))
    if ranked and pd.isna(_number(row.get("stop_loss_price"))):
        critical.append("no stop level (volatility missing)")

    stale = [
        label for key, label in ANALYST_COMPONENTS.items()
        if row.get(f"{key}_cache_status") == "STALE_FALLBACK"
    ]
    if stale:
        check.append("analyst data not refreshed today: " + ", ".join(stale))
    # EPS estimate momentum is the core of the analyst pillar.  Inputs that
    # need stored history (revenue revisions, target changes) or recent
    # events (upgrades) are missing for every stock early on or in quiet
    # weeks, so they are not stock-specific gaps and are not flagged.
    if pd.notna(_number(row.get("short_term_rank"))) and pd.isna(
        _number(row.get("eps_short_horizon_momentum_score"))
    ):
        check.append("no EPS estimate trend")
    if row.get("fundamental_cache_status") in {"STALE_FALLBACK", "ERROR"} and pd.notna(
        _number(row.get("long_term_rank"))
    ):
        check.append("fundamentals not refreshed")

    days = _number(row.get("days_to_next_earnings"))
    if pd.isna(days):
        check.append("no next earnings date — check before buying")
    elif 0 <= days <= HOLDING_WINDOW_DAYS:
        check.append(f"earnings in ~{days:.0f} days, inside a 2-week hold")
    if row.get("market_regime") in (None, "UNKNOWN") or pd.isna(row.get("market_regime")):
        check.append("market trend unknown (benchmark missing)")

    status = INCOMPLETE if critical else CHECK if check else OK
    return {
        "data_check_status": status,
        "data_check_notes": "; ".join(critical + check),
    }
