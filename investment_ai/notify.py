"""Slack notifications for the daily run.

The webhook URL comes from the ``SLACK_WEBHOOK_URL`` environment variable or
the project's git-ignored ``.env`` file.  Posting never raises: a Slack outage
must not fail a run, so problems are logged and reported as ``False``.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from investment_ai.dashboard_data import list_runs, load_run

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
BASKET_SIZE = 5
HOLD_SESSIONS = 10
# Calendar days that cover a 10-session hold plus the entry day.
HOLDING_WINDOW_DAYS = 16
VOLATILE_STOP_PCT = 24.9
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "GBp": "", "CHF": "CHF ", "PLN": "zł ",
                    "DKK": "kr ", "SEK": "kr ", "NOK": "kr "}
STATUS_ICONS = {"VALID": "✅", "DEGRADED": "⚠️", "INVALID": "⛔"}


def webhook_url(env_file: Path = ENV_FILE) -> str | None:
    """Webhook from the environment, else from ``KEY=value`` lines in .env."""
    value = os.getenv("SLACK_WEBHOOK_URL")
    if value:
        return value.strip()
    try:
        for line in env_file.read_text().splitlines():
            key, _, raw = line.partition("=")
            if key.strip() == "SLACK_WEBHOOK_URL" and raw.strip():
                return raw.strip().strip("'\"")
    except OSError:
        return None
    return None


def post(text: str, url: str | None = None, timeout: float = 15) -> bool:
    url = url or webhook_url()
    logger = logging.getLogger("investment_ai")
    if not url:
        logger.warning("slack skipped: SLACK_WEBHOOK_URL not configured",
                       extra={"symbol": "-", "component": "notify"})
        return False
    try:
        response = requests.post(url, json={"text": text}, timeout=timeout)
        if response.status_code == 200:
            return True
        logger.warning("slack post failed status=%s", response.status_code,
                       extra={"symbol": "-", "component": "notify"})
    except requests.RequestException as exc:
        logger.warning("slack post failed: %s", str(exc)[:200],
                       extra={"symbol": "-", "component": "notify"})
    return False


def pick_basket(action: pd.DataFrame, size: int = BASKET_SIZE) -> pd.DataFrame:
    """Action-list order, at most one stock per sector, first ``size`` names."""
    if action.empty:
        return action
    ordered = action.sort_values("action_rank") if "action_rank" in action else action
    sector = ordered.get("sector", pd.Series("", index=ordered.index)).fillna("")
    return ordered[~sector.duplicated()].head(size)


def _money(value, currency) -> str:
    if pd.isna(value):
        return "n/a"
    symbol = CURRENCY_SYMBOLS.get(str(currency), "")
    suffix = "p" if currency == "GBp" else ""
    return f"{symbol}{value:,.2f}{suffix}" if value < 1000 else f"{symbol}{value:,.0f}{suffix}"


def _price_date(full: pd.DataFrame) -> date | None:
    stamps = pd.to_datetime(full.get("price_as_of"), utc=True, errors="coerce").dropna()
    return stamps.max().date() if len(stamps) else None


def _warnings(row: pd.Series) -> list[str]:
    notes = []
    days = pd.to_numeric(row.get("days_to_next_earnings"), errors="coerce")
    if pd.isna(days):
        notes.append("no earnings date in data — check before buying")
    elif days <= HOLDING_WINDOW_DAYS:
        notes.append(f"reports earnings in ~{days:.0f} days, inside the hold")
    stop = pd.to_numeric(row.get("stop_loss_pct"), errors="coerce")
    if pd.notna(stop) and stop >= VOLATILE_STOP_PCT:
        notes.append("very volatile (stop at the 25% cap) — consider half size")
    return notes


def due_for_sale(runs: list[dict], current_date: date) -> tuple[str, date, list[str]] | None:
    """The run whose basket reaches its 10-session hold now, if one exists."""
    for run in runs:
        try:
            full, action = load_run(run["path"])
        except (OSError, ValueError):
            continue
        run_date = _price_date(full)
        if run_date is None or run_date >= current_date:
            continue
        sessions = int(np.busday_count(run_date, current_date))
        if HOLD_SESSIONS <= sessions <= HOLD_SESSIONS + 2:
            return run["run_id"], run_date, pick_basket(action).symbol.tolist()
        if sessions > HOLD_SESSIONS + 2:
            break
    return None


def daily_message(runs_dir: Path) -> str:
    runs = [
        run for run in list_runs(runs_dir)
        if "-replay-" not in run["run_id"] and "-rescore-" not in run["run_id"]
    ]
    if not runs:
        return "⛔ *Investment AI* — no completed run found."
    latest = runs[0]
    full, action = load_run(latest["path"])
    manifest = latest["manifest"]
    status = manifest.get("overall_run_status", "UNKNOWN")
    as_of = _price_date(full)
    regimes = full.get("market_regime", pd.Series(dtype=object)).value_counts().to_dict()
    regime_text = ", ".join(f"{name} ({count})" for name, count in regimes.items()) or "unknown"
    lines = [
        f"📈 *Investment AI* — prices as of {as_of or 'unknown'} · "
        f"{STATUS_ICONS.get(status, '❔')} {status} · market {regime_text}",
    ]
    if status == "INVALID":
        lines.append("⛔ Data quality is too low today — *no new trades*. "
                     "Check run_manifest.json for the reasons.")
        return "\n".join(lines)
    if any(name != "RISK_ON" for name in regimes):
        lines.append("⚠️ Market is not fully risk-on: use *half* the usual amounts.")
    basket = pick_basket(action)
    if basket.empty:
        lines.append("No action-list names today.")
    else:
        sell_on = np.busday_offset(date.today(), HOLD_SESSIONS, roll="forward")
        lines.append(f"\n*Top {len(basket)}* (one per sector) — buy near today's close, "
                     f"equal amounts, sell on ~{pd.Timestamp(sell_on).strftime('%a %d %b')}")
        details = full.set_index("symbol")
        rows, notes = [], []
        for item in basket.itertuples():
            row = details.loc[item.symbol] if item.symbol in details.index else pd.Series(dtype=object)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            currency = row.get("trading_currency")
            rows.append(
                f"{int(item.action_rank):>2}. {item.symbol:<9}{str(item.sector)[:22]:<23}"
                f"{_money(row.get('current_price'), currency):>11}  stop {_money(row.get('stop_loss_price'), currency)}"
            )
            notes += [f"• {item.symbol}: {note}" for note in _warnings(row)]
        lines.append("```" + "\n".join(rows) + "```")
        if notes:
            lines.append("\n".join(notes))
    due = due_for_sale(runs[1:], as_of) if as_of else None
    if due:
        _, run_date, symbols = due
        lines.append(f"🔔 *Sell today* if you bought the {run_date:%d %b} list: {', '.join(symbols)}")
    reasons = manifest.get("run_status_reasons") or []
    if status == "DEGRADED" and reasons:
        lines.append("_Coverage notes: " + "; ".join(reasons[:2]) + "_")
    lines.append(f"_Full list: {len(action)} names in action_list.csv · dashboard: "
                 "streamlit run dashboard.py · research only, not advice_")
    return "\n".join(lines)


def failure_message(exit_code: str, log_path: str | None = None) -> str:
    lines = [f"⛔ *Investment AI* — the scheduled run failed (exit code {exit_code})."]
    if log_path:
        try:
            tail = Path(log_path).read_text(errors="replace").strip().splitlines()[-6:]
            lines.append("```" + "\n".join(line[:180] for line in tail) + "```")
        except OSError:
            pass
        lines.append(f"Log: {log_path}")
    lines.append("Resume with: python main.py --resume <run_id>")
    return "\n".join(lines)
