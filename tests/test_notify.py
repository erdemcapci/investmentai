import json

import numpy as np
import pandas as pd
import pytest
import requests

from investment_ai import notify


def _write_run(root, run_id, price_as_of, status="DEGRADED", sectors=None, regime="RISK_ON"):
    directory = root / run_id
    directory.mkdir(parents=True)
    sectors = sectors or ["Tech", "Tech", "Energy", "Health", "Banks", "Retail", "Utilities"]
    symbols = [f"S{i}" for i in range(len(sectors))]
    full = pd.DataFrame(
        {
            "symbol": symbols,
            "sector": sectors,
            "price_as_of": price_as_of,
            "market_regime": regime,
            "trading_currency": "USD",
            "current_price": 100.0,
            "stop_loss_price": 90.0,
            "stop_loss_pct": [25.0] + [10.0] * (len(sectors) - 1),
            "days_to_next_earnings": [40, 40, 5, np.nan] + [40] * (len(sectors) - 4),
        }
    )
    full.to_csv(directory / "full_analysis.csv", index=False)
    full.assign(action_rank=range(1, len(sectors) + 1)).to_csv(directory / "action_list.csv", index=False)
    (directory / "run_manifest.json").write_text(
        json.dumps({"overall_run_status": status, "run_status_reasons": ["coverage note"]})
    )


def test_webhook_url_prefers_environment_then_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nSLACK_WEBHOOK_URL='https://hooks.example/abc'\n")
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert notify.webhook_url(env) == "https://hooks.example/abc"
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.example/env")
    assert notify.webhook_url(env) == "https://hooks.example/env"
    monkeypatch.delenv("SLACK_WEBHOOK_URL")
    assert notify.webhook_url(tmp_path / "missing") is None


def test_post_never_raises(monkeypatch):
    class Response:
        status_code = 500

    monkeypatch.setattr(notify.requests, "post", lambda *a, **k: Response())
    assert notify.post("x", url="https://hooks.example/a") is False

    def boom(*a, **k):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(notify.requests, "post", boom)
    assert notify.post("x", url="https://hooks.example/a") is False
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(notify, "ENV_FILE", notify.Path("/nonexistent/.env"))
    assert notify.post("x", url=None) is False


def test_post_sends_text_payload(monkeypatch):
    sent = {}

    class Response:
        status_code = 200

    def fake(url, json, timeout):
        sent.update(url=url, json=json)
        return Response()

    monkeypatch.setattr(notify.requests, "post", fake)
    assert notify.post("hello", url="https://hooks.example/a")
    assert sent == {"url": "https://hooks.example/a", "json": {"text": "hello"}}


def test_basket_takes_one_stock_per_sector_in_rank_order():
    action = pd.DataFrame(
        {"symbol": list("ABCDEFG"), "sector": ["T", "T", "E", "H", "B", "R", "U"], "action_rank": range(1, 8)}
    )
    assert notify.pick_basket(action).symbol.tolist() == ["A", "C", "D", "E", "F"]


def test_daily_message_contents_and_warnings(tmp_path):
    _write_run(tmp_path, "20261001T000000Z-a", "2026-09-30T00:00:00+00:00")
    text = notify.daily_message(tmp_path)
    assert "⚠️ DEGRADED" in text and "Top 5" in text
    assert "S1" not in text  # second Tech name is skipped
    assert "S0: very volatile" in text
    assert "S2: reports earnings in ~5 days" in text
    assert "S3: no earnings date" in text
    assert "not fully risk-on" not in text  # risk-on market keeps full size


def test_daily_message_invalid_run_says_no_trades(tmp_path):
    _write_run(tmp_path, "20261001T000000Z-a", "2026-09-30T00:00:00+00:00", status="INVALID")
    text = notify.daily_message(tmp_path)
    assert "no new trades" in text and "Top 5" not in text


def test_daily_message_halves_size_outside_risk_on(tmp_path):
    _write_run(tmp_path, "20261001T000000Z-a", "2026-09-30T00:00:00+00:00", regime="RISK_OFF")
    assert "*half*" in notify.daily_message(tmp_path)


def test_sell_reminder_names_the_list_from_ten_sessions_ago(tmp_path):
    _write_run(tmp_path, "20260916T000000Z-old", "2026-09-16T00:00:00+00:00")
    _write_run(tmp_path, "20260930T000000Z-new", "2026-09-30T00:00:00+00:00")
    text = notify.daily_message(tmp_path)
    assert "Sell today" in text and "16 Sep" in text and "S0, S2, S3, S4, S5" in text


def test_no_sell_reminder_without_a_matured_list(tmp_path):
    _write_run(tmp_path, "20260929T000000Z-old", "2026-09-29T00:00:00+00:00")
    _write_run(tmp_path, "20260930T000000Z-new", "2026-09-30T00:00:00+00:00")
    assert "Sell today" not in notify.daily_message(tmp_path)


def test_failure_message_includes_log_tail(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join(f"line {i}" for i in range(10)))
    text = notify.failure_message("1", str(log))
    assert "exit code 1" in text and "line 9" in text and "line 3" not in text


@pytest.mark.parametrize("value, currency, expected", [
    (43.14, "EUR", "€43.14"), (1065.1, "USD", "$1,065"), (14370.0, "GBp", "14,370p"), (np.nan, "USD", "n/a"),
])
def test_money_formatting(value, currency, expected):
    assert notify._money(value, currency) == expected
