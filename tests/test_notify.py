import json

import numpy as np
import pandas as pd
import pytest
import requests

from investment_ai import notify
from investment_ai.scoring.data_check import stock_data_check


def _write_run(root, run_id, price_as_of, status="VALID", sectors=None, regime="RISK_ON"):
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
            "short_term_rank": range(1, len(sectors) + 1),
            "long_term_rank": range(1, len(sectors) + 1),
            "price_data_status": "FRESH",
            "eps_short_horizon_momentum_score": 60.0,
        }
    )
    checks = pd.DataFrame([stock_data_check(row) for row in full.to_dict("records")])
    full = pd.concat([full, checks], axis=1)
    full.to_csv(directory / "full_analysis.csv", index=False)
    full.assign(action_rank=range(1, len(sectors) + 1)).to_csv(directory / "action_list.csv", index=False)
    (directory / "run_manifest.json").write_text(
        json.dumps({"overall_run_status": status, "run_status_reasons": [], "coverage_notes": ["coverage note"]})
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
    assert "✅ data OK" in text and "Top 5" in text and "Check before buying" in text
    basket = text.split("*Top 5*")[1].split("Short-term top 20")[0]
    assert " S1 " not in basket  # second Tech name is skipped in the basket
    assert "S0: very volatile" in text
    assert "S2: earnings in ~5 days" in text
    assert "S3: no next earnings date" in text
    assert "Coverage notes" not in text
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


def test_ranking_tables_show_company_other_rank_and_analyst_target(tmp_path):
    full = pd.DataFrame(
        {
            "symbol": ["AAA", "LONGSYMBOL.CO", "CCC"],
            "company_name": ["Alpha", "A Very Long Company Name Holdings", "Gamma"],
            "short_term_rank": [1, 2, np.nan],
            "long_term_rank": [np.nan, 7, 1],
            "current_price": [100.0, 50.0, 10.0],
            "target_mean": [120.0, np.nan, 9.0],
            "rating_count": [12, 3, np.nan],
            "trading_currency": ["USD", "DKK", "EUR"],
        }
    )
    table = notify.ranking_table(full, "short_term_rank", "long_term_rank", "LT")
    lines = table.strip("`").splitlines()
    assert lines[0].startswith(" # Symbol") and len(lines) == 3  # header + 2 ranked names
    assert "Alpha" in lines[1] and "$120.00 +20% (12)" in lines[1] and " — " in lines[1]
    assert "A Very Long Compa…" in lines[2] and lines[2].rstrip().endswith("n/a")
    # Columns stay aligned even with a long symbol.
    assert lines[1].index("Alpha") == lines[2].index("A Very")
    lt = notify.ranking_table(full, "long_term_rank", "short_term_rank", "ST")
    assert "Gamma" in lt and "€9.00 -10%" in lt


def test_daily_message_includes_both_top_lists(tmp_path):
    _write_run(tmp_path, "20261001T000000Z-a", "2026-09-30T00:00:00+00:00")
    text = notify.daily_message(tmp_path)
    assert "Short-term top 20" in text and "Long-term top 20" in text


def test_degraded_message_explains_outdated_prices(tmp_path):
    _write_run(tmp_path, "20261001T000000Z-a", "2026-09-30T00:00:00+00:00", status="DEGRADED")
    manifest = tmp_path / "20261001T000000Z-a" / "run_manifest.json"
    manifest.write_text(json.dumps({"overall_run_status": "DEGRADED", "coverage_notes": [],
                                    "run_status_reasons": ["price_fresh_coverage_pct=93.0% is below 97%"]}))
    text = notify.daily_message(tmp_path)
    assert "Prices are outdated for part of the market" in text and "93.0%" in text
