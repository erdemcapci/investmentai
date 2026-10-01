"""Walk-forward backtest of the price-based signals.

Analyst, valuation and fundamental inputs have no point-in-time history from
Yahoo, so only price and volume signals can be replayed.  For every rebalance
date the production ``price_features`` function sees only bars up to that
date, and the production pillar functions score the resulting cross-section,
so the backtest measures the same code that ranks stocks today.

Known biases, reported in the manifest:
* survivorship: the universe is today's constituent list;
* sectors are today's classifications;
* peer groups use sector, then region, then universe (production resolves
  dual index membership slightly differently).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from investment_ai.config import (
    BACKTEST_DIR,
    HISTORY_DB,
    MIN_PEERS,
    RUNS_DIR,
    SCORING_MODEL_VERSION,
)
from investment_ai.features.benchmark import (
    BENCHMARKS,
    RISK_ON,
    add_benchmark_relative_strength,
    assign_benchmark,
    attach_benchmark_prices,
    market_regime,
)
from investment_ai.features.peers import fallback_percentile
from investment_ai.features.technical import (
    RS_METRICS,
    add_model_relative_strength,
    earnings_drift_score,
    earnings_reaction,
    price_features,
    setup_scores,
)
from investment_ai.scoring.action import trade_plan
from investment_ai.scoring.common import weighted
from investment_ai.scoring.long_term import long_trend_pillar
from investment_ai.scoring.short_term import (
    CREDIBLE_SETUPS,
    SHORT_TERM_WEIGHTS,
    momentum_pillar,
    technical_pillar,
    volume_pillar,
)

HORIZONS = (5, 10, 20)
MIN_HISTORY_SESSIONS = 64
TOP_N = 25
# Model 3.2.0's price-pillar weights, kept to compare against production.
COMPOSITE_3_2_0_WEIGHTS = {"momentum": 0.20, "setup": 0.20, "volume": 0.10, "technical": 0.05}


def _close_panel(histories: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    closes = {}
    for symbol in symbols:
        if symbol in histories.columns.get_level_values(0):
            series = pd.to_numeric(histories[symbol]["Close"], errors="coerce")
            series.index = pd.to_datetime(series.index, utc=True)
            closes[symbol] = series[series > 0].dropna()
    panel = pd.DataFrame(closes).sort_index()
    return panel.ffill(limit=5)


def _symbol_histories(histories: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    output = {}
    for symbol in symbols:
        if symbol not in histories.columns.get_level_values(0):
            continue
        frame = histories[symbol].dropna(how="all").copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        output[symbol] = frame.sort_index()
    return output


def _features_as_of(history: dict[str, pd.DataFrame], symbols: list[str], as_of: pd.Timestamp) -> pd.DataFrame:
    # Advancing "now" by a day keeps the as-of bar, which is complete by then.
    now = (as_of + timedelta(days=1)).to_pydatetime()
    rows = []
    for symbol in symbols:
        frame = history.get(symbol)
        if frame is None:
            continue
        visible = frame.loc[:as_of]
        if len(visible) < MIN_HISTORY_SESSIONS:
            continue
        rows.append({"symbol": symbol, **price_features(visible, now=now)})
    return pd.DataFrame(rows)


def score_cross_section(features: pd.DataFrame, universe: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    """Score one date's cross-section with the production pillar functions."""
    frame = universe[["symbol", "index_name", "sector"]].merge(features, on="symbol", how="inner")
    frame = attach_benchmark_prices(universe, frame, benchmarks)
    region = frame.index_name.map(lambda value: assign_benchmark(value)[0])
    for key, metric in RS_METRICS.items():
        frame[f"rs_{key}_percentile"] = fallback_percentile(
            frame.get(metric, pd.Series(np.nan, index=frame.index)),
            [frame.sector.fillna(""), region],
            MIN_PEERS,
        )
    frame = add_benchmark_relative_strength(frame, MIN_PEERS)
    frame["market_regime"] = [
        market_regime(ma, ret)
        for ma, ret in zip(
            frame.get("benchmark_price_vs_ma200_pct", pd.Series(np.nan, index=frame.index)),
            frame.get("benchmark_return_20d_pct", pd.Series(np.nan, index=frame.index)),
        )
    ]
    signals = {}
    for basis in ("raw", "benchmark"):
        based = add_model_relative_strength(frame, basis)
        records = based.to_dict("records")
        momentum = [momentum_pillar(row)[0] for row in records]
        setups = [setup_scores(row) for row in records]
        volume = [volume_pillar(row) for row in records]
        technical = [technical_pillar(row) for row in records]
        composite = [
            weighted(
                {"momentum": m, "setup": s["setup_quality_score"], "volume": v, "technical": t},
                COMPOSITE_3_2_0_WEIGHTS,
                0.6,
            )[0]
            for m, s, v, t in zip(momentum, setups, volume, technical)
        ]
        signals[f"momentum_pillar_{basis}"] = momentum
        signals[f"st_price_composite_{basis}"] = composite
        signals[f"long_trend_pillar_{basis}"] = [long_trend_pillar(row)[0] for row in records]
        if basis == "raw":  # the production default basis
            signals["setup_quality"] = [s["setup_quality_score"] for s in setups]
            signals["credible_setup"] = [
                100.0 if s["short_term_setup"] in CREDIBLE_SETUPS else 0.0 for s in setups
            ]
            signals["volume_pillar"] = volume
            signals["technical_pillar"] = technical
            signals["credible_composite"] = [
                c if s["short_term_setup"] in CREDIBLE_SETUPS else np.nan
                for c, s in zip(composite, setups)
            ]
    for key in RS_METRICS:
        signals[f"rs_{key}_raw"] = frame[f"rs_{key}_percentile"].tolist()
        signals[f"rs_{key}_excess"] = frame[f"benchmark_rs_{key}_percentile"].tolist()
    signals["reversal_20d_excess"] = (100 - frame["benchmark_rs_20d_percentile"]).tolist()
    # The pre-3.2 short-term relative-strength blend, kept for comparison.
    signals["legacy_short_rs_raw"] = [
        weighted({"a": a, "b": b, "c": c}, {"a": 0.50, "b": 0.35, "c": 0.15}, 0.60)[0]
        for a, b, c in zip(
            frame["rs_20d_percentile"], frame["rs_60d_percentile"], frame["rs_126d_percentile"]
        )
    ]
    columns = ["symbol", "sector", "benchmark_symbol", "market_regime", "volatility_20d", "volatility_60d"]
    output = frame.reindex(columns=columns).copy()
    for name, values in signals.items():
        output[name] = values
    return output


def earnings_event_table(
    events: pd.DataFrame, stock: pd.Series, benchmark: pd.Series | None
) -> pd.DataFrame:
    """Per-report surprise, announcement reaction and the date it became known.

    ``available_at`` is the close that completes the two-session reaction
    window; a rebalance date may only use reports available by its close.
    """
    rows = []
    if events is None or events.empty or stock is None or stock.dropna().empty:
        return pd.DataFrame(columns=["event", "surprise_pct", "reaction_pct", "available_at"])
    close = stock.dropna()
    for event, surprise in zip(events.event, events.surprise_pct):
        day = event.normalize()
        after = close.index[close.index >= day]
        if not len(after) or not len(close.index[close.index < day]):
            continue
        available = after[min(1, len(after) - 1)]
        reaction = earnings_reaction(close.loc[:available], benchmark, event)
        value = reaction["earnings_reaction_excess_pct"]
        if pd.isna(value):
            value = reaction["earnings_reaction_pct"]
        rows.append({"event": event, "surprise_pct": surprise, "reaction_pct": value, "available_at": available})
    return pd.DataFrame(rows)


def drift_as_of(tables: dict[str, pd.DataFrame], symbols: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    """Point-in-time earnings drift score for each symbol on ``as_of``."""
    values = []
    for symbol in symbols:
        table = tables.get(symbol)
        known = table[table.available_at <= as_of] if table is not None and len(table) else None
        if known is None or known.empty:
            values.append(np.nan)
            continue
        last = known.iloc[-1]
        days = (as_of - last.event).total_seconds() / 86400
        values.append(earnings_drift_score(last.surprise_pct, last.reaction_pct, days))
    return pd.Series(values, index=symbols.index, dtype=float)


def load_earnings_events(symbols: list[str], workers: int = 4) -> dict[str, pd.DataFrame]:
    """Historical report dates and EPS surprises from Yahoo, cached for a week."""
    from concurrent.futures import ThreadPoolExecutor

    import yfinance as yf

    from investment_ai.config import CACHE_DIR
    from investment_ai.data.cache import JsonCache
    from investment_ai.data.yahoo import call_with_retry

    cache = JsonCache(CACHE_DIR)

    def fetch(symbol: str) -> dict:
        frame = call_with_retry(lambda: yf.Ticker(symbol).get_earnings_dates(limit=60))
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return {}
        column = next(
            (c for c in frame if str(c).replace(" ", "").lower() in {"surprise(%)", "surprise%"}),
            None,
        )
        if column is None:
            return {}
        surprises = pd.to_numeric(frame[column], errors="coerce").dropna()
        return {"events": [[str(pd.Timestamp(i).tz_convert("UTC")), float(v)] for i, v in surprises.items()]}

    def load(symbol: str) -> tuple[str, pd.DataFrame]:
        item, _ = cache.get_or_fetch(
            "backtest_earnings_events", symbol, 24 * 7, lambda: fetch(symbol),
            validator=lambda data: bool(data.get("events")),
        )
        events = item.get("data", {}).get("events", [])
        frame = pd.DataFrame(events, columns=["event", "surprise_pct"])
        frame["event"] = pd.to_datetime(frame.event, utc=True)
        return symbol, frame.sort_values("event").reset_index(drop=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(load, symbols))


STRATEGY_PICKS = 20
STRATEGY_MAX_PER_SECTOR = 3
HOLD_SESSIONS = 10


def strategy_definitions(scored: pd.DataFrame) -> dict[str, pd.Series]:
    """Candidate short-term selection rules, each as a ranking signal.

    NaN excludes a security; the highest values are bought.
    """
    credible = scored.credible_setup.eq(100)
    drift = scored.get("earnings_drift", pd.Series(np.nan, index=scored.index))
    # Production short-term weights restricted to the replayable pillars.
    production = pd.Series(
        [
            weighted(
                {"momentum": m, "earnings_drift": d},
                {key: SHORT_TERM_WEIGHTS[key] for key in ("momentum", "earnings_drift")},
                0.5,
                neutral_fill=50,
            )[0]
            for m, d in zip(scored.momentum_pillar_raw, drift)
        ],
        index=scored.index,
    )
    definitions = {
        "production_price_model": production,
        "momentum_all": scored.momentum_pillar_raw,
        "gated_momentum": scored.momentum_pillar_raw.where(credible),
        "gated_composite_3_2_0": scored.st_price_composite_raw.where(credible),
    }
    if "earnings_drift" in scored:
        definitions["drift_all"] = drift
    return definitions


def _pick(scored: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    ranked = scored.assign(_signal=signal).dropna(subset=["_signal"])
    ranked = ranked.sort_values(["_signal", "symbol"], ascending=[False, True], kind="stable")
    sector = ranked.sector.fillna("")
    ranked = ranked[sector.groupby(sector).cumcount() < STRATEGY_MAX_PER_SECTOR]
    return ranked.head(STRATEGY_PICKS)


def simulate_trades(
    picks: pd.DataFrame, closes: pd.DataFrame, bench: pd.DataFrame, position: int, cost_bps: float
) -> list[dict]:
    """Hold ten sessions, and separately follow the production trade plan.

    Entry is the next session's close. The trade-plan exit checks closes only
    (no intraday highs or lows): the first close at or below the stop or at or
    above the target ends the trade, otherwise it exits after ten sessions.
    """
    entry = position + 1
    final = entry + HOLD_SESSIONS
    if final >= len(closes):
        return []
    round_trip = 2 * cost_bps / 100
    trades = []
    for row in picks.to_dict("records"):
        if row["symbol"] not in closes:
            continue
        path = closes[row["symbol"]].iloc[entry:final + 1]
        market = bench.get(row["benchmark_symbol"])
        if path.isna().any() or market is None:
            continue
        market_path = market.iloc[entry:final + 1]
        start = path.iloc[0]
        plan = trade_plan({**row, "current_price": start})
        exits = {"hold_10": (HOLD_SESSIONS, "TIME")}
        if pd.notna(plan["stop_loss_price"]):
            exit_at, reason = HOLD_SESSIONS, "TIME"
            for day in range(1, HOLD_SESSIONS + 1):
                price = path.iloc[day]
                if price <= plan["stop_loss_price"]:
                    exit_at, reason = day, "STOP"
                    break
                if price >= plan["target_price"]:
                    exit_at, reason = day, "TARGET"
                    break
            exits["trade_plan"] = (exit_at, reason)
        for mode, (day, reason) in exits.items():
            stock_return = (path.iloc[day] / start - 1) * 100
            market_return = (market_path.iloc[day] / market_path.iloc[0] - 1) * 100
            trades.append(
                {
                    "symbol": row["symbol"],
                    "exit_mode": mode,
                    "exit_reason": reason,
                    "sessions_held": day,
                    "return_net_pct": stock_return - round_trip,
                    "excess_net_pct": stock_return - market_return - round_trip,
                }
            )
    return trades


def summarize_strategies(trades: pd.DataFrame, step: int) -> pd.DataFrame:
    """Per strategy and exit mode: trade and per-rebalance portfolio statistics."""
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (strategy, mode), group in trades.groupby(["strategy", "exit_mode"], sort=True):
        portfolio = group.groupby("as_of").excess_net_pct.mean()
        n_effective = max(len(portfolio) * min(1.0, step / HOLD_SESSIONS), 1.0)
        std = portfolio.std(ddof=1) if len(portfolio) > 1 else np.nan
        rows.append(
            {
                "strategy": strategy,
                "exit_mode": mode,
                "rebalances": len(portfolio),
                "trades": len(group),
                "mean_trade_excess_net_pct": group.excess_net_pct.mean(),
                "median_trade_excess_net_pct": group.excess_net_pct.median(),
                "trade_win_rate": group.excess_net_pct.gt(0).mean(),
                "mean_portfolio_excess_net_pct": portfolio.mean(),
                "portfolio_t_stat": portfolio.mean() / (std / np.sqrt(n_effective))
                if std and np.isfinite(std) and std > 0 else np.nan,
                "portfolio_win_rate": portfolio.gt(0).mean(),
                "worst_portfolio_excess_pct": portfolio.min(),
                "mean_sessions_held": group.sessions_held.mean(),
                "stop_rate": group.exit_reason.eq("STOP").mean(),
                "target_rate": group.exit_reason.eq("TARGET").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["exit_mode", "mean_portfolio_excess_net_pct"], ascending=[True, False]
    ).reset_index(drop=True)


def _forward_returns(
    closes: pd.DataFrame, bench: pd.DataFrame, position: int, horizon: int,
    symbols: pd.Series, benchmark_symbols: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Enter at the next session's close and exit ``horizon`` sessions later."""
    entry, exit_ = position + 1, position + 1 + horizon
    if exit_ >= len(closes):
        nan = pd.Series(np.nan, index=symbols.index)
        return nan, nan
    stock = (closes.iloc[exit_] / closes.iloc[entry] - 1) * 100
    market = (bench.iloc[exit_] / bench.iloc[entry] - 1) * 100
    raw = symbols.map(stock)
    excess = raw - benchmark_symbols.map(market)
    return raw, excess


def run_backtest(
    histories: pd.DataFrame,
    universe: pd.DataFrame,
    step: int = 5,
    cost_bps: float = 10.0,
    horizons: tuple[int, ...] = HORIZONS,
    progress=None,
    earnings_events: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (observations, signal summary, trades, strategy summary)."""
    universe = universe.drop_duplicates("symbol").copy()
    universe["sector"] = universe.get("sector", pd.Series("", index=universe.index)).fillna("")
    symbols = universe.symbol.tolist()
    benchmark_symbols = sorted({value[0] for value in BENCHMARKS.values()})
    closes = _close_panel(histories, symbols)
    bench = _close_panel(histories, benchmark_symbols)
    if closes.empty or bench.empty:
        raise ValueError("backtest needs stock and benchmark price history")
    bench = bench.reindex(closes.index).ffill(limit=5)
    stock_history = _symbol_histories(histories, symbols)
    bench_history = _symbol_histories(histories, benchmark_symbols)
    last = len(closes) - 2 - max(max(horizons), HOLD_SESSIONS)
    positions = list(range(MIN_HISTORY_SESSIONS, last + 1, step))
    event_tables = {}
    if earnings_events is not None:
        regions = dict(zip(universe.symbol, universe.index_name.map(lambda v: assign_benchmark(v)[0])))
        for symbol, events in earnings_events.items():
            if symbol in closes:
                event_tables[symbol] = earnings_event_table(
                    events, closes[symbol], bench.get(regions.get(symbol))
                )
    rows, trades = [], []
    for count, position in enumerate(positions, 1):
        as_of = closes.index[position]
        features = _features_as_of(stock_history, symbols, as_of)
        benchmark_features = _features_as_of(bench_history, benchmark_symbols, as_of)
        if features.empty or benchmark_features.empty:
            continue
        scored = score_cross_section(features, universe, benchmark_features)
        scored["as_of"] = as_of
        if earnings_events is not None:
            scored["earnings_drift"] = drift_as_of(event_tables, scored.symbol, as_of)
        for name, signal in strategy_definitions(scored).items():
            for trade in simulate_trades(_pick(scored, signal), closes, bench, position, cost_bps):
                trades.append({"as_of": as_of, "strategy": name, **trade})
        for horizon in horizons:
            raw, excess = _forward_returns(
                closes, bench, position, horizon, scored.symbol, scored.benchmark_symbol
            )
            scored[f"forward_{horizon}d_return"] = raw
            scored[f"forward_{horizon}d_excess"] = excess
        rows.append(scored)
        if progress:
            progress(count, len(positions), as_of)
    if not rows:
        raise ValueError("not enough history for any rebalance date")
    panel = pd.concat(rows, ignore_index=True)
    trade_frame = pd.DataFrame(trades)
    return (
        panel,
        summarize(panel, step, cost_bps, horizons),
        trade_frame,
        summarize_strategies(trade_frame, step),
    )


def _signal_columns(panel: pd.DataFrame) -> list[str]:
    excluded = {"symbol", "sector", "benchmark_symbol", "market_regime", "as_of",
                "volatility_20d", "volatility_60d"}
    return [c for c in panel if c not in excluded and not c.startswith("forward_")]


def summarize(
    panel: pd.DataFrame, step: int, cost_bps: float, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Per-signal information coefficient and portfolio statistics.

    IC is the per-date Spearman correlation between signal and forward excess
    return.  Its t-statistic uses an effective sample of dates × step/horizon
    because overlapping holding windows are not independent.  Top-N returns
    subtract a round-trip cost on every rebalance.
    """
    output = []
    round_trip = 2 * cost_bps / 100
    for horizon in horizons:
        target = f"forward_{horizon}d_excess"
        for signal in _signal_columns(panel):
            per_date = []
            for as_of, group in panel.groupby("as_of", sort=True):
                valid = group[[signal, target, "market_regime"]].dropna(subset=[signal, target])
                if len(valid) < 30 or valid[signal].nunique() < 2:
                    continue
                ranked = valid.sort_values(signal, ascending=False)
                decile = max(len(valid) // 10, 1)
                per_date.append(
                    {
                        "as_of": as_of,
                        "ic": valid[signal].corr(valid[target], method="spearman"),
                        "spread": ranked[target].head(decile).mean() - ranked[target].tail(decile).mean(),
                        "top_n_net": ranked[target].head(TOP_N).mean() - round_trip,
                        "risk_on": valid.market_regime.eq(RISK_ON).mean() >= 0.5,
                    }
                )
            if not per_date:
                continue
            frame = pd.DataFrame(per_date)
            n_effective = max(len(frame) * min(1.0, step / horizon), 1.0)
            ic_std = frame.ic.std(ddof=1) if len(frame) > 1 else np.nan
            output.append(
                {
                    "horizon_sessions": horizon,
                    "signal": signal,
                    "dates": len(frame),
                    "mean_ic": frame.ic.mean(),
                    "ic_t_stat": frame.ic.mean() / (ic_std / np.sqrt(n_effective))
                    if ic_std and np.isfinite(ic_std) and ic_std > 0 else np.nan,
                    "ic_hit_rate": frame.ic.gt(0).mean(),
                    "mean_decile_spread_pct": frame.spread.mean(),
                    "mean_top_n_excess_net_pct": frame.top_n_net.mean(),
                    "mean_ic_risk_on": frame.loc[frame.risk_on, "ic"].mean(),
                    "mean_ic_other_regimes": frame.loc[~frame.risk_on, "ic"].mean(),
                }
            )
    summary = pd.DataFrame(output)
    if summary.empty:
        return summary
    return summary.sort_values(["horizon_sessions", "mean_ic"], ascending=[True, False]).reset_index(drop=True)


def latest_universe(runs_dir: Path = RUNS_DIR, minimum_rows: int = 100) -> pd.DataFrame:
    """Universe and canonical sectors from the newest completed normal run."""
    candidates = sorted(
        (
            path for path in Path(runs_dir).glob("*")
            if (path / "full_analysis.csv").exists()
            and "-replay-" not in path.name and "-rescore-" not in path.name
        ),
        reverse=True,
    )
    for candidate in candidates:
        full = pd.read_csv(candidate / "full_analysis.csv", low_memory=False)
        if len(full) < minimum_rows:
            continue
        columns = [c for c in ("symbol", "security_id", "index_name", "sector") if c in full]
        return full[columns].dropna(subset=["symbol"]).drop_duplicates("symbol")
    raise FileNotFoundError("no completed run found; run `python main.py` first")


def load_histories(universe: pd.DataFrame, years: float | None) -> pd.DataFrame:
    benchmark_symbols = sorted({value[0] for value in BENCHMARKS.values()})
    symbols = universe.symbol.tolist() + benchmark_symbols
    if years:
        from investment_ai.data.prices import download_prices_range

        start = date.today() - timedelta(days=int(years * 365.25) + 30)
        return download_prices_range(symbols, start, date.today() + timedelta(days=1))
    from investment_ai.data.price_store import PriceStore

    ids = dict(zip(universe.symbol, universe.get("security_id", universe.symbol)))
    connection = sqlite3.connect(HISTORY_DB)
    try:
        store = PriceStore(connection)
        securities = {
            (ids.get(symbol) if pd.notna(ids.get(symbol)) else None)
            or store.security_id_for_symbol(symbol) or symbol: symbol
            for symbol in symbols
        }
        return store.histories(securities)
    finally:
        connection.close()


def run_backtest_cli(
    years: float | None = None, step: int = 5, cost_bps: float = 10.0, earnings: bool = False
) -> int:
    if step < 1:
        raise ValueError("--backtest-step must be >= 1")
    universe = latest_universe()
    histories = load_histories(universe, years)
    events = None
    if earnings:
        from investment_ai.config import MAX_WORKERS

        print("loading earnings history for the drift backtest...", flush=True)
        events = load_earnings_events(universe.symbol.tolist(), MAX_WORKERS)

    def progress(count: int, total: int, as_of: pd.Timestamp) -> None:
        if count == 1 or count % 10 == 0 or count == total:
            print(f"backtest {count}/{total} as of {as_of.date()}", flush=True)

    panel, summary, trades, strategies = run_backtest(
        histories, universe, step, cost_bps, progress=progress, earnings_events=events
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(BACKTEST_DIR) / f"backtest-{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory / "signal_summary.csv", index=False)
    strategies.to_csv(directory / "strategy_summary.csv", index=False)
    trades.to_csv(directory / "trades.csv", index=False)
    panel.to_csv(directory / "observations.csv", index=False)
    manifest = {
        "scoring_model_version": SCORING_MODEL_VERSION,
        "created_at_utc": stamp,
        "price_source": f"download {years}y" if years else "local price store",
        "universe_count": int(universe.symbol.nunique()),
        "rebalance_dates": int(panel.as_of.nunique()),
        "first_date": str(panel.as_of.min()),
        "last_date": str(panel.as_of.max()),
        "step_sessions": step,
        "one_way_cost_bps": cost_bps,
        "entry": "next session close after the signal date",
        "earnings_drift_included": bool(earnings),
        "strategy_rules": {
            "picks": STRATEGY_PICKS,
            "max_per_sector": STRATEGY_MAX_PER_SECTOR,
            "holding_sessions": HOLD_SESSIONS,
            "trade_plan_exit": "first close at or below stop or at or above target",
        },
        "target": "forward excess return over the assigned regional benchmark",
        "biases": [
            "survivorship: today's constituents only",
            "sector labels are today's",
            "analyst, valuation and fundamental pillars are not replayable",
        ],
    }
    (directory / "backtest_manifest.json").write_text(json.dumps(manifest, indent=2))
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary.round(3).to_string(index=False))
        print("\nSTRATEGIES (top 20, max 3 per sector, net of costs, excess vs benchmark)\n")
        print(strategies.round(3).to_string(index=False))
    print(f"\nBacktest written to {directory}")
    return 0
