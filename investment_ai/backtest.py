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
    price_features,
    setup_scores,
)
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
PRICE_ONLY_PILLARS = ("momentum", "setup", "volume", "technical")


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
        weights = {key: SHORT_TERM_WEIGHTS[key] for key in PRICE_ONLY_PILLARS}
        composite = [
            weighted(
                {"momentum": m, "setup": s["setup_quality_score"], "volume": v, "technical": t},
                weights,
                0.6,
            )[0]
            for m, s, v, t in zip(momentum, setups, volume, technical)
        ]
        signals[f"momentum_pillar_{basis}"] = momentum
        signals[f"st_price_composite_{basis}"] = composite
        signals[f"long_trend_pillar_{basis}"] = [long_trend_pillar(row)[0] for row in records]
        if basis == "benchmark":
            signals["setup_quality"] = [s["setup_quality_score"] for s in setups]
            signals["credible_setup"] = [
                100.0 if s["short_term_setup"] in CREDIBLE_SETUPS else 0.0 for s in setups
            ]
            signals["volume_pillar"] = volume
            signals["technical_pillar"] = technical
            signals["credible_composite_benchmark"] = [
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
    output = frame[["symbol", "sector", "benchmark_symbol", "market_regime"]].copy()
    for name, values in signals.items():
        output[name] = values
    return output


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-date observations, per-signal summary)."""
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
    last = len(closes) - 2 - max(horizons)
    positions = list(range(MIN_HISTORY_SESSIONS, last + 1, step))
    rows = []
    for count, position in enumerate(positions, 1):
        as_of = closes.index[position]
        features = _features_as_of(stock_history, symbols, as_of)
        benchmark_features = _features_as_of(bench_history, benchmark_symbols, as_of)
        if features.empty or benchmark_features.empty:
            continue
        scored = score_cross_section(features, universe, benchmark_features)
        scored["as_of"] = as_of
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
    return panel, summarize(panel, step, cost_bps, horizons)


def _signal_columns(panel: pd.DataFrame) -> list[str]:
    excluded = {"symbol", "sector", "benchmark_symbol", "market_regime", "as_of"}
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


def run_backtest_cli(years: float | None = None, step: int = 5, cost_bps: float = 10.0) -> int:
    if step < 1:
        raise ValueError("--backtest-step must be >= 1")
    universe = latest_universe()
    histories = load_histories(universe, years)

    def progress(count: int, total: int, as_of: pd.Timestamp) -> None:
        if count == 1 or count % 10 == 0 or count == total:
            print(f"backtest {count}/{total} as of {as_of.date()}", flush=True)

    panel, summary = run_backtest(histories, universe, step, cost_bps, progress=progress)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(BACKTEST_DIR) / f"backtest-{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory / "signal_summary.csv", index=False)
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
    print(f"\nBacktest written to {directory}")
    return 0
