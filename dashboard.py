"""Investment AI dashboard.  Run with:  streamlit run dashboard.py

Read-only view of the latest `python main.py` results: recommendations,
filtering, per-stock detail, sector views, live re-weighting and the backtest
evidence behind the model.  Research support only, not investment advice.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from investment_ai.config import BACKTEST_DIR, HISTORY_DB, RUNS_DIR
from investment_ai.dashboard_data import (
    LONG_TERM_PILLARS,
    SHORT_TERM_PILLARS,
    apply_filters,
    best_in_both,
    latest_backtest,
    list_runs,
    load_run,
    pillar_breakdown,
    price_history,
    rerank,
)

# Reference palette (validated): series blue, highlight orange, diverging
# blue/red around a gray midpoint, recessive gray for context marks.
BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#e34948", "#b5b4ae"
STATUS = {"VALID": ("good", "✅"), "DEGRADED": ("warning", "⚠️"), "INVALID": ("critical", "⛔")}

st.set_page_config(page_title="Investment AI", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def _runs() -> list[dict]:
    return [
        {"run_id": run["run_id"], "path": str(run["path"]), "manifest": run["manifest"]}
        for run in list_runs(RUNS_DIR)
    ]


@st.cache_data(show_spinner=False)
def _load(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from pathlib import Path

    return load_run(Path(path))


@st.cache_data(show_spinner=False)
def _prices(symbol: str) -> pd.DataFrame:
    return price_history(HISTORY_DB, symbol)


@st.cache_data(show_spinner=False)
def _backtest() -> dict:
    return latest_backtest(BACKTEST_DIR)


def score_column(label: str, help_text: str | None = None):
    return st.column_config.ProgressColumn(
        label, min_value=0, max_value=100, format="%.0f", help=help_text
    )


COMMON_CONFIG = {
    "symbol": st.column_config.TextColumn("Symbol", width="small"),
    "company_name": st.column_config.TextColumn("Company"),
    "sector": st.column_config.TextColumn("Sector"),
    "index_name": st.column_config.TextColumn("Index"),
    "current_price": st.column_config.NumberColumn("Price", format="%.2f"),
    "long_term_rank": st.column_config.NumberColumn("LT rank", format="%d"),
    "short_term_rank": st.column_config.NumberColumn("ST rank", format="%d"),
    "action_rank": st.column_config.NumberColumn("#", format="%d", width="small"),
    "long_term_score": score_column("LT score"),
    "short_term_score": score_column("ST score"),
    "long_term_risk_adjusted_score": score_column("LT score (risk-adj.)"),
    "short_term_risk_adjusted_score": score_column("ST score (risk-adj.)"),
    "risk_score": score_column("Risk", "Higher = riskier"),
    "confidence_score": score_column("Confidence", "Data coverage and freshness"),
    "stop_loss_price": st.column_config.NumberColumn("Stop", format="%.2f"),
    "target_price": st.column_config.NumberColumn("Target", format="%.2f"),
    "stop_loss_pct": st.column_config.NumberColumn("Stop %", format="%.1f%%"),
    "portfolio_weight_pct": st.column_config.NumberColumn("Weight", format="%.1f%%"),
    "target_upside_pct": st.column_config.NumberColumn("Analyst upside", format="%.1f%%"),
    "momentum_12_1_pct": st.column_config.NumberColumn("12-1m return", format="%.1f%%"),
    "return_20d_pct": st.column_config.NumberColumn("20d return", format="%.1f%%"),
    "days_to_next_earnings": st.column_config.NumberColumn("Days to earnings", format="%.0f"),
    "custom_score": score_column("Your score"),
    "custom_rank": st.column_config.NumberColumn("Your rank", format="%d"),
    "st_positive_drivers": st.column_config.TextColumn("Why (short term)", width="large"),
    "lt_positive_drivers": st.column_config.TextColumn("Why (long term)", width="large"),
}


def show_table(frame: pd.DataFrame, columns: list[str], height: int | None = None) -> None:
    present = [column for column in columns if column in frame]
    sizing = {"height": height} if height else {}
    st.dataframe(
        frame[present],
        column_config={key: value for key, value in COMMON_CONFIG.items() if key in present},
        hide_index=True,
        width="stretch",
        **sizing,
    )


# --------------------------------------------------------------------- sidebar
runs = _runs()
if not runs:
    st.error("No completed runs found. Run `python main.py` first.")
    st.stop()

with st.sidebar:
    st.title("📈 Investment AI")
    labels = {
        run["run_id"]: f"{run['run_id'][:8]} {run['run_id'][9:13]}Z · "
        f"{run['manifest'].get('overall_run_status', '?')} · "
        f"model {run['manifest'].get('scoring_model_version', '?')}"
        for run in runs
    }
    run_id = st.selectbox("Run", list(labels), format_func=labels.get)
    run = next(item for item in runs if item["run_id"] == run_id)
    manifest = run["manifest"]
    full, action = _load(run["path"])

    status = manifest.get("overall_run_status", "UNKNOWN")
    _, icon = STATUS.get(status, ("warning", "❔"))
    st.markdown(f"**Run status:** {icon} {status}")
    for reason in manifest.get("run_status_reasons", [])[:5]:
        st.caption(f"• {reason}")
    regimes = manifest.get("market_regime_counts") or {}
    if regimes:
        st.markdown("**Market regime:** " + ", ".join(f"{k} ({v})" for k, v in regimes.items()))
    st.caption(f"Prices as of {str(full.get('price_as_of', pd.Series([''])).dropna().max())[:10]}")

    st.divider()
    st.subheader("Filters")
    index_choice = st.multiselect("Index", ["S&P 500", "STOXX Europe 600"])
    sector_choice = st.multiselect("Sector", sorted(full.sector.dropna().unique()))
    min_confidence = st.slider("Minimum confidence", 0, 100, 0, step=5)
    skip_earnings = st.checkbox("Hide stocks reporting earnings within 2 days")
    st.divider()
    st.caption(
        "Research support only — not investment advice. Scores are deterministic "
        "and auditable; see the Model evidence tab for what has been tested."
    )

view = apply_filters(full, index_choice, sector_choice, min_confidence, skip_earnings)
view_symbols = set(view.symbol)
action_view = action[action.symbol.isin(view_symbols)] if len(action) else action

st.title("Investment AI")
cols = st.columns(5)
cols[0].metric("Stocks analysed", f"{len(full):,}")
cols[1].metric("Long-term ranked", f"{int(full.long_term_rank.notna().sum()):,}")
cols[2].metric("Short-term ranked", f"{int(full.short_term_rank.notna().sum()):,}")
cols[3].metric("Action list", f"{len(action)}")
cols[4].metric("Model", str(manifest.get("scoring_model_version", "?")))
if status == "INVALID":
    st.error("This run is INVALID — rankings are diagnostic only.")
elif status == "DEGRADED":
    st.warning("This run is DEGRADED — some data coverage is below target. See the sidebar for reasons.")

tabs = st.tabs(
    ["Recommendations", "Explore", "Stock detail", "Sectors", "Play with weights", "Model evidence"]
)

# --------------------------------------------------------------- recommendations
with tabs[0]:
    st.subheader("Short-term action list")
    st.caption(
        "Strongest short-term rankings inside the top half of the long-term ranking, "
        "max 3 per sector. Hold about 10 sessions; the stop is two standard deviations "
        "of a 10-session move, the target 1.5× that distance. Weights sum to ≤100%."
    )
    if len(action_view):
        show_table(
            action_view,
            ["action_rank", "symbol", "company_name", "sector", "current_price",
             "stop_loss_price", "target_price", "stop_loss_pct", "portfolio_weight_pct",
             "short_term_risk_adjusted_score", "long_term_rank", "st_top_positive_driver"],
        )
        chart_data = action_view.assign(
            stop_gap=lambda d: (d.stop_loss_price / d.current_price - 1) * 100,
            target_gap=lambda d: (d.target_price / d.current_price - 1) * 100,
        )
        base = alt.Chart(chart_data).encode(
            y=alt.Y("symbol:N", sort=alt.EncodingSortField("action_rank"), title=None)
        )
        rule = base.mark_rule(strokeWidth=2, color=GRAY).encode(
            x=alt.X("stop_gap:Q", title="Distance from current price (%)"), x2="target_gap:Q"
        )
        stop = base.mark_point(filled=True, size=70, color=RED).encode(
            x="stop_gap:Q",
            tooltip=[alt.Tooltip("symbol:N"), alt.Tooltip("stop_loss_price:Q", title="Stop", format=".2f"),
                     alt.Tooltip("stop_gap:Q", title="Stop %", format=".1f")],
        )
        target = base.mark_point(filled=True, size=70, color=BLUE).encode(
            x="target_gap:Q",
            tooltip=[alt.Tooltip("symbol:N"), alt.Tooltip("target_price:Q", title="Target", format=".2f"),
                     alt.Tooltip("target_gap:Q", title="Target %", format=".1f")],
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#52514e").encode(x="x:Q")
        st.markdown("**Trade plan ranges** — red: stop, blue: target, line at 0: today's price")
        st.altair_chart((rule + stop + target + zero).properties(height=max(220, 22 * len(chart_data))),
                        width="stretch")
    else:
        st.info("No action-list names match the current filters.")

    left, right = st.columns(2)
    with left:
        st.subheader("Best long-term holdings")
        top_lt = view[view.long_term_rank.notna()].sort_values("long_term_rank").head(15)
        show_table(top_lt, ["long_term_rank", "symbol", "company_name", "sector",
                            "long_term_risk_adjusted_score", "risk_score", "lt_positive_drivers"])
    with right:
        st.subheader("Strong on both horizons")
        st.caption("Top quarter of both the long-term and short-term rankings.")
        both = best_in_both(view).head(15)
        show_table(both, ["symbol", "company_name", "sector", "long_term_rank",
                          "short_term_rank", "long_term_score", "short_term_score"])

# ----------------------------------------------------------------------- explore
with tabs[1]:
    st.subheader("Long-term vs short-term score")
    plot = view.dropna(subset=["long_term_score", "short_term_score"]).copy()
    plot["group"] = plot.symbol.isin(set(action.symbol)).map({True: "Action list", False: "Other ranked"})
    scatter = (
        alt.Chart(plot)
        .mark_circle(size=60, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("long_term_score:Q", title="Long-term score", scale=alt.Scale(zero=False)),
            y=alt.Y("short_term_score:Q", title="Short-term score", scale=alt.Scale(zero=False)),
            color=alt.Color("group:N", scale=alt.Scale(domain=["Action list", "Other ranked"],
                                                      range=[ORANGE, GRAY]), title=None),
            opacity=alt.condition(alt.datum.group == "Action list", alt.value(1), alt.value(0.55)),
            tooltip=["symbol", "company_name", "sector",
                     alt.Tooltip("long_term_score:Q", format=".1f"),
                     alt.Tooltip("short_term_score:Q", format=".1f"),
                     alt.Tooltip("risk_score:Q", format=".0f")],
        )
        .properties(height=460)
        .interactive()
    )
    st.altair_chart(scatter, width="stretch")
    st.caption("Scroll to zoom, drag to pan. Top-right = attractive on both horizons.")

    st.subheader("All stocks")
    search = st.text_input("Search symbol or company", "")
    table = view
    if search:
        needle = search.lower()
        table = view[view.symbol.str.lower().str.contains(needle, regex=False)
                     | view.company_name.fillna("").str.lower().str.contains(needle, regex=False)]
    default_columns = ["symbol", "company_name", "sector", "index_name", "current_price",
                       "long_term_rank", "long_term_score", "short_term_rank", "short_term_score",
                       "risk_score", "confidence_score", "momentum_12_1_pct", "target_upside_pct",
                       "short_term_setup", "days_to_next_earnings"]
    chosen = st.multiselect("Columns", sorted(full.columns), default=default_columns)
    sort_by = st.selectbox("Sort by", ["long_term_rank", "short_term_rank", "long_term_score",
                                        "short_term_score", "risk_score", "momentum_12_1_pct"])
    ascending = sort_by.endswith("rank") or sort_by == "risk_score"
    show_table(table.sort_values(sort_by, ascending=ascending, na_position="last"), chosen, height=520)
    st.download_button("Download filtered table (CSV)", table[chosen].to_csv(index=False),
                       file_name=f"investment_ai_{run_id}.csv", mime="text/csv")

# ------------------------------------------------------------------ stock detail
with tabs[2]:
    ordered = full.sort_values(["long_term_rank", "short_term_rank"], na_position="last")
    options = ordered.symbol.tolist()
    default_symbol = action.symbol.iloc[0] if len(action) else options[0]
    symbol = st.selectbox(
        "Stock",
        options,
        index=options.index(default_symbol) if default_symbol in options else 0,
        format_func=lambda s: f"{s} — {full.loc[full.symbol.eq(s), 'company_name'].iloc[0]}",
    )
    row = full[full.symbol.eq(symbol)].iloc[0]
    st.subheader(f"{row.get('company_name', symbol)} ({symbol})")
    st.caption(f"{row.get('sector', '')} · {row.get('index_name', '')} · setup: "
               f"{row.get('short_term_setup', 'n/a')} · fundamentals: {row.get('fundamentals_basis', 'n/a')}")

    def fmt(value, pattern="{:.1f}"):
        return "—" if pd.isna(value) else pattern.format(value)

    metrics = st.columns(6)
    metrics[0].metric("Price", fmt(row.get("current_price"), "{:,.2f}"))
    metrics[1].metric("LT rank", fmt(row.get("long_term_rank"), "{:.0f}"), help="Out of all long-term ranked")
    metrics[2].metric("ST rank", fmt(row.get("short_term_rank"), "{:.0f}"))
    metrics[3].metric("Risk", fmt(row.get("risk_score"), "{:.0f}"))
    metrics[4].metric("Analyst upside", fmt(row.get("target_upside_pct"), "{:.1f}%"))
    metrics[5].metric("Next earnings", fmt(row.get("days_to_next_earnings"), "{:.0f} days"))

    history = _prices(symbol)
    if len(history):
        long_form = history.melt("date", ["close", "ma_50", "ma_200"], "series", "price").dropna()
        names = {"close": "Close", "ma_50": "50-day average", "ma_200": "200-day average"}
        long_form["series"] = long_form.series.map(names)
        price_chart = alt.Chart(long_form).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("price:Q", title="Adjusted close", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", scale=alt.Scale(
                domain=list(names.values()), range=[BLUE, ORANGE, "#1baf7a"]), title=None),
            tooltip=[alt.Tooltip("date:T"), "series:N", alt.Tooltip("price:Q", format=",.2f")],
        )
        layers = [price_chart]
        levels = [(row.get("stop_loss_price"), "Stop", RED), (row.get("target_price"), "Target", BLUE)]
        for value, label, color in levels:
            if pd.notna(value):
                layers.append(alt.Chart(pd.DataFrame({"y": [value], "label": [label]}))
                              .mark_rule(strokeDash=[6, 4], color=color)
                              .encode(y="y:Q", tooltip=["label:N", alt.Tooltip("y:Q", format=",.2f")]))
        st.altair_chart(alt.layer(*layers).properties(height=340).interactive(bind_y=False),
                        width="stretch")
        st.caption("Dashed lines: suggested stop (red) and target (blue) for a ~10-session hold.")
    else:
        st.info("No local price history for this symbol.")

    def breakdown_chart(pillars, title):
        data = pillar_breakdown(row, pillars)
        data = data[data.weight > 0]
        data["direction"] = data.contribution.ge(0).map({True: "Above neutral", False: "Below neutral"})
        chart = alt.Chart(data.dropna(subset=["score"])).mark_bar(cornerRadiusEnd=4, height=16).encode(
            y=alt.Y("pillar:N", sort=None, title=None),
            x=alt.X("contribution:Q", title="Points vs neutral (weight × (score − 50))"),
            color=alt.Color("direction:N", scale=alt.Scale(
                domain=["Above neutral", "Below neutral"], range=[BLUE, RED]), title=None),
            tooltip=["pillar:N", alt.Tooltip("score:Q", format=".0f"),
                     alt.Tooltip("weight:Q", format=".0%"), alt.Tooltip("contribution:Q", format="+.1f")],
        )
        st.markdown(f"**{title}**")
        st.altair_chart(chart.properties(height=40 * max(len(data), 1)), width="stretch")
        missing = data[data.score.isna()].pillar.tolist()
        if missing:
            st.caption("Missing (counted as neutral): " + ", ".join(missing))

    left, right = st.columns(2)
    with left:
        breakdown_chart(LONG_TERM_PILLARS, f"Long-term score {fmt(row.get('long_term_score'))}")
        st.caption(str(row.get("lt_positive_drivers", "")))
        st.caption(str(row.get("lt_negative_drivers", "")))
    with right:
        breakdown_chart({k: v for k, v in SHORT_TERM_PILLARS.items() if v[1] > 0},
                        f"Short-term score {fmt(row.get('short_term_score'))}")
        st.caption(str(row.get("st_positive_drivers", "")))
        st.caption(str(row.get("st_negative_drivers", "")))

    with st.expander("All fields for this stock"):
        st.dataframe(row.dropna().astype(str).rename("value").to_frame(), width="stretch")

# ----------------------------------------------------------------------- sectors
with tabs[3]:
    ranked = view[view.long_term_score.notna() | view.short_term_score.notna()]
    by_sector = ranked.groupby("sector").agg(
        stocks=("symbol", "count"),
        long_term=("long_term_score", "median"),
        short_term=("short_term_score", "median"),
        risk=("risk_score", "median"),
        momentum=("momentum_12_1_pct", "median"),
    ).reset_index()
    metric = st.radio("Measure", ["long_term", "short_term", "momentum", "risk"], horizontal=True,
                      format_func={"long_term": "Median long-term score", "short_term": "Median short-term score",
                                   "momentum": "Median 12-1 month return (%)", "risk": "Median risk"}.get)
    sector_chart = alt.Chart(by_sector).mark_bar(cornerRadiusEnd=4, color=BLUE, height=18).encode(
        y=alt.Y("sector:N", sort="-x", title=None),
        x=alt.X(f"{metric}:Q", title=None),
        tooltip=["sector:N", "stocks:Q", alt.Tooltip(f"{metric}:Q", format=".1f")],
    )
    st.altair_chart(sector_chart.properties(height=34 * max(len(by_sector), 1)), width="stretch")
    st.markdown("**Where the recommendations come from**")
    counts = pd.DataFrame({
        "Long-term top 50": view[view.long_term_rank.le(50)].sector.value_counts(),
        "Short-term top 50": view[view.short_term_rank.le(50)].sector.value_counts(),
        "Action list": action_view.sector.value_counts() if len(action_view) else pd.Series(dtype=int),
    }).fillna(0).astype(int).sort_values("Long-term top 50", ascending=False)
    st.dataframe(counts, width="stretch")

# ------------------------------------------------------------- play with weights
with tabs[4]:
    st.write(
        "Change the pillar weights and see how the ranking changes. Missing pillars count as "
        "neutral (50). This does not change the saved results; the defaults are the production model."
    )
    horizon = st.radio("Horizon", ["Short term", "Long term"], horizontal=True)
    pillars = SHORT_TERM_PILLARS if horizon == "Short term" else LONG_TERM_PILLARS
    eligible_column = "short_term_score" if horizon == "Short term" else "long_term_score"
    model_rank = "short_term_rank" if horizon == "Short term" else "long_term_rank"
    weights = {}
    slider_cols = st.columns(len(pillars))
    for column, (key, (label, default)) in zip(slider_cols, pillars.items()):
        weights[key] = column.slider(label, 0, 100, int(round(default * 100)), step=5, key=f"{horizon}-{key}") / 100
    if sum(weights.values()) == 0:
        st.warning("Give at least one pillar a weight.")
    else:
        eligible = view[eligible_column].notna()
        scored = view.join(rerank(view, weights, eligible))
        top = scored[scored.custom_rank.notna()].sort_values("custom_rank").head(25).copy()
        top["model_rank_change"] = top[model_rank] - top["custom_rank"]
        show_table(top, ["custom_rank", "symbol", "company_name", "sector", "custom_score",
                         model_rank, "risk_score"])
        overlap = len(set(top.symbol) & set(scored.sort_values(model_rank).head(25).symbol))
        st.caption(f"{overlap} of your top 25 are also in the model's top 25 for these filters.")
        if horizon == "Short term":
            st.info(
                "Backtest note: momentum alone earned +0.71% excess per 10-session trade "
                "(2021–2026); adding setup, volume and technical pillars cut that to +0.06%."
            )

# ---------------------------------------------------------------- model evidence
with tabs[5]:
    backtest = _backtest()
    if not backtest:
        st.info("No backtest yet. Run `python main.py --backtest --backtest-years 5 --backtest-earnings`.")
    else:
        info = backtest["manifest"]
        st.caption(
            f"Backtest {backtest['path'].name}: {info.get('rebalance_dates', '?')} weekly dates, "
            f"{str(info.get('first_date', ''))[:10]} → {str(info.get('last_date', ''))[:10]}, "
            f"{info.get('one_way_cost_bps', '?')} bps cost each way. In-sample and subject to "
            "survivorship bias — expect live results to be weaker."
        )
        strategies = backtest["strategies"]
        if len(strategies):
            st.markdown("**Strategy results** — top 20 picks, excess return vs benchmark per 10-session trade, net of costs")
            hold = strategies[strategies.exit_mode.eq("hold_10")]
            strategy_chart = alt.Chart(hold).mark_bar(cornerRadiusEnd=4, color=BLUE, height=18).encode(
                y=alt.Y("strategy:N", sort="-x", title=None),
                x=alt.X("mean_portfolio_excess_net_pct:Q", title="Mean excess return per trade (%)"),
                tooltip=["strategy:N", alt.Tooltip("mean_portfolio_excess_net_pct:Q", format=".2f", title="Excess %"),
                         alt.Tooltip("portfolio_t_stat:Q", format=".2f", title="t-stat"),
                         alt.Tooltip("portfolio_win_rate:Q", format=".0%", title="Win rate")],
            )
            st.altair_chart(strategy_chart.properties(height=40 * max(len(hold), 1)), width="stretch")
            st.dataframe(strategies.round(3), hide_index=True, width="stretch")
        signals = backtest["signals"]
        horizon_choice = st.select_slider("Signal horizon (sessions)", sorted(signals.horizon_sessions.unique()))
        subset = signals[signals.horizon_sessions.eq(horizon_choice)].copy()
        subset["evidence"] = subset.ic_t_stat.abs().ge(2).map({True: "|t| ≥ 2", False: "|t| < 2"})
        ic_chart = alt.Chart(subset).mark_bar(cornerRadiusEnd=4, height=12).encode(
            y=alt.Y("signal:N", sort="-x", title=None),
            x=alt.X("mean_ic:Q", title="Mean information coefficient (rank correlation with future excess return)"),
            color=alt.Color("evidence:N", scale=alt.Scale(domain=["|t| ≥ 2", "|t| < 2"], range=[BLUE, GRAY]),
                            title="Significance"),
            tooltip=["signal:N", alt.Tooltip("mean_ic:Q", format=".4f"),
                     alt.Tooltip("ic_t_stat:Q", format=".2f", title="t-stat"),
                     alt.Tooltip("ic_hit_rate:Q", format=".0%", title="Positive dates")],
        )
        st.markdown("**Signal strength**")
        st.altair_chart(ic_chart.properties(height=18 * max(len(subset), 1)), width="stretch")

    st.markdown("**Live validation**")
    st.caption(
        "Live evidence for the analyst and fundamental pillars builds up from daily runs. "
        "Check it with `python main.py --validation-report` once outcomes have matured "
        "(about 4 weeks for 20-day returns, about 3 months for a usable sample)."
    )
