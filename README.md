# Investment AI v3

A deterministic, point-in-time research scanner for the combined **S&P 500 and STOXX Europe 600** universe. It produces exactly two primary views: a long-term investment ranking and a short-term opportunity ranking. It does not trade, place orders, use an LLM for scoring, or claim guaranteed profitability.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Optional environment controls (the command remains the same program):

```bash
EXPORT_RESULTS=true python main.py
FORCE_REFRESH=true python main.py
```

`TOP_N` changes display length only; every deduplicated constituent is scored before ranking. Duplicate Yahoo symbols retain all index memberships.

## What the scores mean

**Long Term (1–3 year research horizon)** is 25% quality, 20% growth, 20% valuation, 20% long-term expectations, 10% long-term relative strength/trend, and 5% financial safety. **Short Term (days to weeks)** is 20% relative strength, 25% classified setup quality, 20% short-term expectations, 10% volume confirmation, 15% technical trend, and 10% event timing. Pullback and breakout/momentum engines are independent, and a security can have no credible setup.

Scores are bounded to 0–100 and available weights are renormalized only after minimum coverage is met. Missing data remains unknown (`NaN`), never zero. Each metric uses a normalized-sector peer group only with at least 15 valid observations, then falls back to index membership and the combined universe. Financial firms exclude generic FCF, net-debt/EBITDA, and ROIC measures where they are not economically applicable.

**Risk (0–100, higher is riskier)** separately describes observed price, balance-sheet, event, analyst-disagreement, and liquidity risk. **Confidence (0–100)** separately describes coverage, freshness, analyst breadth, and provider success. Neither modifies attractiveness.

## Data and freshness

Yahoo Finance supplies batched, adjusted two-year daily OHLCV plus recommendation summaries, targets, EPS trends/revisions, earnings and revenue estimates, growth estimates, rating actions, earnings history/dates, quarterly valuation measures, and annual financial statements. One `Ticker` instance is reused per symbol and concurrency defaults to four workers.

Prices refresh each run. Expectations cache for 18 hours, valuation for 24 hours, and fundamentals for seven days under `investment_ai_data/cache`. A failed refresh preserves the last successful cache. `FORCE_REFRESH=true` bypasses freshness checks.

`investment_ai_data/history.sqlite` stores normalized analyst field observations by symbol, component, and exact UTC fetch timestamp, plus every successful ranking run. Target and revenue momentum remain unavailable until genuine observations are old enough; history is never fabricated. Rank change uses positive numbers for movement upward (for example, +20 means 20 places better).

With `EXPORT_RESULTS=true`, one run directory contains `full_analysis.csv`, both ranking CSVs, `insufficient_data.csv`, metadata, and an Excel workbook with methodology and metadata sheets.

## Limitations

Yahoo fields and geographic coverage vary and may be delayed or absent. Currency-normalized benchmark strength is intentionally deferred; peer return percentiles avoid pretending mixed-currency absolute returns are a benchmark. Generic ROIC is omitted for financial companies. Historical revenue/target estimate momentum only becomes usable after locally collected point-in-time history exists. This is research decision support, not investment advice or a validated backtest; current analyst observations are never applied retroactively to historical prices.

## v3.1 request budget and integrity rules

A completely cold run makes 100–200-symbol batched two-year price downloads with retries, then, per symbol, up to nine analyst-component calls plus one small `info` call, one valuation call, and three annual-statement calls. Component TTL caches make warm runs substantially cheaper. The three previously discarded quarterly-statement calls per symbol remain removed.

Analyst components cache independently under schema version 2. Cached payloads must pass both schema and semantic validation before reuse or stale fallback. Each fresh component writes normalized point-in-time field observations under its own exact `fetched_at_utc`; cached, stale, and failed components write nothing, so a later failed refresh cannot manufacture target or revenue history.

Target and revenue revision momentum remain unknown until a local observation exists at or before each 7/30/90-day cutoff. Separate Long Expectations and Short Expectations pillars use medium/long-horizon and recency-weighted near-term signals respectively; direction uses directional inputs only. Rank and score changes use the closest stored run at least seven days old; positive rank delta means improvement.

Provider limitations remain: Yahoo can omit endpoints by exchange, financial-company regulatory capital data is not consistently available, mixed-currency returns are peer ranks rather than currency-normalized benchmark excess returns, and event dates may be tentative. Bank leverage is therefore not inferred from industrial debt ratios, and missing regulatory metrics reduce confidence rather than being invented.
