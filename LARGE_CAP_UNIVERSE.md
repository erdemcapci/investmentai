# Investment AI — Large-Cap Analysis Universe

`large_cap_main.py` is the expanded scanner entry point. It keeps the existing analyst-first long-term and short-term models, but removes S&P 500 membership as the eligibility gate.

## Pipeline

1. Discover NYSE/Nasdaq equities with market capitalization of at least **$10B**.
2. Exclude non-equities and obvious SPAC/shell-company candidates.
3. Apply a **$20M minimum dollar-liquidity** screen.
4. Use a neutral market-cap/liquidity prefilter to limit expensive Yahoo analyst calls to the default **500** most liquid/large candidates. This stage does **not** use Buy/Sell direction, price-target upside, momentum, or pullbacks.
5. Download current recommendation distributions, analyst targets, EPS revisions, earnings dates, and recent analyst-rating activity.
6. Require at least **8 current analyst opinions**, a usable analyst target, current price data, and sufficient data quality.
7. Rank eligible companies by **analysability**, not bullishness:
   - 35% analyst coverage
   - 25% market cap
   - 20% liquidity
   - 20% data completeness/freshness
8. Select the dynamic **Top 100 Analysis Universe**.
9. Download full two-year price history for those 100 stocks only.
10. Run the existing long-term analyst model, short-term entry model, combined ranking, risk flags, and candidate profiles.

S&P 500 membership is retained as `is_sp500` metadata, so results can still be split into S&P 500 and non-S&P 500 companies without restricting discovery.

## Analyst coverage confidence

The large-cap runner uses the following confidence bands for the long-term analyst score:

| Current analyst opinions | Band | Confidence |
| ---: | --- | ---: |
| 0–4 | Insufficient | 0.50 |
| 5–7 | Low | 0.70 |
| 8–14 | Acceptable | 0.85 |
| 15–24 | Strong | 0.95 |
| 25+ | Very strong | 1.00 |

Because the large-cap analysis universe requires at least eight analysts, selected stocks normally start at 0.85 coverage confidence. The existing softened coverage multiplier remains in place downstream, so analyst count affects confidence without becoming the investment thesis by itself.

## Why the Top 100 selection is neutral

The universe selector intentionally does **not** use:

- Buy/Strong Buy percentage
- Sell/Strong Sell percentage
- target-price upside
- positive or negative EPS revision direction
- recent price return
- pullback magnitude
- technical momentum

Those variables are investment signals and belong to the ranking model after universe selection. Using them to choose the Top 100 would bias the candidate pool before the model even runs.

Analyst-rating activity is used only as a freshness/availability signal. An upgrade and a downgrade both count as analyst activity for universe quality; their direction does not improve the universe score.

## Run

```sh
python -m pip install -r requirements.txt
python large_cap_main.py
```

To export CSV and Excel outputs:

```sh
EXPORT_RESULTS=true python large_cap_main.py
```

Configuration can be changed in `.env`:

```text
ANALYSIS_UNIVERSE_SIZE=100
UNIVERSE_PREFETCH_LIMIT=500
UNIVERSE_MIN_MARKET_CAP=10000000000
UNIVERSE_MIN_DOLLAR_VOLUME=20000000
UNIVERSE_MIN_ANALYST_COUNT=8
UNIVERSE_MIN_DATA_QUALITY=60
```

The default prefetch limit of 500 keeps Yahoo request volume broadly comparable with the old S&P 500 scan while still allowing non-S&P large caps to compete for the final Top 100. Increase it if broader discovery is more important than runtime/API pressure.

## Outputs

With `EXPORT_RESULTS=true`, a run creates `large_cap_fresh_runs/<run-id>/` with:

- `universe_diagnostics.csv` — all prefiltered companies and eligibility failures
- `eligible_large_cap_universe.csv` — companies passing analyst/data gates
- `analysis_universe_top100.csv` — final neutral Top 100
- `fresh_analyst_data.csv`
- `price_metrics.csv`
- `daily_price_history.csv.gz`
- `full_dual_score_analysis.csv`
- `long_term_ranking.csv`
- `short_term_entry_ranking.csv`
- `combined_candidates.csv`
- `large_cap_dual_score_analysis.xlsx`

This remains a research screener, not an automatic trading system or a guarantee of future returns.
