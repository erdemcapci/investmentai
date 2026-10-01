# Investment AI 1.3.0

Investment AI is a deterministic, auditable research system that ranks the combined
S&P 500 and STOXX Europe 600 universe. It produces separate long-term investment and
short-term setup views, captures point-in-time inputs and predictions, and refuses to
present low-coverage runs as decision-ready.

It is **not** a trading bot, broker, return guarantee, personalized recommendation,
web application, or machine-learned weight optimizer.

## Installation

Supported Python: 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## One-command run

```bash
python main.py
```

A normal repeat run automatically refreshes constituents lightly, extends the local price
store with a small revision overlap and missing sessions, and reuses fresh analyst (~18h),
valuation (~24h), and fundamentals (~7d) caches. The user does not select components to
skip: scores and ranks are recomputed from the latest canonical inputs on every run.

The command fetches the two universes, daily prices, and normalized Yahoo provider
inputs; calculates rankings; stores prediction history; evaluates run health; and
writes critical audit artifacts unconditionally under `investment_ai_runs/<run_id>/`.
`EXPORT_RESULTS` is deprecated and reserved for future optional rich/heavy exports; it
does not disable core artifacts. Exit codes are: `0` valid/degraded/partial success, `1`
unexpected failure, `2`
configuration/universe failure, `3` invalid data quality, and `4` database schema
failure.

## Outputs

Each run contains `run_manifest.json`, `run.log`, `checkpoint.json`, `errors.csv`,
`universe.csv`, `price_features.csv`, `normalized_provider.csv`, primary LT/ST CSVs,
`investment_ranking.csv`, `action_list.csv`,
`full_analysis.csv`, `insufficient_data.csv`, `methodology.md`, and
`validation_snapshot.json`. Ranking output columns use `OUTPUT_SCHEMA_VERSION = 1`;
breaking changes are recorded in the changelog.

## Long-term methodology

Scoring model 3.2.0 combines Quality (25%), Growth (20%), peer Valuation (20%),
Long-Term Expectations (20%), Long Trend (10%), and Financial Safety (5%). Core
pillars (Quality, Growth, Valuation, Expectations) must be present; a missing
non-core pillar counts as neutral (50) instead of re-weighting the rest.

- Quality and growth inputs blend an absolute curve with the security's
  percentile inside its sector and region (US or Europe), so sector norms such as
  software margins do not dominate the ranking.
- Fundamentals prefer trailing-twelve-month figures from quarterly statements
  (`fundamentals_basis`), falling back to the last fiscal year. Growth also uses
  the latest quarter's revenue versus the same quarter a year earlier.
- Long Trend uses skip-month momentum (6-1 and 12-1 month returns, which exclude
  the most recent month) plus distance from the 200-day average.

## Short-term methodology

Model 3.3.0 weights follow the 5-year walk-forward backtest (see Backtest):
Relative Strength 65%, Short-Term Expectations 20%, Earnings Drift 10%, and
Short Interest 5%. Ranking needs Relative Strength and Short-Term Expectations.

- Relative Strength is 50% 12-1 month, 30% 6-1 month and 20% 60-day momentum
  percentiles. Over 2021–2026, a top-20 list ranked on it alone earned +0.71%
  excess return per 10 sessions after costs (t = 2.6).
- Earnings Drift scores the last EPS surprise and the two-session announcement
  reaction versus the benchmark, decaying to neutral between 45 and 90 days
  (IC t = 2.0). A larger weight diluted momentum, so it is 10%.
- Setup (pullback, breakout, continuation, mixed), signed volume and technical
  trend are still computed and exported. They have no weight and no longer
  restrict which stocks rank, because in the backtest the setup filter halved the
  momentum list's return and adding these pillars cut it to +0.06%.
- Short-Term Expectations and Short Interest cannot be replayed historically, so
  their weights rest on published evidence until live validation accumulates.
- Event timing is a risk input only.
- In a `REBOUND_RISK` market regime (benchmark below its 200-day average but up
  5%+ over 20 days) the momentum weight halves, because momentum crashes cluster
  there.

`RS_BASIS` selects the relative-strength basis: `raw` (default) ranks raw returns
within sector; `benchmark` ranks excess return over the assigned regional index.

## Risk-adjusted ranking and the action list

Ranks sort by a risk-adjusted score: `score - 0.20 x max(risk - 50, 0)`, minus 5
short-term points when earnings are due within 2 days. Raw scores remain
exported.

`action_list.csv` is the "what to act on" view. It keeps short-term ranked
names that sit in the top half of the long-term ranking (this long-term filter
cannot be backtested), orders them by
risk-adjusted short-term score, and allows at most 3 per sector (20 names by
default). Each row carries a trade plan:

- stop = two standard deviations of a 10-session move below the last close (3–25%);
- target = 1.5 times the stop distance above;
- `suggested_position_pct` risks 1% of the portfolio at the stop (capped at 10%);
- `portfolio_weight_pct` spreads the list by equal risk and sums to at most 100%;
- both sizes halve outside a `RISK_ON` regime.

These are research outputs, not orders. The parameters live in `config.py` and
can be overridden with environment variables.

## Risk and confidence

Risk is reported separately and feeds the risk-adjusted rank. Confidence is separate from attractiveness: 40% pillar coverage, 20% analyst breadth, 20% freshness, and 20% provider completeness. Stale cache fallbacks remain visibly stale, and any true
analyst component error makes a mixed bundle `PARTIAL` rather than masking it.

## Data quality and run status

`VALID` requires price >=97%, quality >=70%, valuation >=70%, and expectations >=75%.
`DEGRADED` requires at least 90%, 55%, 55%, and 60%, respectively. Anything below a
floor, or an invalid combined universe, is `INVALID`; Top 100 tables are suppressed and
the process exits 3. The manifest also contains contract counts, dependency versions,
timings, cache ratios, score distributions, and benchmark health.

## Cache and history

Cache schema 4 invalidates all older parsed objects through normal schema validation.
SQLite schema 5 uses WAL, a busy timeout, explicit metadata versioning, normalized
`analyst_observations`, idempotent ranking history, immutable `prediction_snapshots`,
and attach-only `prediction_outcomes`. The wide `analyst_snapshots` table is retained
as archival data but the product flow does not write it.

## Scheduled daily run

Live validation needs one capture per trading day. To schedule one at 03:30
local time, Tuesday to Saturday (after the US close has become a completed UTC
day), run:

```bash
scripts/install_daily_run.sh     # installs a launchd job for this user
scripts/uninstall_daily_run.sh   # removes it
```

A run missed while the Mac sleeps starts when it wakes. Logs are written to
`investment_ai_logs/`, and overlapping runs are skipped.

## Resume and replay

```bash
python main.py --resume <run_id>
python main.py --replay <run_id>
python main.py --rescore <run_id>
```

Checkpoints record universe/prices, successful and failed provider symbols, analysis,
history, and exports. Resume reuses completed inputs, skips successful symbols, retries
failures, and uses idempotent database keys. Replay reads the original CSV inputs and
does not invoke Yahoo. Exact replay requires matching scoring versions and intact input
checksums, and preserves source rank-change diagnostics rather than consulting today's
history database. Rescore intentionally applies the current model and is labelled
`RESCORE`. Neither mode writes ranking, prediction, or outcome history. Replay writes to
`<run_id>-replay-<UTC timestamp>` (and rescore uses an equivalent timestamped child); the original capture remains
unchanged.

## Benchmark-adjusted relative strength

US members use Yahoo `^GSPC` (S&P 500); European members use Yahoo `^STOXX` (STOXX
Europe 600). Dual membership is resolved deterministically by the same stable lexical
tie rule used by peers. Excess return is `stock return - assigned benchmark return`
at 20/60/126/252 sessions and for 6-1 and 12-1 month momentum, then percentiled by
sufficiently populated sector, assigned benchmark, or the full universe. With
`RS_BASIS=benchmark` the models use these percentiles and fall back to raw per
security (`rs_basis = RAW_FALLBACK`); missing benchmark data is never fabricated.
The assigned benchmark's trend also sets `market_regime`.

## Backtest

```bash
python main.py --backtest                     # local ~2-year price store
python main.py --backtest --backtest-years 5  # download a longer history
python main.py --backtest --backtest-years 5 --backtest-earnings  # add earnings drift
```

The backtest replays the price-based signals weekly (`--backtest-step`, default 5
sessions) with the production `price_features` and pillar functions, each date
seeing only bars up to that date. Entry is the next session's close. It reports,
per signal and horizon (5/10/20 sessions): mean Spearman IC against forward excess
return, an overlap-adjusted t-statistic, IC hit rate, top-minus-bottom decile
spread, top-25 excess return net of `--backtest-cost-bps` (default 10 one way),
and IC by market regime. Results go to `investment_ai_backtests/`.

With `--backtest-earnings` it also fetches each stock's report history (dates and
EPS surprises, cached for a week) and scores the Earnings Drift pillar on each
date using only reports whose two-session reaction had completed by that close.

It also simulates candidate short-term strategies (`strategy_summary.csv`,
`trades.csv`): top 20 names, at most 3 per sector, entered at the next close and
either held 10 sessions or exited by the production trade plan (first close at
or below the stop or at or above the target). Exits use closes only, so gaps
through a stop are taken in full. Strategies compare the production price model
with momentum alone, drift alone, and the 3.2.0 setup-gated composite.

Results of the 5-year run (2021-11 to 2026-08, 246 weekly dates, 10 bps each
way), per 10-session trade:

| Strategy | Excess net | t-stat |
|---|---|---|
| Momentum, no setup gate (basis of 3.3.0) | +0.71% | 2.6 |
| Momentum, credible setups only | +0.32% | 1.4 |
| 3.2.0 composite, credible setups only | +0.12% | 0.7 |
| Earnings drift only | +0.23% | 1.3 |

A one-sigma stop with a 1.5-sigma target lowered the momentum list to +0.57%;
a two-sigma stop cost little (+0.66%). These figures are in-sample (the
momentum weights were chosen on the same period) and subject to survivorship
bias, so expect live results to be weaker.

Analyst, valuation and fundamental pillars have no point-in-time history and are
not replayed; the live validation below covers them. The universe and sector
labels are today's (survivorship bias), which inflates return levels more than
cross-sectional IC.

## Validation framework

Validation is cross-sectional and run-based: Top 10/25/50 equal-weight returns and
Spearman IC are calculated independently inside each eligible historical run, then
mean/median/hit-rate statistics are aggregated over runs. The primary sample size is
the evaluation-run count; stock observations and censoring counts remain secondary,
visible diagnostics. `<20` runs is `INSUFFICIENT_SAMPLE`, 20–59 is `EARLY_SAMPLE`, and
60+ is `USABLE`. `ALL_RUNS` is explicitly overlapping; `NON_OVERLAPPING` samples at
the horizon spacing. Missing and delisted outcomes remain in coverage denominators.
There is no automatic model-weight tuning.

Validation uses price returns for both stocks and frozen regional benchmarks. European
excess return is `LOCAL_CURRENCY_EXCESS_RETURN_EXPERIMENTAL`, not FX-neutral, and is
not used in scoring. Analyst 7/30/90-day observations use exact UTC cutoff timestamps.

## Known limitations

- Yahoo is an unofficial external dependency and its availability/schema can change.
- Pillar weights are hand-set. The backtest measures the price-based signals, but
  weights are not fitted automatically.
- Short interest is mostly unavailable for European listings and counts as neutral
  there.
- Resumed runs without retained raw price history defer outcome attachment to the next
  fresh normal run; the report cannot manufacture outcomes that have not matured.
- CSV is used for portable, dependency-light intermediate storage rather than Parquet.
- Scores are research support, not investment advice; taxes, spreads, FX, and portfolio
  suitability are outside scope.

## Reproducible dependencies

`requirements.lock` records the complete resolved environment. Install it with
`python -m pip install -r requirements.lock`. To update, change top-level pins and run
`python -m pip freeze --all > requirements.lock` in a clean Python 3.12 environment,
then run the full test suite and dependency audit.

## Universe integrity

Constituent health floors are 480 S&P 500 and 560 STOXX Europe 600 rows. Mapping below
95% is degraded and below 90% invalid. Source caches are fresh through 72 hours, stale
through 14 days, and too stale thereafter. Persisted verified mappings win; known
share-class exceptions and provider metadata can verify candidates; suffix-only guesses
remain `HEURISTIC` and are emitted in `unmapped_constituents.csv`.
