# Investment AI 1.0

Investment AI is a deterministic, auditable research system that ranks the combined
S&P 500 and STOXX Europe 600 universe. It produces separate long-term investment and
short-term setup views, captures point-in-time inputs and predictions, and refuses to
present low-coverage runs as decision-ready.

It is **not** a trading bot, broker, return guarantee, personalized recommendation,
web application, or machine-learned weight optimizer.

## Installation

Python 3.11 and 3.12 are supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## One-command run

```bash
python main.py
```

The command fetches the two universes, daily prices, and normalized Yahoo provider
inputs; calculates rankings; stores prediction history; evaluates run health; and
writes a durable folder under `investment_ai_runs/<run_id>/`. `EXPORT_RESULTS` remains
a configuration field for compatibility, but critical audit artifacts are always
written. Exit codes are: `0` valid/degraded success, `1` unexpected failure, `2`
configuration/universe failure, `3` invalid data quality, and `4` database schema
failure.

## Outputs

Each run contains `run_manifest.json`, `run.log`, `checkpoint.json`, `errors.csv`,
`universe.csv`, `price_features.csv`, `normalized_provider.csv`, primary LT/ST CSVs,
`full_analysis.csv`, `insufficient_data.csv`, `methodology.md`, and
`validation_snapshot.json`. Ranking output columns are treated as the stable v1 schema;
breaking changes are recorded in the changelog.

## Long-term methodology

The v3.1.1 score combines Quality (25%), Growth (20%), peer Valuation (20%), Long-Term
Expectations (20%), Long Trend (10%), and Financial Safety (5%). Core pillars must be
present. Raw relative strength remains the production trend signal in v1.0.

## Short-term methodology

The score combines raw Relative Strength (20%), Setup Quality (25%), Short-Term
Expectations (20%), Volume (10%), Technical Trend (15%), and Event Timing (10%). Only
credible pullback, breakout, continuation, or mixed setups can be ranked; `NONE` is
reported as `NO_CREDIBLE_SETUP`.

## Risk and confidence

Risk is reported separately rather than hidden in alpha scores. Confidence reflects
coverage and freshness. Stale cache fallbacks remain visibly stale, and any true
analyst component error makes a mixed bundle `PARTIAL` rather than masking it.

## Data quality and run status

`VALID` requires price >=97%, quality >=70%, valuation >=70%, and expectations >=75%.
`DEGRADED` requires at least 90%, 55%, 55%, and 60%, respectively. Anything below a
floor, or an invalid combined universe, is `INVALID`; Top 100 tables are suppressed and
the process exits 3. The manifest also contains contract counts, dependency versions,
timings, cache ratios, score distributions, and benchmark health.

## Cache and history

Cache schema 3 invalidates all older parsed objects through normal schema validation.
SQLite schema 2 uses WAL, a busy timeout, explicit metadata versioning, normalized
`analyst_observations`, idempotent ranking history, immutable `prediction_snapshots`,
and attach-only `prediction_outcomes`. The wide `analyst_snapshots` table is retained
as archival data but the product flow does not write it.

## Resume and replay

```bash
python main.py --resume <run_id>
python main.py --replay <run_id>
```

Checkpoints record universe/prices, successful and failed provider symbols, analysis,
history, and exports. Resume reuses completed inputs, skips successful symbols, retries
failures, and uses idempotent database keys. Replay reads the original CSV inputs and
does not invoke Yahoo. Replay writes to `<run_id>-replay`; the original capture remains
unchanged.

## Benchmark-adjusted relative strength

US members use Yahoo `^GSPC` (S&P 500); European members use Yahoo `^STOXX` (STOXX
Europe 600). Dual membership is resolved deterministically by the same stable lexical
tie rule used by peers. Excess return is `stock return - assigned benchmark return` at
20/60/126/252 sessions, then percentiled by sufficiently populated sector, assigned
benchmark, or the full universe. Raw and excess features are both retained. Benchmark
RS is **experimental and not used in v3.1.1 scoring**; missing benchmark data is marked
`FALLBACK_RAW`, never fabricated. The direct tickers could not be live-verified in the
restricted build environment, so operational benchmark health is explicit.

## Validation framework

Every ranked observation saves its scores, ranks, risk/confidence, setup, prediction
prices, benchmark, region, sector, and pillar scores exactly as known at that run.
Outcome attachment adds realized 5/10/20/63/126/252-session stock, benchmark, and
excess returns without recomputing historical scores. Generate summaries with:

```bash
python main.py --validation-report
```

Reports include N, sample status, mean/median/standard deviation, win rate, Spearman
rank correlation, Top 10/25/50 medians, and score buckets. N below 30 is
`INSUFFICIENT_SAMPLE`, 30–99 is `EARLY_SAMPLE`, and 100+ is `USABLE`. There is no
automatic model-weight tuning.

## Known limitations

- Yahoo is an unofficial external dependency and its availability/schema can change.
- Benchmark RS remains experimental pending accumulated side-by-side evidence.
- Outcome updating requires future daily price histories supplied to the updater; the
  report cannot manufacture outcomes that have not matured.
- CSV is used for portable, dependency-light intermediate storage rather than Parquet.
- Scores are research support, not investment advice; taxes, spreads, FX, and portfolio
  suitability are outside scope.
