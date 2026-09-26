# Investment AI 1.1

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
`full_analysis.csv`, `insufficient_data.csv`, `methodology.md`, and
`validation_snapshot.json`. Ranking output columns use `OUTPUT_SCHEMA_VERSION = 1`;
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

Risk is reported separately rather than hidden in alpha scores. Confidence is separate from attractiveness: 40% pillar coverage, 20% analyst breadth, 20% freshness, and 20% provider completeness. Stale cache fallbacks remain visibly stale, and any true
analyst component error makes a mixed bundle `PARTIAL` rather than masking it.

## Data quality and run status

`VALID` requires price >=97%, quality >=70%, valuation >=70%, and expectations >=75%.
`DEGRADED` requires at least 90%, 55%, 55%, and 60%, respectively. Anything below a
floor, or an invalid combined universe, is `INVALID`; Top 100 tables are suppressed and
the process exits 3. The manifest also contains contract counts, dependency versions,
timings, cache ratios, score distributions, and benchmark health.

## Cache and history

Cache schema 3 invalidates all older parsed objects through normal schema validation.
SQLite schema 5 uses WAL, a busy timeout, explicit metadata versioning, normalized
`analyst_observations`, idempotent ranking history, immutable `prediction_snapshots`,
and attach-only `prediction_outcomes`. The wide `analyst_snapshots` table is retained
as archival data but the product flow does not write it.

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
tie rule used by peers. Excess return is `stock return - assigned benchmark return` at
20/60/126/252 sessions, then percentiled by sufficiently populated sector, assigned
benchmark, or the full universe. Raw and excess features are both retained. Benchmark
RS is **experimental and not used in v3.1.1 scoring**; missing benchmark data is marked
`FALLBACK_RAW`, never fabricated. The direct tickers could not be live-verified in the
restricted build environment, so operational benchmark health is explicit.

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
- Benchmark RS remains experimental pending accumulated side-by-side evidence.
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
