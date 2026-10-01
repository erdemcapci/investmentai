# Changelog

## 1.2.0 (scoring model 3.2.0)

- Short term: removed the double-counted one-day return from the pullback score;
  relative strength is now 50% 12-1, 30% 6-1 and 20% 60-day momentum (the
  20-day term had no edge in the 5-year backtest); pullbacks require an uptrend and setup thresholds rose from 55 to 65.
- Added an Earnings Drift pillar (last surprise, two-session announcement reaction
  versus the benchmark, 45–90 day decay) and a Short Interest pillar. Volume is now
  signed by price direction. Event timing moved from alpha to risk only.
- Long term: skip-month (6-1, 12-1) momentum in Long Trend; quality and growth
  inputs blend in sector-and-region percentiles; trailing-twelve-month
  fundamentals from quarterly statements with a latest-quarter growth input.
- Missing non-core pillars count as neutral instead of re-weighting the rest.
- `RS_BASIS` switch (raw default, or benchmark-excess relative strength); benchmark
  relative strength now also covers the momentum measures, and runs faster.
- Market regime per region (`RISK_ON`, `RISK_OFF`, `REBOUND_RISK`).
- Ranks follow a risk-adjusted score; new trade plan columns, and a
  sector-capped `action_list.csv` that applies the long-term filter to short-term
  timing.
- New `python main.py --backtest` walk-forward harness for the price signals.
- The fundamentals cache refreshes once to capture quarterly statements.
- Tests no longer write run folders, caches or history into the working tree.

## 1.1.5

- Added `investment_ranking.csv` with Long-Term and Short-Term rankings side-by-side.
- Exposed existing rank/score and analyst historical-change fields for easier filtering and comparison.
- Preserved separate LT/ST ranking outputs and independent scoring semantics.
- No scoring, validation, risk, confidence, cache, or history-semantic changes.

## 1.1.4

- Added exchange-aware normalization for iShares STOXX local ticker formats.
- Fixed London trailing-separator ticker handling.
- Added deterministic Nordic share-class normalization.
- Preserved Yahoo identity verification and invalid-symbol fail-fast behavior.
- No scoring, ranking, validation, risk, or confidence changes.

## 1.1.3

- Renamed the generated benchmark assignment metadata field from
  `benchmark_method` to the persisted `benchmark_assignment_method` contract.
- Explicitly record `benchmark_return_basis` as `ADJUSTED_CLOSE_RETURN` for every
  benchmark assignment.

## 1.1.2

- Froze the 3.1.2 scoring implementation with unchanged published weights and
  corrected canonical-sector and currency-safe FCF inputs.
- Invalidated pre-1.1.2 provider caches with cache schema 4, unified Yahoo
  listing metadata, isolated validation by model version, retried omitted
  incremental price symbols, and made outcome maturity session-aware.

## 1.1.1

- Completed the final v1.1 integration of authoritative STOXX resolution,
  universe health, provider currencies, unbiased validation cohorts, and the
  incremental local price store. Scoring model weights remain unchanged.

## v1.1.0 — data integrity and validation correctness

- Added durable security mappings, stable identities, explicit heuristic/unresolved states,
  constituent telemetry policy, local daily-price storage, anomaly flags, and outcome states.
- Made historical outcome fetches independent of current membership and added query indexes.
- Rebuilt validation around per-run cross-sectional Top-N portfolios and Spearman IC, with
  run-level aggregation, overlap modes, and explicit censoring coverage.
- Added exact timestamp history, independent non-null rank history, currency-safe FCF yield,
  canonical sector precedence, growth fallback, separate freshness/completeness confidence,
  PARTIAL status, scoring parameter snapshots, and replay output verification.
- Application 1.1.0, scoring model 3.1.1 unchanged, database schema 5, output schema 1.

## v1.0.2 — runtime hardening

- Upgraded the GitHub-maintained checkout and Python setup actions to their Node 24
  majors. The v1.0.1 baseline was green (126 tests plus compile, Ruff, and whitespace);
  the visible messages were action-runtime deprecation warnings, not application failures.
- Made every JSON artifact strict and scientific-data safe by recursively mapping missing
  and non-finite values to JSON `null` while retaining `allow_nan=False`.
- Added health-gated prediction persistence and database schema 4 horizon-status metadata;
  validation excludes INVALID horizons while retaining VALID and DEGRADED observations.
- Added deterministic scoring-code fingerprints, mandatory frozen-artifact checksums for
  exact replay, and collision-resistant replay/rescore child run IDs.
- Clarified provider capture versus usability, expanded component error telemetry, removed
  the misleading parse-empty metric, guaranteed SQLite cleanup, and added a compact run
  summary. Scoring model 3.1.1 and all published scoring weights remain unchanged.

## v1.0.1 — stabilization

- Repaired GitHub CI packaging and made Python 3.12 the sole supported runtime.
- Made resume point-in-time correct with durable run-start and analysis-as-of timestamps,
  incremental provider checkpoints, artifact reconciliation, and retained errors.
- Split compatible exact replay from explicit rescore, with version/integrity checks and
  frozen rank-change diagnostics.
- Corrected validation baselines, automated matured-outcome updates, added excess-return
  reporting, and migrated prediction storage non-destructively to database schema 3.
- Added common/LT/ST health gates, invalid-view export suppression, and clearer degraded
  outputs without changing scoring model 3.1.1.
- Expanded operational errors, timing, artifact checksums, and regional benchmark health
  telemetry; removed the placeholder provider schema-rejection metric.

## v1.0.0 — application release

- **Scoring model:** 3.1.1 (unchanged). Benchmark-adjusted RS is captured side-by-side
  but remains experimental, so production score semantics did not change.
- **Integrity:** cache schema 3, database schema 2, atomic JSON/CSV writes, normalized
  analyst history, immutable prediction snapshots, structured errors, run health gates,
  manifests, logs, checkpoints, resume, and offline replay.
- **Validation:** trading-session forward outcomes, benchmark excess outcomes, sample
  guards, Top-N/bucket/correlation reports, and stored pillar diagnostics. No automatic
  tuning was introduced.
- **Output changes:** every run now has a stable artifact directory and machine-readable
  manifest. Invalid runs suppress decision-ready tables and exit with code 3.
- **Engineering:** pinned production/development dependencies and Python 3.11/3.12 CI
  for pytest, compileall, Ruff, and whitespace checks.
