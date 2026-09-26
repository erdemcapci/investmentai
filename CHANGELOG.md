# Changelog

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
