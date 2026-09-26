# Changelog

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
