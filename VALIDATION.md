# Validation policy

Version 3.1 saves component-specific point-in-time observations and ranks without look-ahead. Offline tests cover cache schema/semantic validation, independent analyst timestamps, recency-weighted Long/Short Expectations, financial edge cases, metric-valid peer groups, price freshness and completed bars, risk coverage, provider contracts, and the canonical mocked `main()` flow. Live Yahoo checks remain optional because availability and schemas are external.

Scoring model 3.2.0: the Long-Term score is 25% Quality, 20% Growth, 20% Valuation, 20% Long Expectations, 10% Long Trend, and 5% Financial Safety. The Short-Term score is 20% Relative Strength, 20% Setup Quality, 20% Short Expectations, 20% Earnings Drift, 10% Signed Volume, 5% Technical Trend and 5% Short Interest. Ranks use risk-adjusted scores.

`python main.py --backtest` replays the price-based signals point-in-time and reports information coefficients and net top-N returns; `tests/test_v32.py` checks that altering future prices never changes a past signal. Live prediction snapshots remain the only evidence for the analyst, valuation and fundamental pillars.

Release checks are `pytest -q`, `python -m compileall -q investment_ai main.py`, and an import/static check. No profitability claim is made.
