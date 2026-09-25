# Validation policy

Version 3.1 saves component-specific point-in-time observations and ranks without look-ahead. Offline tests cover cache schema/semantic validation, independent analyst timestamps, recency-weighted Long/Short Expectations, financial edge cases, metric-valid peer groups, price freshness and completed bars, risk coverage, provider contracts, and the canonical mocked `main()` flow. Live Yahoo checks remain optional because availability and schemas are external.

The Long-Term score remains 25% Quality, 20% Growth, 20% Valuation, 20% Long Expectations, 10% Long Trend, and 5% Financial Safety. The Short-Term score remains 20% Relative Strength, 25% Setup Quality, 20% Short Expectations, 10% Volume, 15% Technical Trend, and 10% Event Timing.

Release checks are `pytest -q`, `python -m compileall -q investment_ai main.py`, and an import/static check. No profitability claim is made.
