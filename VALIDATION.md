# Validation policy

Version 3.0.1 saves point-in-time inputs and ranks without look-ahead. Offline provider-contract fixtures cover yfinance-style period-column recommendations, CamelCase annual statements, valuation labels, rating actions, cache failure, history age gates, price cleanup, financial applicability, and the canonical mocked `main()` flow. Live Yahoo checks are optional smoke tests because availability and schemas are external.

Release checks are `pytest -q`, `python -m compileall -q investment_ai main.py`, and an import/static check. No profitability claim is made.
