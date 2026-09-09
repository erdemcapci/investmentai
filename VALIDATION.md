# Verification — September 9, 2026

Verified in Python 3.12 with pandas 3.0.3 and yfinance 1.5.1. The existing environment supplied lxml from its base Python installation.

- **22 scanner tests passed**, including analyst conviction priority, upside and Strong Buy sensitivity, current-period selection, missing/invalid counts, rating coverage, target fallback and inconsistent ranges, zero prices, stale/future timestamps, earnings and EPS risks, extreme declines, stabilization, trading-session windows, floating-point threshold boundaries, and incomplete current-day bars.
- **First-run test passed:** a mocked constituent response is parsed and cached with the configured download URL. The HTML parser dependency is included in `requirements.txt`.
- **Legacy regression checks:** all 29 existing plain assertion test functions in `tests/test_scoring.py` passed when run as `unittest.FunctionTestCase` checks. The local environment did not have pytest installed.
- **Failure-path refresh test passed:** failed Yahoo responses retain the prior analyst cache byte-for-byte and preserve its original timestamp, while recording the new failed attempt.
- **Offline behavior test passed:** no Yahoo calls while scanning missing caches.
- **Notebook verified:** valid notebook schema; all eight code cells ran in order in a fresh Python namespace. Yahoo entry points were mocked to fail if called, confirming the default notebook path performs no requests.
- **Full cache integration passed:** 503 symbols processed. All 503 were labeled `DATA_REFRESH_REQUIRED`; current candidate and analyst watchlists were correctly empty. The generated CSV exports match the in-memory notebook ranking.
- **Single ranking:** the three original ranking paths and mismatched display/export shortlists were replaced by one module. The overwritten price-source label and the `ApiResult` definition that depended on running a later cell were eliminated.
- **EPS bug fixed:** the percentage-change denominator now uses absolute prior EPS, so worsening negative EPS is correctly treated as deterioration.

Live Yahoo downloads were not performed in this verification; provider availability and live response changes remain unverified. Refresh behavior was tested with simulated provider failures. No current stock recommendations or backtested return claims are made. Generated cache audit files are not committed.

The notebook uses standard Jupyter tables and was executed programmatically; it was not visually inspected in the Jupyter browser UI.
