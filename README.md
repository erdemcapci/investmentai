# Investment AI — analyst conviction and pullbacks

This scanner implements your preference for strong analyst agreement, substantial target-price upside, and a recent price decline that may offer a better entry. The notebook uses one ranking engine for every table and export. Run `investment_scanner.py` for this workflow; the existing `main.py` and `scoring.py` entry point uses its own scoring model. It is an explainable research screen, not a fitted AI prediction model.

## Use it

Install the repository dependencies with `python -m pip install -r requirements.txt`. For the notebook, install `requirements-notebook.txt` too. Open `Analyst_Pullback_Scanner.ipynb` with the repository root as its working directory and run all cells. By default this reads the `sp500_notebook_data` cache without contacting Yahoo. For a new checkout, first run `python investment_scanner.py --refresh` to populate that cache, or pass `--data-dir` to an existing notebook-format cache. The notebook shows the data status, analyst watchlist, pullback candidates, and companies needing review or more time. Change `Settings(...)` in the setup cell to experiment locally.

Set `REFRESH_DATA = True` to download stale or missing data, then run the notebook. Full-universe refreshes can take many minutes and depend on Yahoo availability. Set it back to `False` afterward. To update just a few symbols, set `REFRESH_SYMBOLS = ["AAPL", "MSFT"]`; ranking still covers the cached universe and flags other stale rows. Failures appear in `cache/refresh_errors.csv`. Failed core updates preserve the previous bundle with its original timestamp.

Command-line equivalents, from the original project folder:

```sh
python investment_scanner.py
python investment_scanner.py --refresh
python investment_scanner.py --refresh --symbols AAPL MSFT
python -m unittest discover -s tests -v
```

For a cache stored elsewhere:

```sh
python investment_scanner.py --data-dir /path/to/sp500_notebook_data --output-dir analysis-output
```

Python 3.12+ is recommended. The scanner loads locally created pickle caches; use only caches you trust. Price caches use `cache/price_history_2y/<symbol>.pkl`, analyst response bundles use `cache/fundamentals/<symbol>.pkl`, and the universe uses `cache/sp500_constituents.csv`. This format is separate from the original script's `sp500_fresh_runs` outputs. The notebook's `DATA_DIR` is configurable.

## Selection and ranking

### 1. Analyst thesis comes first

A stock must have all five counts from Yahoo's **current** recommendation distribution, at least **10 rating opinions**, **70% Buy + Strong Buy**, and at most **10% Sell + Strong Sell**. Earnings-estimate analyst counts cannot substitute for rating coverage. Missing counts are unknown rather than zero. Yahoo opinions are treated as provider-reported counts, not verified independent votes.

Then require **15% target upside** and **$20 million average daily turnover**. Target = positive median, falling back to a positive mean. Upside = `(target / latest dated close - 1) × 100`. The low target is also shown; it is a downside scenario, not a guaranteed floor. Inconsistent target ranges are rejected.

The analyst watchlist contains every stock passing these thesis gates and freshness checks, including ones waiting for timing or risk review. It does not assert that every watchlist stock is ready to buy.

Qualifying stocks sort by conviction bands: **90–100% positive**, **80–<90%**, then **70–<80%** at default settings. Within a band the score is:

- **60% analyst conviction:** `0.8 × positive-rating percentage + 0.2 × Strong-Buy percentage`.
- **25% upside:** upside scaled from 0 to 50%, capped at 50% so extreme targets cannot dominate indefinitely.
- **15% pullback:** a bounded preference for moderate volatility-relative declines. It peaks at 1.5 times the recent daily standard deviation scaled by the square root of the pullback window, and goes to zero at 0 or 3 times that amount.

The score is a preference index from 0 to 100, not a probability or expected percentage return. Sorting bands enforce analyst priority; within a band, upside and entry timing can outweigh small differences in analyst percentages. Nonqualifying rows remain visible after qualifying rows.

### 2. Look for a recent, moderate decline

The pullback screen accepts a **3–10% drop over five trading sessions**, or a **5–15% drop over ten sessions with the last five still negative**. It displays both returns and drawdown from the highest close over 20 sessions. Six closing observations are required for a five-session return.

It also requires a positive latest daily return and a close at or above the five-session average. This is only an initial stabilization condition. It cannot establish that a bottom has formed.

A decline beyond the percentage limits or above three volatility units is routed to review. Bigger drops do not automatically earn higher scores.

### 3. Review risks before calling it a pullback candidate

The separate pullback shortlist requires the analyst thesis, qualifying decline, stabilization, and no blocking review conditions:

| Review condition | Default threshold |
| --- | --- |
| Target disagreement | `(high - low) / selected target` above 60%, or unknown range |
| Damaged trend | More than 10% below the full 200-session average, or more than 20% down over 21 sessions |
| Earnings deterioration | Current-quarter EPS estimate cut by more than 5% in 30 days |
| Event risk | Earnings within seven calendar days, or no known future earnings date |
| Missing timing context | Unknown volatility, full 200-session trend, or EPS revision data |

EPS changes use the absolute prior EPS in the denominator: moving from a loss of 1 to a loss of 1.5 is deterioration of 50%. A zero baseline is unknown. These risk conditions can be adjusted in `Settings` where thresholds exist.

The labels are `PULLBACK_CANDIDATE`, `WAIT_FOR_PULLBACK`, `WAIT_FOR_STABILIZATION`, `REVIEW_RISK`, `DOES_NOT_QUALIFY`, and `DATA_REFRESH_REQUIRED`. Each row states why it received that label. A pullback candidate is a research lead requiring review of the reason for the decline, not an automatic order.

## Freshness and interpretation

Price observations older than **four calendar days**, analyst bundles retrieved more than **seven days** ago, missing timestamps, and future timestamps fail the current shortlist. Four days allows ordinary weekends, but is not an exchange-calendar guarantee; inspect the displayed date, especially after market holidays. Analysis uses a single dated daily close for both target upside and price behavior. Today's Yahoo daily bar is excluded before 16:00 New York time. Early-close exchange holidays are not modeled; this policy may conservatively defer that day's bar until 16:00.

Retrieving an analyst bundle today does **not** mean every underlying analyst report was updated today. Actual report publication dates and a reliable target deadline are not provided by these aggregate endpoints, so the scanner shows that limitation. Median targets and dispersion reduce some outlier influence; neither verifies the analysts' valuation assumptions.

A 40% target gap is more potential upside than 20% under those target assumptions. It does **not** imply a faster climb, a reliable recovery, or a 40% short-term return. This is why target upside and entry conditions are separate. Analyst reports often address much longer periods than a one-week dip; the scanner does not invent a deadline or annualize the gap.

The screen does not inspect company news, model transaction costs, determine position sizes, or place trades. Sector-wide valuation levels are not used as universal pass/fail rules. Fresh EPS deterioration, target disagreement, liquidity and trend are the limited extra checks chosen for this strategy.

## Validation and next evidence to collect

An offline integration run processed 503 locally cached symbols on September 9, 2026. Every row was correctly marked `DATA_REFRESH_REQUIRED`. Cached market data and generated reports are not included in this repository. See `VALIDATION.md` for implementation verification.

These default thresholds have not been optimized or backtested. The cache contains only one analyst snapshot per stock; using that snapshot to rank past dates would introduce look-ahead bias. Historical results using today's S&P membership would also have survivorship bias. The refresh function now preserves dated analyst response snapshots and previous bundles to support future point-in-time validation.

Before interpreting the screen as a profitable strategy, collect dated signals, measure subsequent 5/10/20-session returns and adverse excursions, compare against an appropriate market/sector benchmark, include trading costs and delisted constituents, and validate on unseen periods. This implementation makes no measured claim about recovery speed or strategy profitability.

## Sources behind the design

- [SEC: Analyzing Analyst Recommendations](https://www.sec.gov/about/reports-publications/investorpubsanalystshtm): analyst rating definitions, assumptions and conflicts require scrutiny; a recommendation alone is insufficient.
- [Fidelity: S&P research methodology](https://research2.fidelity.com/fidelity/research/reports/SandP.asp): an example of analyst assessments using a 12-month performance horizon. This does not establish the horizon of every Yahoo target.
- [Federal Reserve Bank of New York: Decomposing Short-Term Return Reversal](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr513.pdf): research distinguishes components of short-term reversal and fundamental news. It supports examining the reason for a decline, not a blanket assumption that declines rebound.

The numerical thresholds above are design choices for your preferences, not recommendations or results supplied by those sources.
