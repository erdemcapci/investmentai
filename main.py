from __future__ import annotations

import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm.auto import tqdm

from scoring import (
    INSUFFICIENT_DATA,
    MIN_ANALYST_LONG_TERM_COVERAGE,
    MIN_SHORT_TERM_COVERAGE,
    PARTIAL_DATA,
    RANKED,
    SCORING_MODEL_VERSION,
    assign_candidate_profile,
    assign_long_term_category,
    assign_short_term_category,
    calculate_analyst_backed_pullback_bonus,
    calculate_coverage_confidence,
    calculate_coverage_multiplier,
    candidate_profile_priority,
    format_driver_text,
    score_absolute_volatility,
    score_analyst_sentiment,
    score_earnings_timing,
    score_eps_revisions,
    score_long_term_trend,
    score_ma_alignment,
    score_momentum,
    score_pullback_quality,
    score_sector_volatility_percentile,
    score_selloff_stability,
    score_target_agreement,
    score_target_upside,
    safe_float,
    weighted_score_available,
)


def load_dotenv_file(
    path: Path = Path(".env"),
) -> None:

    if not path.exists():
        return

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or "=" not in stripped
        ):
            continue

        key, value = stripped.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )


load_dotenv_file()


# ============================================================
# SETTINGS
# ============================================================

# Ayrıntısını terminalde görmek istediğin ticker
TICKER_TO_CHECK = "VRTX"

# Terminalde gösterilecek maksimum satır
TOP_N = 100

# Dosya çıktıları varsayılan olarak kapalı.
# CSV/Excel üretmek için EXPORT_RESULTS=true kullan.
EXPORT_RESULTS = (
    os.getenv(
        "EXPORT_RESULTS",
        "false",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "y",
    }
)

# Yahoo analyst request worker sayısı
# Rate-limit hatası olursa 2 yap.
MAX_WORKERS = 4

PRICE_PERIOD = "2y"
PRICE_INTERVAL = "1d"

REQUEST_RETRIES = 3
RETRY_SLEEP_SECONDS = 2

LOW_ANALYST_COVERAGE_THRESHOLD = 10
HIGH_TARGET_DISPERSION_THRESHOLD_PCT = 50
EARNINGS_WARNING_DAYS = 7
HIGH_VOLATILITY_THRESHOLD_PCT = 60
LOW_AVERAGE_DOLLAR_VOLUME = float(
    os.getenv(
        "LOW_AVERAGE_DOLLAR_VOLUME",
        "10000000",
    )
)
VERY_LOW_AVERAGE_DOLLAR_VOLUME = float(
    os.getenv(
        "VERY_LOW_AVERAGE_DOLLAR_VOLUME",
        "5000000",
    )
)

RUN_STARTED_AT_UTC = pd.Timestamp.now(tz="UTC")
RUN_ID = RUN_STARTED_AT_UTC.strftime("%Y%m%d_%H%M%S_UTC")

OUTPUT_ROOT = Path("sp500_fresh_runs")
RUN_DIR = OUTPUT_ROOT / RUN_ID

RUN_DIR.mkdir(
    parents=True,
    exist_ok=False,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_number(value: Any) -> float:
    try:
        if value is None:
            return np.nan

        number = float(value)

        if math.isfinite(number):
            return number

        return np.nan

    except (TypeError, ValueError):
        return np.nan


def normalize_name(value: Any) -> str:
    text = str(value).strip()

    text = re.sub(
        r"(?<!^)(?=[A-Z])",
        "_",
        text,
    )

    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def request_with_retry(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, str]:

    last_error: Exception | None = None

    for attempt in range(
        1,
        REQUEST_RETRIES + 1,
    ):
        try:
            result = function(
                *args,
                **kwargs,
            )

            return result, ""

        except Exception as exc:
            last_error = exc

            if attempt < REQUEST_RETRIES:
                time.sleep(
                    RETRY_SLEEP_SECONDS
                    * attempt
                )

    return (
        None,
        f"{type(last_error).__name__}: {last_error}",
    )


def linear_score(
    values: pd.Series,
    minimum: float,
    maximum: float,
) -> pd.Series:

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    score = (
        (numeric - minimum)
        / (maximum - minimum)
        * 100
    )

    return score.clip(
        lower=0,
        upper=100,
    )


# ============================================================
# S&P 500 CONSTITUENTS
# ============================================================

def download_sp500_constituents() -> pd.DataFrame:

    url = (
        "https://raw.githubusercontent.com/datasets/"
        "s-and-p-500-companies/main/data/constituents.csv"
    )

    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=30,
    )

    response.raise_for_status()

    raw = pd.read_csv(
        StringIO(response.text)
    )

    constituents = raw.rename(
        columns={
            "Symbol": "source_symbol",
            "Security": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "Headquarters Location": "headquarters",
            "Date added": "date_added",
            "Founded": "founded",
        }
    ).copy()

    # Yahoo BRK.B yerine BRK-B kullanır
    constituents["symbol"] = (
        constituents["source_symbol"]
        .astype(str)
        .str.strip()
        .str.replace(
            ".",
            "-",
            regex=False,
        )
    )

    desired_columns = [
        "symbol",
        "source_symbol",
        "company_name",
        "sector",
        "sub_industry",
        "headquarters",
        "date_added",
        "CIK",
        "founded",
    ]

    constituents = constituents[
        [
            column
            for column in desired_columns
            if column in constituents.columns
        ]
    ]

    constituents = (
        constituents
        .drop_duplicates(
            subset="symbol"
        )
        .reset_index(drop=True)
    )

    return constituents


def download_sp500_constituents_from_wikipedia() -> pd.DataFrame:

    url = (
        "https://en.wikipedia.org/wiki/"
        "List_of_S%26P_500_companies"
    )

    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        timeout=60,
    )

    response.raise_for_status()

    tables = pd.read_html(
        StringIO(response.text)
    )

    raw = next(
        table
        for table in tables
        if {
            "Symbol",
            "Security",
        }.issubset(table.columns)
    )

    constituents = raw.rename(
        columns={
            "Symbol": "source_symbol",
            "Security": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "Headquarters Location": "headquarters",
            "Date added": "date_added",
            "Founded": "founded",
        }
    ).copy()

    # Yahoo BRK.B yerine BRK-B kullanır
    constituents["symbol"] = (
        constituents["source_symbol"]
        .astype(str)
        .str.strip()
        .str.replace(
            ".",
            "-",
            regex=False,
        )
    )

    return constituents


# ============================================================
# PRICE HISTORY
# ============================================================

def download_price_history(
    symbols: list[str],
) -> pd.DataFrame:

    downloaded = yf.download(
        tickers=symbols,
        period=PRICE_PERIOD,
        interval=PRICE_INTERVAL,
        group_by="ticker",
        auto_adjust=True,
        actions=False,
        threads=True,
        progress=True,
        repair=True,
        timeout=30,
        multi_level_index=True,
    )

    if (
        downloaded is None
        or downloaded.empty
    ):
        raise RuntimeError(
            "Yahoo fiyat geçmişi döndürmedi."
        )

    return downloaded


def get_symbol_history(
    downloaded: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:

    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(
        downloaded.columns,
        pd.MultiIndex,
    ):
        level_0 = (
            downloaded.columns
            .get_level_values(0)
        )

        level_1 = (
            downloaded.columns
            .get_level_values(1)
        )

        if symbol in level_0:
            history = downloaded[
                symbol
            ].copy()

        elif symbol in level_1:
            history = downloaded.xs(
                symbol,
                axis=1,
                level=1,
            ).copy()

        else:
            return pd.DataFrame()

    else:
        history = downloaded.copy()

    history = history.dropna(
        how="all"
    )

    history.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        for column in history.columns
    ]

    return history


def build_price_tables(
    downloaded: pd.DataFrame,
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    price_rows: list[dict[str, Any]] = []
    all_history: list[pd.DataFrame] = []

    for symbol in tqdm(
        symbols,
        desc="Fiyat metrikleri hesaplanıyor",
    ):
        history = get_symbol_history(
            downloaded,
            symbol,
        )

        if (
            history.empty
            or "close" not in history.columns
        ):
            price_rows.append(
                {
                    "symbol": symbol,
                    "history_price": np.nan,
                    "price_as_of": pd.NaT,
                    "ma_20": np.nan,
                    "ma_50": np.nan,
                    "ma_200": np.nan,
                    "price_vs_200d_ma_pct": np.nan,
                    "volatility_annual_pct": np.nan,
                    "average_volume_20d": np.nan,
                    "average_dollar_volume_20d": np.nan,
                    "history_error": "NO_PRICE_HISTORY",
                }
            )

            continue

        close = pd.to_numeric(
            history["close"],
            errors="coerce",
        ).dropna()

        if "volume" in history.columns:
            volume = pd.to_numeric(
                history["volume"],
                errors="coerce",
            )
        else:
            volume = None

        if close.empty:
            price_rows.append(
                {
                    "symbol": symbol,
                    "history_price": np.nan,
                    "price_as_of": pd.NaT,
                    "ma_20": np.nan,
                    "ma_50": np.nan,
                    "ma_200": np.nan,
                    "price_vs_200d_ma_pct": np.nan,
                    "volatility_annual_pct": np.nan,
                    "average_volume_20d": np.nan,
                    "average_dollar_volume_20d": np.nan,
                    "history_error": "NO_VALID_CLOSE",
                }
            )

            continue

        latest_price = float(
            close.iloc[-1]
        )

        price_as_of = pd.Timestamp(
            close.index[-1]
        )

        ma_20 = (
            float(close.tail(20).mean())
            if len(close) >= 20
            else np.nan
        )

        ma_50 = (
            float(close.tail(50).mean())
            if len(close) >= 50
            else np.nan
        )

        ma_200 = (
            float(close.tail(200).mean())
            if len(close) >= 200
            else np.nan
        )

        returns = (
            close
            .pct_change()
            .dropna()
        )

        volatility_annual_pct = (
            float(
                returns.tail(252).std()
                * np.sqrt(252)
                * 100
            )
            if len(returns) >= 20
            else np.nan
        )

        average_volume_20d = (
            float(
                volume.tail(20).mean()
            )
            if (
                volume is not None
                and volume.notna().any()
            )
            else np.nan
        )

        average_dollar_volume_20d = (
            latest_price
            * average_volume_20d
            if pd.notna(
                average_volume_20d
            )
            else np.nan
        )

        price_vs_200d_ma_pct = (
            (
                latest_price
                / ma_200
                - 1
            )
            * 100
            if (
                pd.notna(ma_200)
                and ma_200 != 0
            )
            else np.nan
        )

        price_rows.append(
            {
                "symbol": symbol,
                "history_price": latest_price,
                "price_as_of": price_as_of,
                "ma_20": ma_20,
                "ma_50": ma_50,
                "ma_200": ma_200,
                "price_vs_200d_ma_pct": (
                    price_vs_200d_ma_pct
                ),
                "volatility_annual_pct": (
                    volatility_annual_pct
                ),
                "average_volume_20d": (
                    average_volume_20d
                ),
                "average_dollar_volume_20d": (
                    average_dollar_volume_20d
                ),
                "history_error": "",
            }
        )

        history_export = (
            history
            .reset_index()
            .copy()
        )

        history_export.insert(
            0,
            "symbol",
            symbol,
        )

        all_history.append(
            history_export
        )

    price_metrics = pd.DataFrame(
        price_rows
    )

    if not all_history:
        raise RuntimeError(
            "Kullanılabilir fiyat geçmişi bulunamadı."
        )

    complete_price_history = pd.concat(
        all_history,
        ignore_index=True,
    )

    return (
        price_metrics,
        complete_price_history,
    )


# ============================================================
# ANALYST DATA PARSERS
# ============================================================

def read_fast_info_price(
    ticker: yf.Ticker,
) -> float:

    try:
        fast_info = (
            ticker.get_fast_info()
        )

    except Exception:
        try:
            fast_info = ticker.fast_info

        except Exception:
            return np.nan

    possible_keys = [
        "last_price",
        "lastPrice",
        "regular_market_price",
        "regularMarketPrice",
    ]

    for key in possible_keys:
        try:
            value = fast_info[key]
            number = safe_number(value)

            if pd.notna(number):
                return number

        except Exception:
            continue

    return np.nan


def parse_targets(
    raw_targets: Any,
) -> dict[str, float]:

    result = {
        "target_current": np.nan,
        "target_low": np.nan,
        "target_mean": np.nan,
        "target_median": np.nan,
        "target_high": np.nan,
    }

    if raw_targets is None:
        return result

    if isinstance(
        raw_targets,
        pd.DataFrame,
    ):
        if raw_targets.empty:
            return result

        raw_targets = (
            raw_targets
            .iloc[0]
            .to_dict()
        )

    elif isinstance(
        raw_targets,
        pd.Series,
    ):
        raw_targets = (
            raw_targets.to_dict()
        )

    if not isinstance(
        raw_targets,
        dict,
    ):
        return result

    values = {
        normalize_name(key): value
        for key, value
        in raw_targets.items()
    }

    aliases = {
        "target_current": [
            "current",
            "current_price",
        ],
        "target_low": [
            "low",
            "low_price",
        ],
        "target_mean": [
            "mean",
            "mean_price",
        ],
        "target_median": [
            "median",
            "median_price",
        ],
        "target_high": [
            "high",
            "high_price",
        ],
    }

    for (
        output_column,
        possible_keys,
    ) in aliases.items():

        for key in possible_keys:
            if key in values:
                result[
                    output_column
                ] = safe_number(
                    values[key]
                )

                break

    return result


def parse_recommendations(
    raw_recommendations: Any,
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "strong_buy": np.nan,
        "buy": np.nan,
        "hold": np.nan,
        "sell": np.nan,
        "strong_sell": np.nan,
        "recommendation_period": "",
    }

    if not isinstance(
        raw_recommendations,
        pd.DataFrame,
    ):
        return result

    if raw_recommendations.empty:
        return result

    table = raw_recommendations.copy()

    table.columns = [
        normalize_name(column)
        for column in table.columns
    ]

    if "period" in table.columns:
        current_rows = table.loc[
            table["period"]
            .astype(str)
            .str.lower()
            .eq("0m")
        ]

        row = (
            current_rows.iloc[0]
            if not current_rows.empty
            else table.iloc[0]
        )

    else:
        row = table.iloc[0]

    result[
        "recommendation_period"
    ] = str(
        row.get(
            "period",
            "",
        )
    )

    aliases = {
        "strong_buy": [
            "strong_buy",
            "strongbuy",
        ],
        "buy": ["buy"],
        "hold": ["hold"],
        "sell": ["sell"],
        "strong_sell": [
            "strong_sell",
            "strongsell",
        ],
    }

    for (
        output_column,
        possible_columns,
    ) in aliases.items():

        for column in possible_columns:
            if column in row.index:
                result[
                    output_column
                ] = safe_number(
                    row[column]
                )

                break

    return result


def parse_eps_revisions(
    raw_revisions: Any,
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "eps_up_7d": np.nan,
        "eps_up_30d": np.nan,
        "eps_down_7d": np.nan,
        "eps_down_30d": np.nan,
        "eps_revision_period": "",
    }

    if not isinstance(
        raw_revisions,
        pd.DataFrame,
    ):
        return result

    if raw_revisions.empty:
        return result

    table = raw_revisions.copy()

    table.columns = [
        normalize_name(column)
        for column in table.columns
    ]

    normalized_index = [
        str(index)
        .strip()
        .lower()
        for index in table.index
    ]

    row_number = (
        normalized_index.index("0q")
        if "0q" in normalized_index
        else 0
    )

    row = table.iloc[
        row_number
    ]

    result[
        "eps_revision_period"
    ] = str(
        table.index[row_number]
    )

    aliases = {
        "eps_up_7d": [
            "up_last_7_days",
            "uplast7days",
        ],
        "eps_up_30d": [
            "up_last_30_days",
            "uplast30days",
        ],
        "eps_down_7d": [
            "down_last_7_days",
            "downlast7days",
        ],
        "eps_down_30d": [
            "down_last_30_days",
            "downlast30days",
        ],
    }

    for (
        output_column,
        possible_columns,
    ) in aliases.items():

        for column in possible_columns:
            if column in row.index:
                result[
                    output_column
                ] = safe_number(
                    row[column]
                )

                break

    return result


def parse_next_earnings_date(
    raw_earnings_dates: Any,
) -> pd.Timestamp:

    if not isinstance(
        raw_earnings_dates,
        pd.DataFrame,
    ):
        return pd.NaT

    if raw_earnings_dates.empty:
        return pd.NaT

    dates = pd.to_datetime(
        raw_earnings_dates.index,
        errors="coerce",
        utc=True,
    )

    dates = dates[
        ~pd.isna(dates)
    ]

    if len(dates) == 0:
        return pd.NaT

    cutoff = (
        RUN_STARTED_AT_UTC
        - pd.Timedelta(hours=12)
    )

    future_dates = dates[
        dates >= cutoff
    ]

    if len(future_dates) == 0:
        return pd.NaT

    return pd.Timestamp(
        future_dates.min()
    )


def download_one_symbol(
    symbol: str,
) -> dict[str, Any]:

    ticker = yf.Ticker(symbol)

    errors: list[str] = []

    latest_quote = (
        read_fast_info_price(ticker)
    )

    raw_targets, error = (
        request_with_retry(
            ticker.get_analyst_price_targets
        )
    )

    if error:
        errors.append(
            f"targets={error}"
        )

    target_values = parse_targets(
        raw_targets
    )

    raw_recommendations, error = (
        request_with_retry(
            ticker.get_recommendations_summary
        )
    )

    if error:
        errors.append(
            f"recommendations={error}"
        )

    recommendation_values = (
        parse_recommendations(
            raw_recommendations
        )
    )

    raw_revisions, error = (
        request_with_retry(
            ticker.get_eps_revisions
        )
    )

    if error:
        errors.append(
            f"eps_revisions={error}"
        )

    revision_values = parse_eps_revisions(
        raw_revisions
    )

    raw_earnings, error = (
        request_with_retry(
            ticker.get_earnings_dates,
            limit=8,
        )
    )

    if error:
        errors.append(
            f"earnings_dates={error}"
        )

    next_earnings_date = (
        parse_next_earnings_date(
            raw_earnings
        )
    )

    return {
        "symbol": symbol,
        "latest_quote": latest_quote,
        **target_values,
        **recommendation_values,
        **revision_values,
        "next_earnings_date": (
            next_earnings_date
        ),
        "data_fetched_at_utc": (
            pd.Timestamp.now(tz="UTC")
        ),
        "data_errors": " | ".join(
            errors
        ),
    }


def download_analyst_data(
    symbols: list[str],
) -> pd.DataFrame:

    analyst_rows: list[
        dict[str, Any]
    ] = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_to_symbol = {
            executor.submit(
                download_one_symbol,
                symbol,
            ): symbol
            for symbol in symbols
        }

        progress = tqdm(
            as_completed(
                future_to_symbol
            ),
            total=len(
                future_to_symbol
            ),
            desc="Analist verileri indiriliyor",
        )

        for future in progress:
            symbol = future_to_symbol[
                future
            ]

            try:
                analyst_rows.append(
                    future.result()
                )

            except Exception as exc:
                analyst_rows.append(
                    {
                        "symbol": symbol,
                        "latest_quote": np.nan,
                        "target_current": np.nan,
                        "target_low": np.nan,
                        "target_mean": np.nan,
                        "target_median": np.nan,
                        "target_high": np.nan,
                        "strong_buy": np.nan,
                        "buy": np.nan,
                        "hold": np.nan,
                        "sell": np.nan,
                        "strong_sell": np.nan,
                        "recommendation_period": "",
                        "eps_up_7d": np.nan,
                        "eps_up_30d": np.nan,
                        "eps_down_7d": np.nan,
                        "eps_down_30d": np.nan,
                        "eps_revision_period": "",
                        "next_earnings_date": pd.NaT,
                        "data_fetched_at_utc": (
                            pd.Timestamp.now(
                                tz="UTC"
                            )
                        ),
                        "data_errors": (
                            f"UNHANDLED="
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

    analyst_data = pd.DataFrame(
        analyst_rows
    )

    analyst_data = (
        analyst_data
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    return analyst_data


# ============================================================
# MAIN ANALYSIS
# ============================================================

def make_risk_flags(
    row: pd.Series,
) -> str:

    flags: list[str] = []

    if pd.isna(
        row.get("current_price")
    ):
        flags.append(
            "MISSING_CURRENT_PRICE"
        )

    if pd.isna(
        row.get("selected_target_price")
    ):
        flags.append(
            "MISSING_ANALYST_TARGET"
        )

    if row.get("rating_count", 0) == 0:
        flags.append(
            "MISSING_RECOMMENDATIONS"
        )

    if (
        row.get("rating_count", 0)
        < LOW_ANALYST_COVERAGE_THRESHOLD
    ):
        flags.append(
            "LOW_ANALYST_COVERAGE"
        )

    if (
        pd.notna(
            row.get("target_dispersion_pct")
        )
        and row.get("target_dispersion_pct")
        > HIGH_TARGET_DISPERSION_THRESHOLD_PCT
    ):
        flags.append(
            "HIGH_TARGET_DISAGREEMENT"
        )

    if (
        pd.notna(
            row.get("selected_target_upside_pct")
        )
        and row.get("selected_target_upside_pct")
        >= 80
    ):
        flags.append(
            "EXTREME_TARGET_UPSIDE"
        )

    if (
        pd.notna(
            row.get("days_to_next_earnings")
        )
        and 0
        <= row.get("days_to_next_earnings")
        <= 3
    ):
        flags.append(
            "EARNINGS_WITHIN_3_DAYS"
        )

    if (
        pd.notna(
            row.get("days_to_next_earnings")
        )
        and 0
        <= row.get("days_to_next_earnings")
        <= EARNINGS_WARNING_DAYS
    ):
        flags.append(
            "EARNINGS_WITHIN_7_DAYS"
        )

    if (
        pd.notna(
            row.get(
                "days_since_last_earnings",
                np.nan,
            )
        )
        and 0
        <= row.get("days_since_last_earnings")
        <= 5
    ):
        flags.append(
            "POST_EARNINGS_PRICE_DISCOVERY"
        )

    if (
        pd.notna(
            row.get("eps_down_30d")
        )
        and pd.notna(
            row.get("eps_up_30d")
        )
        and row.get("eps_down_30d")
        > row.get("eps_up_30d")
    ):
        flags.append(
            "NEGATIVE_EPS_REVISIONS"
        )

    if (
        pd.notna(
            row.get("current_price")
        )
        and pd.notna(
            row.get("ma_200")
        )
        and row.get("current_price")
        < row.get("ma_200")
    ):
        flags.append(
            "BELOW_200D_MA"
        )

    if (
        pd.notna(
            row.get("volatility_annual_pct")
        )
        and row.get("volatility_annual_pct")
        > HIGH_VOLATILITY_THRESHOLD_PCT
    ):
        flags.append(
            "HIGH_VOLATILITY"
        )

    if (
        pd.notna(
            row.get("average_dollar_volume_20d")
        )
        and row.get("average_dollar_volume_20d")
        < VERY_LOW_AVERAGE_DOLLAR_VOLUME
    ):
        flags.append(
            "VERY_LOW_LIQUIDITY"
        )

    if (
        pd.notna(
            row.get("average_dollar_volume_20d")
        )
        and row.get("average_dollar_volume_20d")
        < LOW_AVERAGE_DOLLAR_VOLUME
    ):
        flags.append(
            "LOW_LIQUIDITY"
        )

    if (
        pd.notna(
            row.get(
                "return_5d_pct",
                np.nan,
            )
        )
        and row["return_5d_pct"]
        <= -10
    ):
        flags.append(
            "SHARP_RECENT_SELLOFF"
        )

    if (
        pd.notna(
            row.get(
                "return_5d_pct",
                np.nan,
            )
        )
        and row["return_5d_pct"]
        >= 15
    ):
        flags.append(
            "SHORT_TERM_OVEREXTENDED"
        )

    if (
        pd.notna(
            row.get(
                "return_20d_pct",
                np.nan,
            )
        )
        and row["return_20d_pct"]
        >= 40
    ):
        flags.append(
            "EXTREME_OVEREXTENSION"
        )

    critical_data_missing = (
        pd.isna(row.get("current_price"))
        or pd.isna(
            row.get("selected_target_price")
        )
        or row.get("rating_count", 0) < 5
    )

    history_error = str(
        row.get(
            "history_error",
            "",
        )
    ).strip()

    yahoo_errors = str(
        row.get(
            "data_errors",
            "",
        )
    ).strip()

    if critical_data_missing:
        flags.append(
            "PARTIAL_DATA"
        )

    elif history_error or yahoo_errors:
        flags.append(
            "OPTIONAL_DATA_MISSING"
        )

    return "|".join(
        dict.fromkeys(flags)
    )


def build_analysis(
    constituents: pd.DataFrame,
    price_metrics: pd.DataFrame,
    analyst_data: pd.DataFrame,
) -> pd.DataFrame:

    analysis = (
        constituents
        .merge(
            price_metrics,
            on="symbol",
            how="left",
            validate="one_to_one",
        )
        .merge(
            analyst_data,
            on="symbol",
            how="left",
            validate="one_to_one",
        )
    )

    numeric_columns = [
        "history_price",
        "latest_quote",
        "target_current",
        "target_low",
        "target_mean",
        "target_median",
        "target_high",
        "strong_buy",
        "buy",
        "hold",
        "sell",
        "strong_sell",
        "eps_up_7d",
        "eps_up_30d",
        "eps_down_7d",
        "eps_down_30d",
        "ma_20",
        "ma_50",
        "ma_200",
        "price_vs_200d_ma_pct",
        "volatility_annual_pct",
        "average_volume_20d",
        "average_dollar_volume_20d",
    ]

    for column in numeric_columns:
        if column in analysis.columns:
            analysis[column] = (
                pd.to_numeric(
                    analysis[column],
                    errors="coerce",
                )
            )

    analysis["current_price"] = (
        analysis["latest_quote"]
        .combine_first(
            analysis["history_price"]
        )
    )

    analysis[
        "current_price_source"
    ] = np.select(
        [
            analysis[
                "latest_quote"
            ].notna(),
            analysis[
                "history_price"
            ].notna(),
        ],
        [
            "FRESH_FAST_INFO",
            "FRESH_DAILY_CLOSE_FALLBACK",
        ],
        default="MISSING",
    )

    analysis["selected_target_price"] = (
        analysis["target_median"]
        .combine_first(
            analysis["target_mean"]
        )
    )

    analysis[
        "selected_target_source"
    ] = np.select(
        [
            analysis[
                "target_median"
            ].notna(),
            analysis[
                "target_mean"
            ].notna(),
        ],
        [
            "MEDIAN",
            "MEAN_FALLBACK",
        ],
        default="MISSING",
    )

    analysis["selected_target"] = (
        analysis["selected_target_price"]
    )

    analysis["selected_target_type"] = (
        analysis["selected_target_source"]
    )

    analysis[
        "selected_target_upside_pct"
    ] = np.where(
        analysis[
            "current_price"
        ].gt(0)
        & analysis[
            "selected_target"
        ].notna(),
        (
            analysis[
                "selected_target_price"
            ]
            / analysis[
                "current_price"
            ]
            - 1
        )
        * 100,
        np.nan,
    )

    recommendation_columns = [
        "strong_buy",
        "buy",
        "hold",
        "sell",
        "strong_sell",
    ]

    analysis["rating_count"] = (
        analysis[
            recommendation_columns
        ]
        .fillna(0)
        .sum(axis=1)
    )

    analysis[
        "positive_rating_pct"
    ] = np.where(
        analysis[
            "rating_count"
        ].gt(0),
        (
            analysis[
                "strong_buy"
            ].fillna(0)
            + analysis[
                "buy"
            ].fillna(0)
        )
        / analysis[
            "rating_count"
        ]
        * 100,
        np.nan,
    )

    analysis[
        "negative_rating_pct"
    ] = np.where(
        analysis[
            "rating_count"
        ].gt(0),
        (
            analysis[
                "sell"
            ].fillna(0)
            + analysis[
                "strong_sell"
            ].fillna(0)
        )
        / analysis[
            "rating_count"
        ]
        * 100,
        np.nan,
    )

    analysis[
        "hold_rating_pct"
    ] = np.where(
        analysis[
            "rating_count"
        ].gt(0),
        analysis[
            "hold"
        ].fillna(0)
        / analysis[
            "rating_count"
        ]
        * 100,
        np.nan,
    )

    sentiment_results = analysis.apply(
        lambda row: score_analyst_sentiment(
            row.get("strong_buy"),
            row.get("buy"),
            row.get("hold"),
            row.get("sell"),
            row.get("strong_sell"),
        ),
        axis=1,
        result_type="expand",
    )

    sentiment_results.columns = [
        "lt_score_sentiment_preview",
        "positive_rating_pct",
        "negative_rating_pct",
        "hold_rating_pct",
        "recommendation_strength_score",
    ]

    for column in sentiment_results.columns:
        analysis[column] = sentiment_results[column]

    analysis[
        "recommendation_score_simple"
    ] = np.where(
        analysis[
            "rating_count"
        ].gt(0),
        (
            2
            * analysis[
                "strong_buy"
            ].fillna(0)
            + analysis[
                "buy"
            ].fillna(0)
            - analysis[
                "sell"
            ].fillna(0)
            - 2
            * analysis[
                "strong_sell"
            ].fillna(0)
        )
        / (
            2
            * analysis[
                "rating_count"
            ]
        ),
        np.nan,
    )

    analysis[
        "mean_vs_median_pct"
    ] = np.where(
        analysis[
            "target_median"
        ].notna()
        & analysis[
            "target_mean"
        ].notna()
        & analysis[
            "target_median"
        ].ne(0),
        (
            analysis[
                "target_mean"
            ]
            / analysis[
                "target_median"
            ]
            - 1
        )
        * 100,
        np.nan,
    )

    analysis[
        "target_dispersion_pct"
    ] = np.where(
        analysis[
            "target_high"
        ].notna()
        & analysis[
            "target_low"
        ].notna()
        & analysis[
            "selected_target"
        ].notna()
        & analysis[
            "selected_target"
        ].ne(0),
        (
            analysis[
                "target_high"
            ]
            - analysis[
                "target_low"
            ]
        )
        / analysis[
            "selected_target_price"
        ]
        * 100,
        np.nan,
    )

    analysis[
        "next_earnings_date"
    ] = pd.to_datetime(
        analysis[
            "next_earnings_date"
        ],
        errors="coerce",
        utc=True,
    )

    analysis[
        "days_to_earnings"
    ] = (
        analysis[
            "next_earnings_date"
        ]
        - RUN_STARTED_AT_UTC
    ).dt.total_seconds() / 86_400

    analysis[
        "days_to_next_earnings"
    ] = analysis[
        "days_to_earnings"
    ]

    analysis[
        "days_since_last_earnings"
    ] = np.where(
        analysis[
            "days_to_next_earnings"
        ].lt(0),
        analysis[
            "days_to_next_earnings"
        ].abs(),
        np.nan,
    )

    analysis[
        "risk_flags"
    ] = analysis.apply(
        make_risk_flags,
        axis=1,
    )

    analysis[
        "has_partial_data"
    ] = (
        analysis[
            "risk_flags"
        ]
        .str.contains(
            "PARTIAL_DATA",
            regex=False,
            na=False,
        )
    )

    return analysis


# ============================================================
# SHORT-TERM METRICS
# ============================================================

def calculate_short_term_metrics(
    symbol_history: pd.DataFrame,
) -> dict[str, Any]:

    close = (
        symbol_history["close"]
        .dropna()
        .astype(float)
    )

    if close.empty:
        return {
            "short_price": np.nan,
            "return_1d_pct": np.nan,
            "return_2d_pct": np.nan,
            "return_5d_pct": np.nan,
            "return_20d_pct": np.nan,
            "short_ma_20": np.nan,
            "short_ma_50": np.nan,
            "distance_from_ma20_pct": np.nan,
            "distance_from_ma50_pct": np.nan,
            "drawdown_from_20d_high_pct": np.nan,
            "short_volatility_20d_pct": np.nan,
            "volatility_20d_annualized_pct": np.nan,
            "negative_days_last_5": np.nan,
            "worst_daily_return_5d_pct": np.nan,
        }

    latest_price = float(
        close.iloc[-1]
    )

    def period_return(
        days: int,
    ) -> float:

        if len(close) <= days:
            return np.nan

        previous_price = float(
            close.iloc[-(days + 1)]
        )

        return (
            latest_price
            / previous_price
            - 1
        ) * 100

    short_ma_20 = (
        float(close.tail(20).mean())
        if len(close) >= 20
        else np.nan
    )

    short_ma_50 = (
        float(close.tail(50).mean())
        if len(close) >= 50
        else np.nan
    )

    high_20 = (
        float(close.tail(20).max())
        if len(close) >= 20
        else np.nan
    )

    recent_returns = (
        close
        .pct_change()
        .dropna()
    )

    short_volatility = (
        float(
            recent_returns
            .tail(20)
            .std()
            * np.sqrt(252)
            * 100
        )
        if len(recent_returns) >= 20
        else np.nan
    )

    negative_days_last_5 = (
        int(
            (
                recent_returns
                .tail(5)
                < 0
            ).sum()
        )
        if len(recent_returns) >= 5
        else np.nan
    )

    worst_daily_return_5d_pct = (
        float(
            recent_returns
            .tail(5)
            .min()
            * 100
        )
        if len(recent_returns) >= 5
        else np.nan
    )

    return {
        "short_price": latest_price,
        "return_1d_pct": (
            period_return(1)
        ),
        "return_2d_pct": (
            period_return(2)
        ),
        "return_5d_pct": (
            period_return(5)
        ),
        "return_20d_pct": (
            period_return(20)
        ),
        "short_ma_20": short_ma_20,
        "short_ma_50": short_ma_50,
        "distance_from_ma20_pct": (
            (
                latest_price
                / short_ma_20
                - 1
            )
            * 100
            if (
                pd.notna(short_ma_20)
                and short_ma_20 != 0
            )
            else np.nan
        ),
        "distance_from_ma50_pct": (
            (
                latest_price
                / short_ma_50
                - 1
            )
            * 100
            if (
                pd.notna(short_ma_50)
                and short_ma_50 != 0
            )
            else np.nan
        ),
        "drawdown_from_20d_high_pct": (
            (
                latest_price
                / high_20
                - 1
            )
            * 100
            if (
                pd.notna(high_20)
                and high_20 != 0
            )
            else np.nan
        ),
        "short_volatility_20d_pct": (
            short_volatility
        ),
        "volatility_20d_annualized_pct": (
            short_volatility
        ),
        "negative_days_last_5": (
            negative_days_last_5
        ),
        "worst_daily_return_5d_pct": (
            worst_daily_return_5d_pct
        ),
    }


def build_short_term_metrics(
    complete_price_history: pd.DataFrame,
) -> pd.DataFrame:

    price_history = (
        complete_price_history
        .copy()
    )

    possible_date_columns = [
        column
        for column
        in price_history.columns
        if str(column).lower()
        in {
            "date",
            "datetime",
            "index",
        }
    ]

    if not possible_date_columns:
        raise KeyError(
            "Fiyat geçmişinde tarih sütunu bulunamadı."
        )

    date_column = (
        possible_date_columns[0]
    )

    price_history = (
        price_history
        .rename(
            columns={
                date_column: "price_date"
            }
        )
    )

    price_history[
        "price_date"
    ] = pd.to_datetime(
        price_history[
            "price_date"
        ],
        errors="coerce",
        utc=True,
    )

    price_history["close"] = (
        pd.to_numeric(
            price_history["close"],
            errors="coerce",
        )
    )

    price_history = (
        price_history
        .dropna(
            subset=[
                "symbol",
                "price_date",
                "close",
            ]
        )
        .sort_values(
            [
                "symbol",
                "price_date",
            ]
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for (
        symbol,
        symbol_history,
    ) in price_history.groupby(
        "symbol"
    ):
        metrics = (
            calculate_short_term_metrics(
                symbol_history
            )
        )

        metrics["symbol"] = symbol

        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# SCORING HELPERS
# ============================================================

def calculate_pullback_score(
    drawdown: Any,
) -> float:

    value = safe_number(
        drawdown
    )

    if pd.isna(value):
        return np.nan

    # Zirveye fazla yakın
    if value >= -3:
        return 55

    # Kontrollü geri çekilme
    if value >= -8:
        return 100

    # Orta düzeyde düzeltme
    if value >= -15:
        return (
            60
            + (
                value + 15
            )
            / 7
            * 40
        )

    # Sert düşüş
    if value >= -25:
        return (
            10
            + (
                value + 25
            )
            / 10
            * 50
        )

    # Falling knife riski
    return 0


def earnings_timing_score(
    days_to_earnings: Any,
) -> float:

    days = safe_number(
        days_to_earnings
    )

    if pd.isna(days):
        return 50

    if days < 0:
        return 70

    if days <= 3:
        return 15

    if days <= 7:
        return 35

    if days <= 21:
        return 70

    return 100


def ma_distance_score(
    value: Any,
) -> float:

    distance = safe_number(
        value
    )

    if pd.isna(distance):
        return 50

    if distance < -15:
        return 5

    if distance < -5:
        return (
            5
            + (
                distance + 15
            )
            / 10
            * 55
        )

    if distance <= 5:
        return (
            60
            + (
                distance + 5
            )
            / 10
            * 40
        )

    if distance <= 15:
        return (
            100
            - (
                distance - 5
            )
            / 10
            * 30
        )

    if distance <= 30:
        return (
            70
            - (
                distance - 15
            )
            / 15
            * 40
        )

    return 20


# ============================================================
# LONG-TERM AND SHORT-TERM SCORES
# ============================================================

def calculate_dual_scores(
    analysis: pd.DataFrame,
    short_term_metrics: pd.DataFrame,
) -> pd.DataFrame:

    scores = analysis.merge(
        short_term_metrics,
        on="symbol",
        how="left",
        validate="one_to_one",
    )

    rating_count = pd.to_numeric(
        scores["rating_count"],
        errors="coerce",
    ).fillna(0)

    scores["price_vs_ma20_pct"] = scores[
        "distance_from_ma20_pct"
    ]
    scores["price_vs_ma50_pct"] = scores[
        "distance_from_ma50_pct"
    ]

    if "volatility_20d_annualized_pct" not in scores.columns:
        scores["volatility_20d_annualized_pct"] = scores[
            "short_volatility_20d_pct"
        ]

    sector_counts = scores.groupby(
        "sector"
    )["volatility_20d_annualized_pct"].transform(
        "count"
    )
    scores["sector_volatility_percentile"] = (
        scores.groupby("sector")[
            "volatility_20d_annualized_pct"
        ]
        .rank(pct=True)
        .mul(100)
        .where(sector_counts >= 5)
    )

    scores["volatility_scoring_method"] = np.where(
        scores["sector_volatility_percentile"].notna(),
        "SECTOR_RELATIVE",
        "ABSOLUTE_FALLBACK",
    )

    scores["lt_score_upside"] = scores[
        "selected_target_upside_pct"
    ].apply(score_target_upside)

    sentiment_results = scores.apply(
        lambda row: score_analyst_sentiment(
            row.get("strong_buy"),
            row.get("buy"),
            row.get("hold"),
            row.get("sell"),
            row.get("strong_sell"),
        ),
        axis=1,
        result_type="expand",
    )
    sentiment_results.columns = [
        "lt_score_sentiment",
        "positive_rating_pct",
        "negative_rating_pct",
        "hold_rating_pct",
        "recommendation_strength_score",
    ]
    for column in sentiment_results.columns:
        scores[column] = sentiment_results[column]

    eps_results = scores.apply(
        lambda row: score_eps_revisions(
            row.get("eps_up_30d"),
            row.get("eps_down_30d"),
        ),
        axis=1,
        result_type="expand",
    )
    scores["lt_score_eps_revisions"] = eps_results[0]
    scores["eps_revision_data_available"] = eps_results[1]

    scores["lt_score_target_agreement"] = scores[
        "target_dispersion_pct"
    ].apply(score_target_agreement)

    scores["lt_score_long_trend"] = scores[
        "price_vs_200d_ma_pct"
    ].apply(score_long_term_trend)

    scores["analyst_coverage_confidence"] = rating_count.apply(
        calculate_coverage_confidence
    )
    scores["analyst_coverage_multiplier"] = scores[
        "analyst_coverage_confidence"
    ].apply(calculate_coverage_multiplier)

    long_term_weights = {
        "upside": 0.35,
        "sentiment": 0.30,
        "eps_revisions": 0.20,
        "target_agreement": 0.10,
        "long_trend": 0.05,
    }

    long_results = scores.apply(
        lambda row: weighted_score_available(
            {
                "upside": row.get("lt_score_upside"),
                "sentiment": row.get("lt_score_sentiment"),
                "eps_revisions": row.get("lt_score_eps_revisions"),
                "target_agreement": row.get("lt_score_target_agreement"),
                "long_trend": row.get("lt_score_long_trend"),
            },
            long_term_weights,
        ),
        axis=1,
        result_type="expand",
    )

    scores["analyst_long_term_raw_score"] = long_results[0]
    scores["analyst_long_term_data_coverage_pct"] = (
        long_results[1] * 100
    )

    long_term_critical_missing = (
        scores["current_price"].isna()
        | scores["selected_target_price"].isna()
        | rating_count.lt(5)
    )

    scores["analyst_long_term_status"] = RANKED
    scores.loc[
        scores["analyst_long_term_data_coverage_pct"].lt(
            100
        ),
        "analyst_long_term_status",
    ] = PARTIAL_DATA
    scores.loc[
        long_term_critical_missing
        | scores["analyst_long_term_data_coverage_pct"].lt(
            MIN_ANALYST_LONG_TERM_COVERAGE * 100
        ),
        "analyst_long_term_status",
    ] = INSUFFICIENT_DATA

    scores.loc[
        scores["analyst_long_term_status"].eq(
            INSUFFICIENT_DATA
        ),
        "analyst_long_term_raw_score",
    ] = np.nan

    scores["long_term_risk_penalty"] = 0.0
    avg_dollar_volume = pd.to_numeric(
        scores["average_dollar_volume_20d"],
        errors="coerce",
    )
    scores["long_term_risk_penalty"] += np.select(
        [
            avg_dollar_volume.lt(VERY_LOW_AVERAGE_DOLLAR_VOLUME),
            avg_dollar_volume.lt(LOW_AVERAGE_DOLLAR_VOLUME),
            avg_dollar_volume.lt(20_000_000),
        ],
        [5, 3, 1],
        default=0,
    )
    scores["long_term_risk_penalty"] += np.where(
        scores["volatility_annual_pct"].gt(
            HIGH_VOLATILITY_THRESHOLD_PCT
        ),
        1,
        0,
    )
    scores["long_term_risk_penalty"] += np.where(
        scores["selected_target_upside_pct"].ge(80),
        2,
        0,
    )
    scores["long_term_risk_penalty"] += np.where(
        scores["target_dispersion_pct"].gt(
            HIGH_TARGET_DISPERSION_THRESHOLD_PCT
        ),
        2,
        0,
    )

    scores["analyst_long_term_score"] = (
        scores["analyst_long_term_raw_score"]
        * scores["analyst_coverage_multiplier"]
        - scores["long_term_risk_penalty"]
    ).clip(lower=0, upper=100)
    scores.loc[
        scores["analyst_long_term_status"].eq(
            INSUFFICIENT_DATA
        ),
        "analyst_long_term_score",
    ] = np.nan
    scores["long_term_score"] = scores[
        "analyst_long_term_score"
    ]

    scores["st_score_momentum"] = scores.apply(
        lambda row: score_momentum(
            row.get("return_5d_pct"),
            row.get("return_20d_pct"),
        ),
        axis=1,
    )
    scores["st_score_ma_alignment"] = scores.apply(
        lambda row: score_ma_alignment(
            row.get("price_vs_ma20_pct"),
            row.get("price_vs_ma50_pct"),
        ),
        axis=1,
    )
    scores["st_score_pullback_quality"] = scores[
        "drawdown_from_20d_high_pct"
    ].apply(score_pullback_quality)
    selloff_results = scores.apply(
        lambda row: score_selloff_stability(
            row.get("return_2d_pct"),
            row.get("negative_days_last_5"),
            row.get("worst_daily_return_5d_pct"),
        ),
        axis=1,
        result_type="expand",
    )
    scores["st_score_selloff_stability"] = selloff_results[0]
    scores["selloff_stability_data_coverage_pct"] = (
        selloff_results[1]
    )
    scores["selloff_stability_status"] = selloff_results[2]
    scores["st_score_volatility"] = np.where(
        scores["sector_volatility_percentile"].notna(),
        scores["sector_volatility_percentile"].apply(
            score_sector_volatility_percentile
        ),
        scores["volatility_20d_annualized_pct"].apply(
            score_absolute_volatility
        ),
    )
    scores["st_score_earnings_timing"] = scores.apply(
        lambda row: score_earnings_timing(
            row.get("days_to_next_earnings"),
            row.get("days_since_last_earnings"),
        ),
        axis=1,
    )
    scores["analyst_backed_pullback_bonus"] = scores.apply(
        lambda row: calculate_analyst_backed_pullback_bonus(
            row.get("selected_target_upside_pct"),
            row.get("positive_rating_pct"),
            row.get("return_5d_pct"),
            row.get("return_1d_pct"),
        ),
        axis=1,
    )

    short_term_weights = {
        "momentum": 0.25,
        "ma_alignment": 0.25,
        "pullback_quality": 0.20,
        "selloff_stability": 0.15,
        "volatility": 0.10,
        "earnings_timing": 0.05,
    }

    short_results = scores.apply(
        lambda row: weighted_score_available(
            {
                "momentum": row.get("st_score_momentum"),
                "ma_alignment": row.get("st_score_ma_alignment"),
                "pullback_quality": row.get("st_score_pullback_quality"),
                "selloff_stability": row.get("st_score_selloff_stability"),
                "volatility": row.get("st_score_volatility"),
                "earnings_timing": row.get("st_score_earnings_timing"),
            },
            short_term_weights,
        ),
        axis=1,
        result_type="expand",
    )

    scores["short_term_raw_score"] = short_results[0]
    scores["short_term_data_coverage_pct"] = (
        short_results[1] * 100
    )

    short_term_critical_missing = (
        scores["current_price"].isna()
        | scores["return_5d_pct"].isna()
        | scores["return_20d_pct"].isna()
        | scores["price_vs_ma20_pct"].isna()
        | scores["price_vs_ma50_pct"].isna()
        | scores["volatility_20d_annualized_pct"].isna()
        | scores["drawdown_from_20d_high_pct"].isna()
    )

    scores["short_term_status"] = RANKED
    scores.loc[
        scores["short_term_data_coverage_pct"].lt(100),
        "short_term_status",
    ] = PARTIAL_DATA
    scores.loc[
        short_term_critical_missing
        | scores["short_term_data_coverage_pct"].lt(
            MIN_SHORT_TERM_COVERAGE * 100
        ),
        "short_term_status",
    ] = INSUFFICIENT_DATA

    scores.loc[
        scores["short_term_status"].eq(INSUFFICIENT_DATA),
        "short_term_raw_score",
    ] = np.nan

    scores["short_term_risk_penalty"] = 0.0
    scores["short_term_risk_penalty"] += np.where(
        avg_dollar_volume.lt(VERY_LOW_AVERAGE_DOLLAR_VOLUME),
        10,
        0,
    )
    scores["short_term_risk_penalty"] += np.where(
        scores["return_5d_pct"].le(-10),
        3,
        0,
    )
    scores["short_term_risk_penalty"] += np.where(
        scores["return_20d_pct"].ge(40),
        3,
        0,
    )

    scores["short_term_entry_score"] = (
        scores["short_term_raw_score"]
        + scores["analyst_backed_pullback_bonus"]
        - scores["short_term_risk_penalty"]
    ).clip(lower=0, upper=100)
    scores.loc[
        scores["short_term_status"].eq(INSUFFICIENT_DATA),
        "short_term_entry_score",
    ] = np.nan

    scores["risk_flags"] = scores.apply(
        make_risk_flags,
        axis=1,
    )

    scores["analyst_long_term_category"] = scores[
        "analyst_long_term_score"
    ].apply(assign_long_term_category)
    scores["long_term_category"] = scores[
        "analyst_long_term_category"
    ]
    scores["short_term_category"] = scores[
        "short_term_entry_score"
    ].apply(assign_short_term_category)

    profiles = scores.apply(
        lambda row: assign_candidate_profile(
            row.get("analyst_long_term_score"),
            row.get("short_term_entry_score"),
        ),
        axis=1,
        result_type="expand",
    )
    scores["candidate_profile"] = profiles[0]
    scores["candidate_profile_explanation"] = profiles[1]
    scores["candidate_profile_priority"] = scores[
        "candidate_profile"
    ].apply(candidate_profile_priority)

    scores["combined_score"] = np.where(
        scores["analyst_long_term_score"].notna()
        & scores["short_term_entry_score"].notna(),
        0.60 * scores["analyst_long_term_score"]
        + 0.40 * scores["short_term_entry_score"],
        np.nan,
    )

    scores["overall_data_quality"] = np.select(
        [
            scores["analyst_long_term_status"].eq(RANKED)
            & scores["short_term_status"].eq(RANKED)
            & scores["analyst_long_term_data_coverage_pct"].ge(90)
            & scores["short_term_data_coverage_pct"].ge(90),
            scores["analyst_long_term_status"].isin(
                [RANKED, PARTIAL_DATA]
            )
            & scores["short_term_status"].isin(
                [RANKED, PARTIAL_DATA]
            ),
            scores["analyst_long_term_status"].isin(
                [RANKED, PARTIAL_DATA]
            )
            | scores["short_term_status"].isin(
                [RANKED, PARTIAL_DATA]
            ),
        ],
        ["HIGH", "MEDIUM", "LOW"],
        default="INSUFFICIENT",
    )

    drivers = scores.apply(
        format_driver_text,
        axis=1,
        result_type="expand",
    )
    scores["top_positive_drivers"] = drivers[0]
    scores["top_negative_drivers"] = drivers[1]

    long_rankable = scores["analyst_long_term_status"].isin(
        [RANKED, PARTIAL_DATA]
    )
    short_rankable = scores["short_term_status"].isin(
        [RANKED, PARTIAL_DATA]
    )

    scores["long_term_rank"] = pd.Series(
        pd.NA,
        index=scores.index,
        dtype="Int64",
    )
    scores.loc[
        long_rankable,
        "long_term_rank",
    ] = scores.loc[
        long_rankable,
        "analyst_long_term_score",
    ].rank(method="min", ascending=False).astype("Int64")

    scores["short_term_rank"] = pd.Series(
        pd.NA,
        index=scores.index,
        dtype="Int64",
    )
    scores.loc[
        short_rankable,
        "short_term_rank",
    ] = scores.loc[
        short_rankable,
        "short_term_entry_score",
    ].rank(method="min", ascending=False).astype("Int64")

    return scores


# ============================================================
# RANKING TABLES
# ============================================================

def create_rankings(
    dual_scores: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    long_rankable = dual_scores[
        "analyst_long_term_status"
    ].isin([RANKED, PARTIAL_DATA])
    short_rankable = dual_scores[
        "short_term_status"
    ].isin([RANKED, PARTIAL_DATA])

    long_term_ranking = (
        dual_scores.loc[long_rankable]
        .sort_values(
            [
                "analyst_long_term_score",
                "short_term_entry_score",
                "positive_rating_pct",
                "rating_count",
            ],
            ascending=[False, False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    short_term_ranking = (
        dual_scores.loc[short_rankable]
        .sort_values(
            [
                "short_term_entry_score",
                "analyst_long_term_score",
                "positive_rating_pct",
            ],
            ascending=[False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    combined_candidates = (
        dual_scores.loc[
            dual_scores["analyst_long_term_score"].ge(60)
            & dual_scores["short_term_entry_score"].ge(60)
            & dual_scores["analyst_long_term_status"].isin(
                [RANKED, PARTIAL_DATA]
            )
            & dual_scores["short_term_status"].isin(
                [RANKED, PARTIAL_DATA]
            )
            & dual_scores["analyst_long_term_data_coverage_pct"].ge(80)
            & dual_scores["short_term_data_coverage_pct"].ge(80)
            & dual_scores["overall_data_quality"].ne("INSUFFICIENT")
        ]
        .copy()
    )

    combined_candidates = (
        combined_candidates
        .sort_values(
            [
                "candidate_profile_priority",
                "combined_score",
                "short_term_entry_score",
                "analyst_long_term_score",
            ],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
    )

    combined_candidates.insert(
        0,
        "combined_rank",
        range(1, len(combined_candidates) + 1),
    )

    strong_candidates = combined_candidates.loc[
        combined_candidates["candidate_profile"].eq("STRONG CANDIDATE")
    ].copy()
    wait_for_entry = dual_scores.loc[
        dual_scores["candidate_profile"].eq("ATTRACTIVE, WAIT FOR ENTRY")
    ].copy()
    tactical_candidates = combined_candidates.loc[
        combined_candidates["candidate_profile"].eq("TACTICAL CANDIDATE")
    ].copy()
    momentum_only = dual_scores.loc[
        dual_scores["candidate_profile"].eq("MOMENTUM ONLY")
    ].copy()
    insufficient_data = dual_scores.loc[
        dual_scores["analyst_long_term_status"].eq(INSUFFICIENT_DATA)
        | dual_scores["short_term_status"].eq(INSUFFICIENT_DATA)
    ].copy()

    return (
        long_term_ranking,
        short_term_ranking,
        combined_candidates,
        strong_candidates,
        wait_for_entry,
        tactical_candidates,
        momentum_only,
        insufficient_data,
    )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

def print_rankings(
    long_term_ranking: pd.DataFrame,
    short_term_ranking: pd.DataFrame,
    combined_candidates: pd.DataFrame,
) -> None:

    pd.set_option(
        "display.max_columns",
        50,
    )

    pd.set_option(
        "display.width",
        250,
    )

    pd.set_option(
        "display.max_colwidth",
        45,
    )

    long_columns = [
        "long_term_rank",
        "short_term_rank",
        "symbol",
        "company_name",
        "analyst_long_term_raw_score",
        "analyst_long_term_score",
        "analyst_long_term_category",
        "analyst_long_term_status",
        "analyst_coverage_confidence",
        "analyst_coverage_multiplier",
        "analyst_long_term_data_coverage_pct",
        "short_term_entry_score",
        "short_term_category",
        "current_price",
        "selected_target_upside_pct",
        "positive_rating_pct",
        "rating_count",
        "target_dispersion_pct",
        "return_5d_pct",
        "analyst_backed_pullback_bonus",
        "risk_flags",
    ]

    short_columns = [
        "short_term_rank",
        "long_term_rank",
        "symbol",
        "company_name",
        "short_term_entry_score",
        "short_term_category",
        "analyst_long_term_score",
        "analyst_long_term_category",
        "current_price",
        "return_1d_pct",
        "return_2d_pct",
        "analyst_backed_pullback_bonus",
        "return_5d_pct",
        "return_20d_pct",
        "drawdown_from_20d_high_pct",
        "short_volatility_20d_pct",
        "selloff_stability_data_coverage_pct",
        "selloff_stability_status",
        "risk_flags",
    ]

    combined_columns = [
        "combined_rank",
        "symbol",
        "company_name",
        "combined_score",
        "analyst_long_term_score",
        "short_term_entry_score",
        "candidate_profile",
        "candidate_profile_priority",
        "current_price",
        "selected_target_upside_pct",
        "positive_rating_pct",
        "rating_count",
        "return_2d_pct",
        "return_5d_pct",
        "risk_flags",
    ]

    print("\n")
    print("=" * 120)
    print("ANALYST-BASED UZUN VADELİ POTANSİYEL SIRALAMASI")
    print("=" * 120)

    print(
        long_term_ranking[
            long_columns
        ]
        .head(TOP_N)
        .round(2)
        .to_string(index=False)
    )

    print("\n")
    print("=" * 120)
    print("KISA VADELİ GİRİŞ SIRALAMASI")
    print("=" * 120)

    print(
        short_term_ranking[
            short_columns
        ]
        .head(TOP_N)
        .round(2)
        .to_string(index=False)
    )

    print("\n")
    print("=" * 120)
    print(
        "UZUN VE KISA VADE SKORU EN AZ 60 OLAN ADAYLAR"
    )
    print("=" * 120)

    if combined_candidates.empty:
        print(
            "Bu çalıştırmada iki eşikten de geçen aday yok."
        )

    else:
        print(
            combined_candidates[
                combined_columns
            ]
            .head(TOP_N)
            .round(2)
            .to_string(index=False)
        )


def print_ticker_details(
    dual_scores: pd.DataFrame,
    ticker_input: str,
) -> None:

    ticker = (
        ticker_input
        .strip()
        .upper()
        .replace(
            ".",
            "-",
        )
    )

    result = dual_scores.loc[
        dual_scores[
            "symbol"
        ].eq(ticker)
    ].copy()

    print("\n")
    print("=" * 120)
    print(
        f"TICKER DETAYI: {ticker}"
    )
    print("=" * 120)

    if result.empty:
        print(
            f"{ticker}, S&P 500 sonuçlarında bulunamadı."
        )

        return

    row = result.iloc[0]

    def display_value(
        value: Any,
        suffix: str = "",
    ) -> str:

        number = safe_float(value)
        if pd.isna(number):
            if pd.isna(value):
                return "N/A"
            return str(value)
        return f"{number:,.2f}{suffix}"

    print(
        f"{row['symbol']} — "
        f"{row['company_name']}"
    )

    print(
        f"Analyst-based uzun vade: "
        f"rank {row['long_term_rank']}"
        f"/{len(dual_scores)}, "
        f"raw {display_value(row['analyst_long_term_raw_score'])}, "
        f"adjusted {display_value(row['analyst_long_term_score'])}, "
        f"{row['analyst_long_term_category']}, "
        f"{row['analyst_long_term_status']}"
    )

    print(
        f"Kısa vade: "
        f"rank {row['short_term_rank']}"
        f"/{len(dual_scores)}, "
        f"score "
        f"raw {display_value(row['short_term_raw_score'])}, "
        f"entry {display_value(row['short_term_entry_score'])}, "
        f"{row['short_term_category']}, "
        f"{row['short_term_status']}"
    )

    print(
        f"Analyst coverage confidence: "
        f"{display_value(row['analyst_coverage_confidence'])}"
    )

    print(
        f"Analyst coverage multiplier: "
        f"{display_value(row['analyst_coverage_multiplier'])}"
    )

    print(
        f"Long-term risk penalty: "
        f"{display_value(row['long_term_risk_penalty'])}"
    )

    print(
        f"Data coverage: long "
        f"{display_value(row['analyst_long_term_data_coverage_pct'], '%')}, "
        f"short {display_value(row['short_term_data_coverage_pct'], '%')}"
    )

    print(
        f"Combined: "
        f"{display_value(row['combined_score'])}, "
        f"{row['candidate_profile']}"
    )

    print(
        f"Current price: "
        f"{display_value(row['current_price'])}"
    )

    print(
        f"Selected target: "
        f"{display_value(row['selected_target_price'])}"
    )

    print(
        f"Target upside: "
        f"{display_value(row['selected_target_upside_pct'], '%')}"
    )

    print(
        f"Positive ratings: "
        f"{display_value(row['positive_rating_pct'], '%')} "
        f"from "
        f"{display_value(row['rating_count'])} ratings"
    )

    print(
        f"Risk flags: "
        f"{row['risk_flags'] or 'None'}"
    )

    print(
        f"Positive drivers: "
        f"{row['top_positive_drivers']}"
    )

    print(
        f"Negative drivers: "
        f"{row['top_negative_drivers']}"
    )

    detail_columns = [
        "symbol",
        "company_name",
        "sector",
        "long_term_rank",
        "analyst_long_term_raw_score",
        "analyst_long_term_score",
        "analyst_long_term_category",
        "analyst_long_term_status",
        "analyst_coverage_confidence",
        "analyst_coverage_multiplier",
        "analyst_long_term_data_coverage_pct",
        "short_term_rank",
        "short_term_raw_score",
        "short_term_entry_score",
        "short_term_category",
        "short_term_status",
        "short_term_data_coverage_pct",
        "analyst_backed_pullback_bonus",
        "combined_score",
        "candidate_profile",
        "candidate_profile_priority",
        "candidate_profile_explanation",
        "current_price",
        "current_price_source",
        "price_as_of",
        "target_low",
        "target_mean",
        "target_median",
        "target_high",
        "selected_target_price",
        "selected_target_source",
        "selected_target_upside_pct",
        "strong_buy",
        "buy",
        "hold",
        "sell",
        "strong_sell",
        "positive_rating_pct",
        "negative_rating_pct",
        "hold_rating_pct",
        "recommendation_strength_score",
        "rating_count",
        "target_dispersion_pct",
        "eps_up_7d",
        "eps_down_7d",
        "eps_up_30d",
        "eps_down_30d",
        "return_1d_pct",
        "return_2d_pct",
        "return_5d_pct",
        "return_20d_pct",
        "distance_from_ma20_pct",
        "distance_from_ma50_pct",
        "drawdown_from_20d_high_pct",
        "short_volatility_20d_pct",
        "negative_days_last_5",
        "next_earnings_date",
        "days_to_earnings",
        "days_to_next_earnings",
        "days_since_last_earnings",
        "lt_score_upside",
        "lt_score_sentiment",
        "lt_score_coverage",
        "lt_score_target_agreement",
        "lt_score_eps_revisions",
        "lt_score_long_trend",
        "long_term_base_score",
        "long_term_risk_penalty",
        "short_term_raw_score",
        "short_term_risk_penalty",
        "st_score_momentum",
        "st_score_ma_alignment",
        "st_score_selloff_stability",
        "selloff_stability_data_coverage_pct",
        "selloff_stability_status",
        "st_score_pullback_quality",
        "st_score_volatility",
        "st_score_earnings_timing",
        "risk_flags",
        "top_positive_drivers",
        "top_negative_drivers",
        "data_errors",
        "data_fetched_at_utc",
    ]

    detail_columns = [
        column
        for column in detail_columns
        if column in result.columns
    ]

    details = (
        result[
            detail_columns
        ]
        .T
    )

    details.columns = ["value"]

    print("\n")
    print(
        details.to_string()
    )


# ============================================================
# EXPORT
# ============================================================

def prepare_dataframe_for_excel(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    excel_frame = frame.copy()

    def normalize_excel_value(
        value: Any,
    ) -> Any:

        if isinstance(value, pd.Timestamp):
            if value.tzinfo is not None:
                return (
                    value
                    .tz_convert("UTC")
                    .tz_localize(None)
                )

        return value

    for column in excel_frame.columns:
        if isinstance(
            excel_frame[column].dtype,
            pd.DatetimeTZDtype,
        ):
            excel_frame[column] = (
                excel_frame[column]
                .dt.tz_convert("UTC")
                .dt.tz_localize(None)
            )

        elif excel_frame[column].dtype == "object":
            excel_frame[column] = (
                excel_frame[column]
                .map(normalize_excel_value)
            )

    return excel_frame


def build_methodology_table() -> pd.DataFrame:

    rows = [
        ("model_version", SCORING_MODEL_VERSION),
        (
            "purpose",
            "Research screener; not a buy/sell recommendation.",
        ),
        (
            "analyst_long_term_score",
            "Analyst-based score: raw score uses target upside 35%, sentiment 30%, EPS revisions 20%, target agreement 10%, MA200 trend 5%; final score applies analyst_coverage_multiplier and modest risk penalties.",
        ),
        (
            "short_term_entry_score",
            "Technical entry score: recent entry timing 25%, MA alignment 25%, pullback 20%, selloff stability 15%, volatility 10%, earnings timing 5%. Recent entry timing favors a controlled 4%-7% five-day decline over a five-day increase, with the 20-day return retained as a lower-weight trend check. A 15-point fundamental bonus requires at least 25% target upside, more than 90% positive analyst ratings, and a latest-day return within +/-0.5%.",
        ),
        (
            "missing_data",
            "Critical missing data produces NaN and INSUFFICIENT_DATA; optional missing components are excluded with dynamic weight normalization.",
        ),
        (
            "confidence",
            "Analyst coverage confidence is softened through analyst_coverage_multiplier = 0.70 + 0.30 * analyst_coverage_confidence; fewer than 5 ratings is insufficient.",
        ),
        (
            "risk_penalties",
            "Long-term penalties: very low liquidity -5, low liquidity -3, liquidity warning -1, high volatility -1, extreme target upside -2, high target disagreement -2. Short-term penalties: very low liquidity -10, sharp recent selloff -3, extreme overextension -3.",
        ),
        (
            "selloff_stability",
            "Selloff stability uses return_2d_pct 45%, negative_days_last_5 30%, worst_daily_return_5d_pct 25%; missing subcomponents are excluded and subweights are dynamically normalized.",
        ),
        (
            "candidate_profiles",
            "ATTRACTIVE BUT TECHNICALLY WEAK means strong analyst-based attractiveness with severely weak technical setup; ATTRACTIVE, WAIT FOR ENTRY means strong analyst-based attractiveness but timing is not favorable yet.",
        ),
        (
            "categories",
            "Long term: WEAK <45, NEUTRAL <60, POSITIVE <75, STRONG >=75. Short term: WAIT/AVOID <40, NEUTRAL <60, WATCH <75, FAVORABLE ENTRY >=75.",
        ),
        (
            "limitations",
            "Scores can be affected by stale analyst targets, delayed revisions, market shocks, sector events, and incomplete source data.",
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["topic", "description"],
    )


def build_run_metadata(
    constituents: pd.DataFrame,
    price_metrics: pd.DataFrame,
    analyst_data: pd.DataFrame,
    dual_scores: pd.DataFrame,
) -> pd.DataFrame:

    metadata = {
        "run_timestamp_utc": RUN_STARTED_AT_UTC,
        "scoring_model_version": SCORING_MODEL_VERSION,
        "sp500_constituents": len(constituents),
        "successful_price_rows": int(
            price_metrics["history_price"].notna().sum()
        ),
        "complete_price_history_symbols": int(
            dual_scores["short_term_status"]
            .isin([RANKED, PARTIAL_DATA])
            .sum()
        ),
        "analyst_data_rows": len(analyst_data),
        "ranked_long_term": int(
            dual_scores["analyst_long_term_status"]
            .isin([RANKED, PARTIAL_DATA])
            .sum()
        ),
        "ranked_short_term": int(
            dual_scores["short_term_status"]
            .isin([RANKED, PARTIAL_DATA])
            .sum()
        ),
        "partial_data_rows": int(
            (
                dual_scores["analyst_long_term_status"].eq(PARTIAL_DATA)
                | dual_scores["short_term_status"].eq(PARTIAL_DATA)
            ).sum()
        ),
        "insufficient_data_rows": int(
            dual_scores["overall_data_quality"]
            .eq("INSUFFICIENT")
            .sum()
        ),
        "data_sources": "GitHub datasets S&P 500 constituents; Yahoo Finance via yfinance.",
        "script_version": SCORING_MODEL_VERSION,
    }

    return pd.DataFrame(
        metadata.items(),
        columns=["field", "value"],
    )


def validate_scores(
    dual_scores: pd.DataFrame,
    combined_candidates: pd.DataFrame,
) -> None:

    score_columns = [
        "analyst_long_term_raw_score",
        "analyst_long_term_score",
        "short_term_raw_score",
        "short_term_entry_score",
        "combined_score",
    ]

    outside_range = 0
    for column in score_columns:
        values = pd.to_numeric(
            dual_scores[column],
            errors="coerce",
        )
        outside_range += int(
            (
                values.notna()
                & ~values.between(0, 100)
            ).sum()
        )

    invalid_long = int(
        (
            dual_scores["analyst_long_term_status"]
            .eq(INSUFFICIENT_DATA)
            & dual_scores["analyst_long_term_score"].notna()
        ).sum()
    )
    invalid_short = int(
        (
            dual_scores["short_term_status"]
            .eq(INSUFFICIENT_DATA)
            & dual_scores["short_term_entry_score"].notna()
        ).sum()
    )
    invalid_combined = int(
        (
            ~(
                combined_candidates["analyst_long_term_score"].ge(60)
                & combined_candidates["short_term_entry_score"].ge(60)
                & combined_candidates[
                    "analyst_long_term_data_coverage_pct"
                ].ge(80)
                & combined_candidates[
                    "short_term_data_coverage_pct"
                ].ge(80)
            )
        ).sum()
    )

    if outside_range or invalid_long or invalid_short or invalid_combined:
        raise AssertionError(
            "Validation failed: "
            f"{outside_range} scores outside 0-100, "
            f"{invalid_long} invalid long-term insufficient rows, "
            f"{invalid_short} invalid short-term insufficient rows, "
            f"{invalid_combined} invalid combined candidates."
        )

    print("\nValidation passed:")
    print(f"- {len(dual_scores)} stocks processed")
    print(
        "- "
        f"{dual_scores['analyst_long_term_status'].isin([RANKED, PARTIAL_DATA]).sum()} "
        "long-term ranked"
    )
    print(
        "- "
        f"{dual_scores['short_term_status'].isin([RANKED, PARTIAL_DATA]).sum()} "
        "short-term ranked"
    )
    print("- 0 scores outside 0-100")
    print("- 0 invalid combined candidates")


def export_results(
    constituents: pd.DataFrame,
    price_metrics: pd.DataFrame,
    complete_price_history: pd.DataFrame,
    analyst_data: pd.DataFrame,
    dual_scores: pd.DataFrame,
    long_term_ranking: pd.DataFrame,
    short_term_ranking: pd.DataFrame,
    combined_candidates: pd.DataFrame,
    strong_candidates: pd.DataFrame,
    wait_for_entry: pd.DataFrame,
    tactical_candidates: pd.DataFrame,
    momentum_only: pd.DataFrame,
    insufficient_data: pd.DataFrame,
) -> None:

    constituents.to_csv(
        RUN_DIR
        / "sp500_constituents.csv",
        index=False,
    )

    price_metrics.to_csv(
        RUN_DIR
        / "price_metrics.csv",
        index=False,
    )

    complete_price_history.to_csv(
        RUN_DIR
        / "daily_price_history.csv.gz",
        index=False,
        compression="gzip",
    )

    analyst_data.to_csv(
        RUN_DIR
        / "fresh_analyst_data.csv",
        index=False,
    )

    dual_scores.to_csv(
        RUN_DIR
        / "full_dual_score_analysis.csv",
        index=False,
    )

    long_term_ranking.to_csv(
        RUN_DIR
        / "long_term_ranking.csv",
        index=False,
    )

    short_term_ranking.to_csv(
        RUN_DIR
        / "short_term_entry_ranking.csv",
        index=False,
    )

    combined_candidates.to_csv(
        RUN_DIR
        / "combined_candidates.csv",
        index=False,
    )

    insufficient_data.to_csv(
        RUN_DIR
        / "insufficient_data.csv",
        index=False,
    )

    excel_file = (
        RUN_DIR
        / "sp500_dual_score_analysis.xlsx"
    )

    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl",
    ) as writer:

        prepare_dataframe_for_excel(
            combined_candidates
        ).to_excel(
            writer,
            sheet_name="combined_candidates",
            index=False,
        )

        prepare_dataframe_for_excel(
            long_term_ranking
        ).to_excel(
            writer,
            sheet_name="long_term_ranking",
            index=False,
        )

        prepare_dataframe_for_excel(
            short_term_ranking
        ).to_excel(
            writer,
            sheet_name="short_term_ranking",
            index=False,
        )

        prepare_dataframe_for_excel(
            strong_candidates
        ).to_excel(
            writer,
            sheet_name="strong_candidates",
            index=False,
        )

        prepare_dataframe_for_excel(
            wait_for_entry
        ).to_excel(
            writer,
            sheet_name="wait_for_entry",
            index=False,
        )

        prepare_dataframe_for_excel(
            tactical_candidates
        ).to_excel(
            writer,
            sheet_name="tactical_candidates",
            index=False,
        )

        prepare_dataframe_for_excel(
            momentum_only
        ).to_excel(
            writer,
            sheet_name="momentum_only",
            index=False,
        )

        prepare_dataframe_for_excel(
            insufficient_data
        ).to_excel(
            writer,
            sheet_name="insufficient_data",
            index=False,
        )

        prepare_dataframe_for_excel(
            dual_scores
        ).to_excel(
            writer,
            sheet_name="all_stocks",
            index=False,
        )

        build_methodology_table().to_excel(
            writer,
            sheet_name="methodology",
            index=False,
        )

        prepare_dataframe_for_excel(
            build_run_metadata(
                constituents,
                price_metrics,
                analyst_data,
                dual_scores,
            )
        ).to_excel(
            writer,
            sheet_name="run_metadata",
            index=False,
        )

        prepare_dataframe_for_excel(
            constituents
        ).to_excel(
            writer,
            sheet_name="constituents",
            index=False,
        )

    print("\n")
    print("=" * 120)
    print("DOSYALAR KAYDEDİLDİ")
    print("=" * 120)

    print(
        RUN_DIR.resolve()
    )

    print(
        excel_file.resolve()
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 120)
    print("S&P 500 DUAL-SCORE SCANNER")
    print("=" * 120)

    print(
        "Run started:",
        RUN_STARTED_AT_UTC,
    )

    print(
        "Output folder:",
        RUN_DIR.resolve(),
    )

    print(
        "yfinance version:",
        yf.__version__,
    )

    print(
        "Scoring model version:",
        SCORING_MODEL_VERSION,
    )

    print(
        "\n1/7 Güncel S&P 500 listesi indiriliyor..."
    )

    constituents = (
        download_sp500_constituents()
    )

    symbols = (
        constituents["symbol"]
        .tolist()
    )

    print(
        "Constituent count:",
        len(constituents),
    )

    print(
        "\n2/7 Güncel fiyat geçmişi indiriliyor..."
    )

    raw_prices = (
        download_price_history(
            symbols
        )
    )

    print(
        "\n3/7 Fiyat metrikleri hesaplanıyor..."
    )

    (
        price_metrics,
        complete_price_history,
    ) = build_price_tables(
        raw_prices,
        symbols,
    )

    print(
        "\n4/7 Analist verileri indiriliyor..."
    )

    analyst_data = (
        download_analyst_data(
            symbols
        )
    )

    analyst_error_count = int(
        analyst_data[
            "data_errors"
        ]
        .fillna("")
        .ne("")
        .sum()
    )

    print(
        "Analyst rows:",
        len(analyst_data),
    )

    print(
        "Rows with optional/request errors:",
        analyst_error_count,
    )

    print(
        "\n5/7 Ana analiz oluşturuluyor..."
    )

    analysis = build_analysis(
        constituents,
        price_metrics,
        analyst_data,
    )

    print(
        "\n6/7 Kısa vadeli metrikler ve skorlar hesaplanıyor..."
    )

    short_term_metrics = (
        build_short_term_metrics(
            complete_price_history
        )
    )

    dual_scores = (
        calculate_dual_scores(
            analysis,
            short_term_metrics,
        )
    )

    (
        long_term_ranking,
        short_term_ranking,
        combined_candidates,
        strong_candidates,
        wait_for_entry,
        tactical_candidates,
        momentum_only,
        insufficient_data,
    ) = create_rankings(
        dual_scores
    )

    validate_scores(
        dual_scores,
        combined_candidates,
    )

    if EXPORT_RESULTS:
        print(
            "\n7/7 Sonuçlar kaydediliyor..."
        )

        export_results(
            constituents=constituents,
            price_metrics=price_metrics,
            complete_price_history=(
                complete_price_history
            ),
            analyst_data=analyst_data,
            dual_scores=dual_scores,
            long_term_ranking=(
                long_term_ranking
            ),
            short_term_ranking=(
                short_term_ranking
            ),
            combined_candidates=(
                combined_candidates
            ),
            strong_candidates=strong_candidates,
            wait_for_entry=wait_for_entry,
            tactical_candidates=tactical_candidates,
            momentum_only=momentum_only,
            insufficient_data=insufficient_data,
        )

    else:
        print(
            "\n7/7 Dosya çıktısı kapalı; sonuçlar terminale yazdırılıyor..."
        )

    print_rankings(
        long_term_ranking,
        short_term_ranking,
        combined_candidates,
    )

    print_ticker_details(
        dual_scores,
        TICKER_TO_CHECK,
    )

    print("\n")
    print("=" * 120)
    print("TAMAMLANDI")
    print("=" * 120)

    print(
        "Long-term yüksek + short-term düşük: "
        "potansiyel var, giriş zamanı zayıf."
    )

    print(
        "İkisi de yüksek: "
        "araştırma için daha dengeli aday."
    )

    print(
        "Long-term düşük + short-term yüksek: "
        "momentum var, uzun vadeli destek daha zayıf."
    )

    print(
        "İkisi de düşük: "
        "genellikle bekle veya uzak dur."
    )


if __name__ == "__main__":
    main()
