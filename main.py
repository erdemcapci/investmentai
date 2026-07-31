from __future__ import annotations

import math
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


# ============================================================
# SETTINGS
# ============================================================

# Ayrıntısını terminalde görmek istediğin ticker
TICKER_TO_CHECK = "MU"

# Terminalde gösterilecek maksimum satır
TOP_N = 50

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
LOW_AVERAGE_DOLLAR_VOLUME = 10_000_000

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
        row["current_price"]
    ):
        flags.append(
            "MISSING_CURRENT_PRICE"
        )

    if pd.isna(
        row["selected_target"]
    ):
        flags.append(
            "MISSING_ANALYST_TARGET"
        )

    if row["rating_count"] == 0:
        flags.append(
            "MISSING_RECOMMENDATIONS"
        )

    if (
        row["rating_count"]
        < LOW_ANALYST_COVERAGE_THRESHOLD
    ):
        flags.append(
            "LOW_ANALYST_COVERAGE"
        )

    if (
        pd.notna(
            row["target_dispersion_pct"]
        )
        and row["target_dispersion_pct"]
        > HIGH_TARGET_DISPERSION_THRESHOLD_PCT
    ):
        flags.append(
            "HIGH_TARGET_DISPERSION"
        )

    if (
        pd.notna(
            row["days_to_earnings"]
        )
        and 0
        <= row["days_to_earnings"]
        <= EARNINGS_WARNING_DAYS
    ):
        flags.append(
            "EARNINGS_WITHIN_7_DAYS"
        )

    if (
        pd.notna(
            row["eps_down_30d"]
        )
        and pd.notna(
            row["eps_up_30d"]
        )
        and row["eps_down_30d"]
        > row["eps_up_30d"]
    ):
        flags.append(
            "NEGATIVE_EPS_REVISIONS"
        )

    if (
        pd.notna(
            row["current_price"]
        )
        and pd.notna(
            row["ma_200"]
        )
        and row["current_price"]
        < row["ma_200"]
    ):
        flags.append(
            "BELOW_200D_MA"
        )

    if (
        pd.notna(
            row["volatility_annual_pct"]
        )
        and row["volatility_annual_pct"]
        > HIGH_VOLATILITY_THRESHOLD_PCT
    ):
        flags.append(
            "HIGH_VOLATILITY"
        )

    if (
        pd.notna(
            row["average_dollar_volume_20d"]
        )
        and row[
            "average_dollar_volume_20d"
        ]
        < LOW_AVERAGE_DOLLAR_VOLUME
    ):
        flags.append(
            "LOW_LIQUIDITY"
        )

    critical_data_missing = (
        pd.isna(row["current_price"])
        or pd.isna(
            row["selected_target"]
        )
        or row["rating_count"] == 0
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

    analysis["selected_target"] = (
        analysis["target_median"]
        .combine_first(
            analysis["target_mean"]
        )
    )

    analysis[
        "selected_target_type"
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
                "selected_target"
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
            "selected_target"
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
            "negative_days_last_5": np.nan,
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
        "negative_days_last_5": (
            negative_days_last_5
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

    # --------------------------------------------------------
    # LONG-TERM SCORE
    # --------------------------------------------------------

    # 1. Analyst target upside — 30%
    scores[
        "lt_score_upside"
    ] = linear_score(
        scores[
            "selected_target_upside_pct"
        ],
        minimum=0,
        maximum=60,
    ).fillna(0)

    # 2. Analyst sentiment — 25%
    positive_pct = pd.to_numeric(
        scores[
            "positive_rating_pct"
        ],
        errors="coerce",
    ).fillna(0)

    recommendation_strength = (
        (
            pd.to_numeric(
                scores[
                    "recommendation_score_simple"
                ],
                errors="coerce",
            )
            .clip(
                lower=-1,
                upper=1,
            )
            + 1
        )
        / 2
        * 100
    )

    recommendation_strength = (
        recommendation_strength
        .where(
            rating_count > 0,
            0,
        )
        .fillna(0)
    )

    scores[
        "lt_score_sentiment"
    ] = (
        0.70
        * positive_pct.clip(
            lower=0,
            upper=100,
        )
        + 0.30
        * recommendation_strength
    )

    # 3. Analyst coverage — 10%
    scores[
        "lt_score_coverage"
    ] = (
        rating_count
        .clip(
            lower=0,
            upper=40,
        )
        / 40
        * 100
    )

    # 4. Target agreement — 15%
    target_dispersion = pd.to_numeric(
        scores[
            "target_dispersion_pct"
        ],
        errors="coerce",
    )

    scores[
        "lt_score_target_agreement"
    ] = (
        100
        - target_dispersion.clip(
            lower=0,
            upper=100,
        )
    ).fillna(35)

    # 5. EPS revisions — 15%
    eps_up = pd.to_numeric(
        scores["eps_up_30d"],
        errors="coerce",
    )

    eps_down = pd.to_numeric(
        scores["eps_down_30d"],
        errors="coerce",
    )

    eps_missing = (
        eps_up.isna()
        & eps_down.isna()
    )

    eps_up_filled = (
        eps_up.fillna(0)
    )

    eps_down_filled = (
        eps_down.fillna(0)
    )

    total_eps_revisions = (
        eps_up_filled
        + eps_down_filled
    )

    scores[
        "lt_score_eps_revisions"
    ] = np.where(
        eps_missing,
        40,
        np.where(
            total_eps_revisions > 0,
            eps_up_filled
            / total_eps_revisions
            * 100,
            50,
        ),
    )

    # 6. 200-day trend — 5%
    price_vs_200 = pd.to_numeric(
        scores[
            "price_vs_200d_ma_pct"
        ],
        errors="coerce",
    )

    scores[
        "lt_score_long_trend"
    ] = np.select(
        [
            price_vs_200 < -30,

            (
                price_vs_200 >= -30
            )
            & (
                price_vs_200 <= 20
            ),

            (
                price_vs_200 > 20
            )
            & (
                price_vs_200 <= 50
            ),

            price_vs_200 > 50,
        ],
        [
            0,

            (
                price_vs_200 + 30
            )
            / 50
            * 100,

            (
                100
                - (
                    price_vs_200 - 20
                )
                / 30
                * 30
            ),

            60,
        ],
        default=50,
    )

    scores[
        "long_term_base_score"
    ] = (
        0.30
        * scores[
            "lt_score_upside"
        ]
        + 0.25
        * scores[
            "lt_score_sentiment"
        ]
        + 0.10
        * scores[
            "lt_score_coverage"
        ]
        + 0.15
        * scores[
            "lt_score_target_agreement"
        ]
        + 0.15
        * scores[
            "lt_score_eps_revisions"
        ]
        + 0.05
        * scores[
            "lt_score_long_trend"
        ]
    )

    risk_text = (
        scores["risk_flags"]
        .fillna("")
        .astype(str)
    )

    scores[
        "long_term_risk_penalty"
    ] = 0.0

    scores[
        "long_term_risk_penalty"
    ] += np.where(
        risk_text.str.contains(
            "LOW_LIQUIDITY",
            regex=False,
        ),
        10,
        0,
    )

    scores[
        "long_term_risk_penalty"
    ] += np.where(
        risk_text.str.contains(
            "HIGH_VOLATILITY",
            regex=False,
        ),
        2,
        0,
    )

    scores["long_term_score"] = (
        scores[
            "long_term_base_score"
        ]
        - scores[
            "long_term_risk_penalty"
        ]
    ).clip(
        lower=0,
        upper=100,
    )

    long_term_missing = (
        scores["current_price"].isna()
        | scores[
            "selected_target"
        ].isna()
        | rating_count.lt(5)
    )

    scores.loc[
        long_term_missing,
        "long_term_score",
    ] = 0

    # --------------------------------------------------------
    # SHORT-TERM ENTRY SCORE
    # --------------------------------------------------------

    # 1. 5-day and 20-day momentum — 25%
    score_5d_momentum = linear_score(
        scores["return_5d_pct"],
        minimum=-15,
        maximum=8,
    )

    score_20d_momentum = linear_score(
        scores["return_20d_pct"],
        minimum=-25,
        maximum=15,
    )

    scores[
        "st_score_momentum"
    ] = (
        0.65
        * score_5d_momentum
        + 0.35
        * score_20d_momentum
    ).fillna(50)

    # 2. Moving average alignment — 20%
    score_ma20 = (
        scores[
            "distance_from_ma20_pct"
        ]
        .apply(
            ma_distance_score
        )
    )

    score_ma50 = (
        scores[
            "distance_from_ma50_pct"
        ]
        .apply(
            ma_distance_score
        )
    )

    scores[
        "st_score_ma_alignment"
    ] = (
        0.65
        * score_ma20
        + 0.35
        * score_ma50
    )

    # 3. Recent selloff stability — 20%
    score_two_day_stability = (
        linear_score(
            scores["return_2d_pct"],
            minimum=-15,
            maximum=2,
        )
    )

    negative_days_score = (
        100
        - (
            pd.to_numeric(
                scores[
                    "negative_days_last_5"
                ],
                errors="coerce",
            )
            .clip(
                lower=0,
                upper=5,
            )
            / 5
            * 100
        )
    )

    scores[
        "st_score_selloff_stability"
    ] = (
        0.75
        * score_two_day_stability
        + 0.25
        * negative_days_score
    ).fillna(50)

    # 4. Pullback quality — 15%
    scores[
        "st_score_pullback_quality"
    ] = (
        scores[
            "drawdown_from_20d_high_pct"
        ]
        .apply(
            calculate_pullback_score
        )
        .fillna(50)
    )

    # 5. Volatility — 15%
    short_volatility = pd.to_numeric(
        scores[
            "short_volatility_20d_pct"
        ],
        errors="coerce",
    )

    scores[
        "st_score_volatility"
    ] = (
        (
            75
            - short_volatility
        )
        / (
            75 - 20
        )
        * 100
    ).clip(
        lower=0,
        upper=100,
    ).fillna(40)

    # 6. Earnings timing — 5%
    scores[
        "st_score_earnings_timing"
    ] = (
        scores[
            "days_to_earnings"
        ]
        .apply(
            earnings_timing_score
        )
    )

    scores[
        "short_term_entry_score"
    ] = (
        0.25
        * scores[
            "st_score_momentum"
        ]
        + 0.20
        * scores[
            "st_score_ma_alignment"
        ]
        + 0.20
        * scores[
            "st_score_selloff_stability"
        ]
        + 0.15
        * scores[
            "st_score_pullback_quality"
        ]
        + 0.15
        * scores[
            "st_score_volatility"
        ]
        + 0.05
        * scores[
            "st_score_earnings_timing"
        ]
    )

    short_term_missing = (
        scores[
            "return_5d_pct"
        ].isna()
        | scores[
            "return_20d_pct"
        ].isna()
        | scores[
            "distance_from_ma20_pct"
        ].isna()
        | scores[
            "short_volatility_20d_pct"
        ].isna()
    )

    scores.loc[
        short_term_missing,
        "short_term_entry_score",
    ] = 0

    scores[
        "short_term_entry_score"
    ] -= np.where(
        risk_text.str.contains(
            "LOW_LIQUIDITY",
            regex=False,
        ),
        15,
        0,
    )

    scores[
        "short_term_entry_score"
    ] = (
        scores[
            "short_term_entry_score"
        ]
        .clip(
            lower=0,
            upper=100,
        )
    )

    # --------------------------------------------------------
    # CATEGORIES AND RANKS
    # --------------------------------------------------------

    scores[
        "long_term_category"
    ] = pd.cut(
        scores["long_term_score"],
        bins=[
            -0.01,
            44.99,
            59.99,
            74.99,
            100,
        ],
        labels=[
            "WEAK",
            "NEUTRAL",
            "POSITIVE",
            "STRONG",
        ],
    )

    scores[
        "short_term_category"
    ] = pd.cut(
        scores[
            "short_term_entry_score"
        ],
        bins=[
            -0.01,
            39.99,
            59.99,
            74.99,
            100,
        ],
        labels=[
            "WAIT / AVOID",
            "NEUTRAL",
            "WATCH",
            "FAVORABLE ENTRY",
        ],
    )

    scores[
        "long_term_rank"
    ] = (
        scores["long_term_score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    scores[
        "short_term_rank"
    ] = (
        scores[
            "short_term_entry_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

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
]:

    long_term_ranking = (
        dual_scores
        .sort_values(
            [
                "long_term_score",
                "short_term_entry_score",
                "positive_rating_pct",
                "rating_count",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    short_term_ranking = (
        dual_scores
        .sort_values(
            [
                "short_term_entry_score",
                "long_term_score",
                "positive_rating_pct",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    combined_candidates = (
        dual_scores.loc[
            dual_scores[
                "long_term_score"
            ].ge(60)
            & dual_scores[
                "short_term_entry_score"
            ].ge(60)
            & ~dual_scores[
                "has_partial_data"
            ]
        ]
        .copy()
    )

    combined_candidates[
        "combined_score"
    ] = (
        0.60
        * combined_candidates[
            "long_term_score"
        ]
        + 0.40
        * combined_candidates[
            "short_term_entry_score"
        ]
    )

    combined_candidates = (
        combined_candidates
        .sort_values(
            [
                "combined_score",
                "long_term_score",
                "short_term_entry_score",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    combined_candidates.insert(
        0,
        "combined_rank",
        range(
            1,
            len(
                combined_candidates
            ) + 1,
        ),
    )

    return (
        long_term_ranking,
        short_term_ranking,
        combined_candidates,
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
        "long_term_score",
        "long_term_category",
        "short_term_entry_score",
        "short_term_category",
        "current_price",
        "selected_target_upside_pct",
        "positive_rating_pct",
        "rating_count",
        "target_dispersion_pct",
        "return_5d_pct",
        "risk_flags",
    ]

    short_columns = [
        "short_term_rank",
        "long_term_rank",
        "symbol",
        "company_name",
        "short_term_entry_score",
        "short_term_category",
        "long_term_score",
        "long_term_category",
        "current_price",
        "return_1d_pct",
        "return_2d_pct",
        "return_5d_pct",
        "return_20d_pct",
        "drawdown_from_20d_high_pct",
        "short_volatility_20d_pct",
        "risk_flags",
    ]

    combined_columns = [
        "combined_rank",
        "symbol",
        "company_name",
        "combined_score",
        "long_term_score",
        "short_term_entry_score",
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
    print("UZUN VADELİ POTANSİYEL SIRALAMASI")
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

    print(
        f"{row['symbol']} — "
        f"{row['company_name']}"
    )

    print(
        f"Uzun vade: "
        f"rank {row['long_term_rank']}"
        f"/{len(dual_scores)}, "
        f"score {row['long_term_score']:.2f}, "
        f"{row['long_term_category']}"
    )

    print(
        f"Kısa vade: "
        f"rank {row['short_term_rank']}"
        f"/{len(dual_scores)}, "
        f"score "
        f"{row['short_term_entry_score']:.2f}, "
        f"{row['short_term_category']}"
    )

    print(
        f"Current price: "
        f"{row['current_price']:,.2f}"
    )

    print(
        f"Selected target: "
        f"{row['selected_target']:,.2f}"
    )

    print(
        f"Target upside: "
        f"{row['selected_target_upside_pct']:.2f}%"
    )

    print(
        f"Positive ratings: "
        f"{row['positive_rating_pct']:.2f}% "
        f"from "
        f"{int(row['rating_count'])} ratings"
    )

    print(
        f"Risk flags: "
        f"{row['risk_flags'] or 'None'}"
    )

    detail_columns = [
        "symbol",
        "company_name",
        "sector",
        "long_term_rank",
        "long_term_score",
        "long_term_category",
        "short_term_rank",
        "short_term_entry_score",
        "short_term_category",
        "current_price",
        "current_price_source",
        "price_as_of",
        "target_low",
        "target_mean",
        "target_median",
        "target_high",
        "selected_target",
        "selected_target_upside_pct",
        "strong_buy",
        "buy",
        "hold",
        "sell",
        "strong_sell",
        "positive_rating_pct",
        "negative_rating_pct",
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
        "lt_score_upside",
        "lt_score_sentiment",
        "lt_score_coverage",
        "lt_score_target_agreement",
        "lt_score_eps_revisions",
        "lt_score_long_trend",
        "long_term_base_score",
        "long_term_risk_penalty",
        "st_score_momentum",
        "st_score_ma_alignment",
        "st_score_selloff_stability",
        "st_score_pullback_quality",
        "st_score_volatility",
        "st_score_earnings_timing",
        "risk_flags",
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

    return excel_frame


def export_results(
    constituents: pd.DataFrame,
    price_metrics: pd.DataFrame,
    complete_price_history: pd.DataFrame,
    analyst_data: pd.DataFrame,
    dual_scores: pd.DataFrame,
    long_term_ranking: pd.DataFrame,
    short_term_ranking: pd.DataFrame,
    combined_candidates: pd.DataFrame,
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

    excel_file = (
        RUN_DIR
        / "sp500_dual_score_analysis.xlsx"
    )

    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl",
    ) as writer:

        prepare_dataframe_for_excel(
            long_term_ranking
        ).to_excel(
            writer,
            sheet_name="Long Term Ranking",
            index=False,
        )

        prepare_dataframe_for_excel(
            short_term_ranking
        ).to_excel(
            writer,
            sheet_name="Short Term Entry",
            index=False,
        )

        prepare_dataframe_for_excel(
            combined_candidates
        ).to_excel(
            writer,
            sheet_name="Combined Candidates",
            index=False,
        )

        prepare_dataframe_for_excel(
            dual_scores
        ).to_excel(
            writer,
            sheet_name="Full Analysis",
            index=False,
        )

        prepare_dataframe_for_excel(
            analyst_data
        ).to_excel(
            writer,
            sheet_name="Analyst Data",
            index=False,
        )

        prepare_dataframe_for_excel(
            constituents
        ).to_excel(
            writer,
            sheet_name="Constituents",
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
    ) = create_rankings(
        dual_scores
    )

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
