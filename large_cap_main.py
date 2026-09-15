"""Investment AI large-cap runner.

This is the expanded entry point for Investment AI.  It replaces S&P 500
membership as the universe gate with a broad NYSE/Nasdaq large-cap screen,
selects a neutral Top-100 analysis universe based on analysability, and then
reuses the established analyst-first long/short scoring pipeline.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm

import main as scanner
from universe import (
    UniverseConfig,
    analyst_coverage_confidence,
    annotate_sp500_membership,
    fetch_large_cap_universe,
    parse_rating_activity,
    select_analysis_universe,
)


ANALYSIS_UNIVERSE_SIZE = int(os.getenv("ANALYSIS_UNIVERSE_SIZE", "100"))
UNIVERSE_PREFETCH_LIMIT = int(os.getenv("UNIVERSE_PREFETCH_LIMIT", "500"))
UNIVERSE_MIN_MARKET_CAP = float(os.getenv("UNIVERSE_MIN_MARKET_CAP", "10000000000"))
UNIVERSE_MIN_DOLLAR_VOLUME = float(
    os.getenv("UNIVERSE_MIN_DOLLAR_VOLUME", "20000000")
)
UNIVERSE_MIN_ANALYST_COUNT = int(os.getenv("UNIVERSE_MIN_ANALYST_COUNT", "8"))
UNIVERSE_MIN_DATA_QUALITY = float(os.getenv("UNIVERSE_MIN_DATA_QUALITY", "60"))

UNIVERSE_CONFIG = UniverseConfig(
    min_market_cap=UNIVERSE_MIN_MARKET_CAP,
    min_average_dollar_volume=UNIVERSE_MIN_DOLLAR_VOLUME,
    min_analyst_count=UNIVERSE_MIN_ANALYST_COUNT,
    min_data_quality_score=UNIVERSE_MIN_DATA_QUALITY,
    analysis_universe_size=ANALYSIS_UNIVERSE_SIZE,
)

# The selected universe already requires >=8 analysts, so 8 is the relevant
# low-coverage warning boundary for this runner.  The downstream scoring
# function resolves this global at runtime.
scanner.LOW_ANALYST_COVERAGE_THRESHOLD = UNIVERSE_MIN_ANALYST_COUNT
scanner.calculate_coverage_confidence = analyst_coverage_confidence

RUN_ROOT = Path("large_cap_fresh_runs")
RUN_DIR = RUN_ROOT / scanner.RUN_ID


# ---------------------------------------------------------------------------
# Neutral prefilter
# ---------------------------------------------------------------------------
def neutral_prefilter(
    universe: pd.DataFrame,
    limit: int = UNIVERSE_PREFETCH_LIMIT,
) -> pd.DataFrame:
    """Limit expensive analyst calls without leaking bullish signals.

    Only market cap and three-month dollar liquidity from the broad Yahoo
    screener are used.  Target upside, Buy/Sell mix, EPS direction, momentum,
    and technical signals are deliberately unavailable at this stage.
    """
    frame = universe.copy()
    market_cap = pd.to_numeric(frame["market_cap"], errors="coerce")
    liquidity = pd.to_numeric(
        frame["screener_average_dollar_volume_3m"], errors="coerce"
    )
    frame = frame.loc[
        market_cap.ge(UNIVERSE_MIN_MARKET_CAP)
        & liquidity.ge(UNIVERSE_MIN_DOLLAR_VOLUME)
    ].copy()
    if frame.empty:
        raise RuntimeError("No stocks passed the market-cap/liquidity prefilter.")

    def pct_score(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").clip(lower=0)
        return np.log1p(numeric).rank(method="average", pct=True).mul(100)

    frame["prefilter_market_cap_score"] = pct_score(frame["market_cap"])
    frame["prefilter_liquidity_score"] = pct_score(
        frame["screener_average_dollar_volume_3m"]
    )
    frame["neutral_prefilter_score"] = (
        0.60 * frame["prefilter_market_cap_score"]
        + 0.40 * frame["prefilter_liquidity_score"]
    )
    frame = frame.sort_values(
        ["neutral_prefilter_score", "market_cap", "symbol"],
        ascending=[False, False, True],
    )
    return frame.head(max(limit, ANALYSIS_UNIVERSE_SIZE)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analyst activity freshness
# ---------------------------------------------------------------------------
def download_one_rating_activity(symbol: str) -> dict[str, Any]:
    ticker = yf.Ticker(symbol)
    raw_actions, error = scanner.request_with_retry(ticker.get_upgrades_downgrades)
    metrics = parse_rating_activity(raw_actions)
    return {
        "symbol": symbol,
        **metrics,
        "rating_activity_error": error,
    }


def download_rating_activity(symbols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=scanner.MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_one_rating_activity, symbol): symbol
            for symbol in symbols
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Analist aktivite güncelliği indiriliyor",
        ):
            symbol = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "symbol": symbol,
                        "rating_actions_30d": np.nan,
                        "rating_actions_90d": np.nan,
                        "latest_rating_date": pd.NaT,
                        "analyst_activity_freshness_score": 40.0,
                        "rating_activity_error": (
                            f"UNHANDLED={type(exc).__name__}: {exc}"
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def enrich_analyst_data_with_activity(
    analyst_data: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    activity = download_rating_activity(symbols)
    return analyst_data.merge(activity, on="symbol", how="left", validate="one_to_one")


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def make_selection_price_stub(prefiltered: pd.DataFrame) -> pd.DataFrame:
    """Provide neutral liquidity/price fields required by universe selection.

    Full two-year price history is intentionally downloaded only after Top-100
    selection.  That keeps the expansion roughly in the same request-cost class
    as the old S&P 500 scanner instead of downloading history for the entire
    broad market first.
    """
    frame = prefiltered[
        [
            "symbol",
            "screener_price",
            "screener_average_volume_3m",
            "screener_average_dollar_volume_3m",
        ]
    ].copy()
    frame = frame.rename(
        columns={
            "screener_price": "history_price",
            "screener_average_volume_3m": "average_volume_20d",
            "screener_average_dollar_volume_3m": "average_dollar_volume_20d",
        }
    )
    frame["price_as_of"] = pd.NaT
    frame["ma_20"] = np.nan
    frame["ma_50"] = np.nan
    frame["ma_200"] = np.nan
    frame["price_vs_200d_ma_pct"] = np.nan
    frame["volatility_annual_pct"] = np.nan
    frame["history_error"] = "PREFILTER_SCREENER_ONLY"
    return frame


def build_analysis_universe() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    broad = fetch_large_cap_universe(UNIVERSE_CONFIG)

    try:
        sp500 = scanner.download_sp500_constituents()
    except Exception as exc:
        print(f"S&P 500 membership metadata unavailable: {type(exc).__name__}: {exc}")
        sp500 = pd.DataFrame(columns=["symbol"])

    broad = annotate_sp500_membership(broad, sp500)
    prefiltered = neutral_prefilter(broad)
    prefetch_symbols = prefiltered["symbol"].tolist()

    analyst_data = scanner.download_analyst_data(prefetch_symbols)
    analyst_data = enrich_analyst_data_with_activity(analyst_data, prefetch_symbols)

    price_stub = make_selection_price_stub(prefiltered)
    diagnostics, eligible, selected = select_analysis_universe(
        prefiltered,
        price_stub,
        analyst_data,
        UNIVERSE_CONFIG,
    )
    selected_symbols = selected["symbol"].tolist()
    analyst_selected = analyst_data.loc[
        analyst_data["symbol"].isin(selected_symbols)
    ].copy()
    analyst_selected = analyst_selected.set_index("symbol").loc[selected_symbols].reset_index()
    return diagnostics, eligible, selected, analyst_selected


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_large_cap_results(
    diagnostics: pd.DataFrame,
    eligible: pd.DataFrame,
    selected: pd.DataFrame,
    analyst_data: pd.DataFrame,
    price_metrics: pd.DataFrame,
    price_history: pd.DataFrame,
    dual_scores: pd.DataFrame,
    long_term_ranking: pd.DataFrame,
    short_term_ranking: pd.DataFrame,
    combined_candidates: pd.DataFrame,
) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(RUN_DIR / "universe_diagnostics.csv", index=False)
    eligible.to_csv(RUN_DIR / "eligible_large_cap_universe.csv", index=False)
    selected.to_csv(RUN_DIR / "analysis_universe_top100.csv", index=False)
    analyst_data.to_csv(RUN_DIR / "fresh_analyst_data.csv", index=False)
    price_metrics.to_csv(RUN_DIR / "price_metrics.csv", index=False)
    price_history.to_csv(
        RUN_DIR / "daily_price_history.csv.gz",
        index=False,
        compression="gzip",
    )
    dual_scores.to_csv(RUN_DIR / "full_dual_score_analysis.csv", index=False)
    long_term_ranking.to_csv(RUN_DIR / "long_term_ranking.csv", index=False)
    short_term_ranking.to_csv(RUN_DIR / "short_term_entry_ranking.csv", index=False)
    combined_candidates.to_csv(RUN_DIR / "combined_candidates.csv", index=False)

    excel_file = RUN_DIR / "large_cap_dual_score_analysis.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        scanner.prepare_dataframe_for_excel(combined_candidates).to_excel(
            writer, sheet_name="combined_candidates", index=False
        )
        scanner.prepare_dataframe_for_excel(long_term_ranking).to_excel(
            writer, sheet_name="long_term_ranking", index=False
        )
        scanner.prepare_dataframe_for_excel(short_term_ranking).to_excel(
            writer, sheet_name="short_term_ranking", index=False
        )
        scanner.prepare_dataframe_for_excel(dual_scores).to_excel(
            writer, sheet_name="all_top100", index=False
        )
        scanner.prepare_dataframe_for_excel(selected).to_excel(
            writer, sheet_name="analysis_universe", index=False
        )
        scanner.prepare_dataframe_for_excel(eligible).to_excel(
            writer, sheet_name="eligible_universe", index=False
        )
        scanner.prepare_dataframe_for_excel(diagnostics).to_excel(
            writer, sheet_name="universe_diagnostics", index=False
        )
        scanner.build_methodology_table().to_excel(
            writer, sheet_name="scoring_methodology", index=False
        )

    print(f"\nLarge-cap outputs saved to: {RUN_DIR.resolve()}")
    print(excel_file.resolve())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 120)
    print("INVESTMENT AI — LARGE-CAP TOP-100 DUAL-SCORE SCANNER")
    print("=" * 120)
    print("Run started:", scanner.RUN_STARTED_AT_UTC)
    print("yfinance version:", yf.__version__)
    print(
        "Universe rules: "
        f"NYSE/Nasdaq, market cap >= ${UNIVERSE_MIN_MARKET_CAP/1e9:.0f}B, "
        f"dollar liquidity >= ${UNIVERSE_MIN_DOLLAR_VOLUME/1e6:.0f}M, "
        f"analysts >= {UNIVERSE_MIN_ANALYST_COUNT}; "
        f"select Top {ANALYSIS_UNIVERSE_SIZE}."
    )

    print("\n1/8 Geniş NYSE/Nasdaq large-cap evreni ve nötr prefilter oluşturuluyor...")
    diagnostics, eligible, selected, analyst_data = build_analysis_universe()
    symbols = selected["symbol"].tolist()
    print("Broad/prefilter diagnostics:", len(diagnostics))
    print("Eligible after analyst/data filters:", len(eligible))
    print("Selected analysis universe:", len(selected))
    print("S&P 500 members in selected Top-100:", int(selected["is_sp500"].sum()))
    print("Non-S&P 500 members in selected Top-100:", int((~selected["is_sp500"]).sum()))

    if len(selected) < ANALYSIS_UNIVERSE_SIZE:
        print(
            f"WARNING: only {len(selected)} stocks passed all eligibility rules; "
            f"requested Top {ANALYSIS_UNIVERSE_SIZE}."
        )

    print("\n2/8 Seçilen Top-100 için iki yıllık fiyat geçmişi indiriliyor...")
    raw_prices = scanner.download_price_history(symbols)

    print("\n3/8 Fiyat metrikleri hesaplanıyor...")
    price_metrics, complete_price_history = scanner.build_price_tables(
        raw_prices, symbols
    )

    print("\n4/8 Ana analyst-first analiz oluşturuluyor...")
    analysis = scanner.build_analysis(selected, price_metrics, analyst_data)

    print("\n5/8 Kısa vadeli metrikler hesaplanıyor...")
    short_term_metrics = scanner.build_short_term_metrics(complete_price_history)

    print("\n6/8 Uzun/kısa vade skorları ve sıralamalar hesaplanıyor...")
    dual_scores = scanner.calculate_dual_scores(analysis, short_term_metrics)
    (
        long_term_ranking,
        short_term_ranking,
        combined_candidates,
        _strong_candidates,
        _wait_for_entry,
        _tactical_candidates,
        _momentum_only,
        _insufficient_data,
    ) = scanner.create_rankings(dual_scores)
    scanner.validate_scores(dual_scores, combined_candidates)

    print("\n7/8 Sonuçlar hazırlanıyor...")
    if scanner.EXPORT_RESULTS:
        export_large_cap_results(
            diagnostics=diagnostics,
            eligible=eligible,
            selected=selected,
            analyst_data=analyst_data,
            price_metrics=price_metrics,
            price_history=complete_price_history,
            dual_scores=dual_scores,
            long_term_ranking=long_term_ranking,
            short_term_ranking=short_term_ranking,
            combined_candidates=combined_candidates,
        )
    else:
        print("Dosya çıktısı kapalı; EXPORT_RESULTS=true ile açabilirsin.")

    print("\n8/8 Top-100 sıralamaları yazdırılıyor...")
    scanner.print_rankings(
        long_term_ranking,
        short_term_ranking,
        combined_candidates,
    )
    scanner.print_ticker_details(dual_scores, scanner.TICKER_TO_CHECK)

    print("\n" + "=" * 120)
    print("TAMAMLANDI")
    print("=" * 120)
    print(
        "Universe selection bullish sinyal kullanmaz; Buy/Strong Buy oranı, "
        "target upside ve teknik sinyaller ancak Top-100 seçildikten sonra "
        "Investment AI skorlamasına girer."
    )


if __name__ == "__main__":
    main()
