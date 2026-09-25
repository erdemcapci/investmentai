from __future__ import annotations
import time
import pandas as pd
import yfinance as yf
from investment_ai.config import PRICE_BATCH_SIZE, PRICE_DOWNLOAD_ATTEMPTS
from investment_ai.features.technical import price_features


def _download_chunk(symbols: list[str], period: str) -> pd.DataFrame:
    error = None
    for attempt in range(PRICE_DOWNLOAD_ATTEMPTS):
        try:
            value = yf.download(tickers=symbols, period=period, interval="1d",
                                auto_adjust=True, group_by="ticker", threads=True, progress=False)
            if isinstance(value, pd.DataFrame) and not value.empty:
                if len(symbols) == 1 and not isinstance(value.columns, pd.MultiIndex):
                    value = pd.concat({symbols[0]: value}, axis=1)
                return value
            raise ValueError("empty price response")
        except Exception as exc:
            error = exc
            if attempt + 1 < PRICE_DOWNLOAD_ATTEMPTS:
                time.sleep(.25 * (2 ** attempt))
    return pd.DataFrame()


def download_prices(symbols: list[str], period="2y") -> pd.DataFrame:
    """Download in bounded batches, retry chunks, then retry omissions in small batches."""
    chunks = [_download_chunk(symbols[i:i + PRICE_BATCH_SIZE], period)
              for i in range(0, len(symbols), PRICE_BATCH_SIZE)]
    successful = [chunk for chunk in chunks if not chunk.empty]
    combined = pd.concat(successful, axis=1) if successful else pd.DataFrame()
    present = set(combined.columns.get_level_values(0)) if isinstance(combined.columns, pd.MultiIndex) else set()
    missing = [symbol for symbol in symbols if symbol not in present]
    retries = [_download_chunk(missing[i:i + 25], period) for i in range(0, len(missing), 25)]
    successful_retries = [chunk for chunk in retries if not chunk.empty]
    if successful_retries:
        combined = pd.concat([combined, *successful_retries], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
    combined.attrs["requested_symbol_count"] = len(symbols)
    return combined


def symbol_history(downloaded: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if isinstance(downloaded.columns, pd.MultiIndex):
        if symbol in downloaded.columns.get_level_values(0):
            return downloaded[symbol].dropna(how="all")
        if symbol in downloaded.columns.get_level_values(1):
            return downloaded.xs(symbol, axis=1, level=1).dropna(how="all")
    return (
        downloaded
        if len(set(downloaded.columns) & {"Close", "Volume"})
        else pd.DataFrame()
    )


def build_price_features(downloaded: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(
        [
            {"symbol": s, **price_features(symbol_history(downloaded, s))}
            for s in symbols
        ]
    )
    present = result.current_price.notna().sum() if "current_price" in result else 0
    result["price_symbol_coverage_pct"] = present / len(symbols) * 100 if symbols else 0
    result["missing_price_symbol_count"] = len(symbols) - present
    return result
