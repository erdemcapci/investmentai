from __future__ import annotations
import pandas as pd
import yfinance as yf
from investment_ai.features.technical import price_features


def download_prices(symbols: list[str], period="2y") -> pd.DataFrame:
    return yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )


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
    return pd.DataFrame(
        [
            {"symbol": s, **price_features(symbol_history(downloaded, s))}
            for s in symbols
        ]
    )
