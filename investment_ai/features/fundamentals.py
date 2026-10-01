"""Provider-tolerant financial-statement parsing and fundamental features."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from investment_ai.scoring.common import change_pct, number, ratio

FINANCIAL_SECTORS = {"financials", "financial services", "banks", "insurance"}
ALIASES = {
    "revenue": ["TotalRevenue", "OperatingRevenue"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncome", "EBIT"],
    "net_income": ["NetIncome", "NetIncomeCommonStockholders"],
    "ebit": ["EBIT", "OperatingIncome"],
    "ebitda": ["EBITDA", "NormalizedEBITDA"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonOperating"],
    "tax_provision": ["TaxProvision"],
    "pretax_income": ["PretaxIncome"],
    "operating_cash_flow": ["OperatingCashFlow", "TotalCashFromOperatingActivities"],
    "capital_expenditure": ["CapitalExpenditure", "CapitalExpenditures"],
    "cash": ["CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalents"],
    "total_debt": ["TotalDebt"],
    "equity": ["StockholdersEquity", "TotalEquityGrossMinorityInterest"],
    "assets": ["TotalAssets"],
}


def normalize_statement_label(value: Any) -> str:
    """Normalize pretty, snake-case and raw CamelCase Yahoo statement labels."""
    return re.sub(r"[\s_\-/]+", "", str(value)).lower()


def is_financial(sector: Any) -> bool:
    text = str(sector).strip().lower()
    return text in FINANCIAL_SECTORS or "bank" in text or "insurance" in text


def _ordered_frame(frame: Any) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    date_like = all(
        isinstance(column, (pd.Timestamp, np.datetime64, str))
        for column in result.columns
    )
    parsed = (
        pd.to_datetime(result.columns, errors="coerce", utc=True)
        if date_like
        else pd.DatetimeIndex([])
    )
    if len(parsed) and parsed.notna().all():
        order = np.argsort(parsed.view("int64"))[::-1]
        result = result.iloc[:, order]
    return result


def _series(frame: Any, names: list[str]) -> pd.Series:
    working = _ordered_frame(frame)
    if working.empty:
        return pd.Series(dtype=float)
    lookup = {normalize_statement_label(label): label for label in working.index}
    for name in names:
        label = lookup.get(normalize_statement_label(name))
        if label is not None:
            return pd.to_numeric(working.loc[label], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _free_cash_flow(operating_cash_flow: Any, capex: Any) -> float:
    operating_cash_flow, capex = number(operating_cash_flow), number(capex)
    if pd.isna(operating_cash_flow) or pd.isna(capex):
        return np.nan
    return operating_cash_flow + capex if capex < 0 else operating_cash_flow - capex


def derive_fundamentals(
    income: Any, balance: Any, cashflow: Any, sector: Any = ""
) -> dict[str, Any]:
    """Derive annual features after explicit newest-to-oldest date ordering."""
    income_keys = {
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "ebit",
        "ebitda",
        "interest_expense",
        "tax_provision",
        "pretax_income",
    }
    cashflow_keys = {"operating_cash_flow", "capital_expenditure"}
    series = {
        key: _series(
            (
                income
                if key in income_keys
                else cashflow if key in cashflow_keys else balance
            ),
            aliases,
        )
        for key, aliases in ALIASES.items()
    }
    current = {
        key: (values.iloc[0] if len(values) else np.nan)
        for key, values in series.items()
    }
    previous = {
        key: (values.iloc[1] if len(values) > 1 else np.nan)
        for key, values in series.items()
    }
    fcf = _free_cash_flow(
        current["operating_cash_flow"], current["capital_expenditure"]
    )
    previous_fcf = _free_cash_flow(
        previous["operating_cash_flow"], previous["capital_expenditure"]
    )
    financial = is_financial(sector)
    equity = number(current["equity"])
    ebitda = number(current["ebitda"])
    ebit = number(current["ebit"])
    negative_equity = pd.notna(equity) and equity <= 0
    negative_ebitda = pd.notna(ebitda) and ebitda <= 0
    negative_operating_profit = pd.notna(ebit) and ebit <= 0
    invested_capital = current["total_debt"] + current["equity"] - current["cash"]
    tax_rate = ratio(current["tax_provision"], current["pretax_income"])
    tax_rate = np.clip(tax_rate, 0, 0.35) if pd.notna(tax_rate) else np.nan
    revenue = number(current["revenue"])
    previous_revenue = number(previous["revenue"])
    prior_growth = np.nan
    revenues = series["revenue"]
    if len(revenues) >= 3 and revenues.iloc[2] > 0:
        prior_growth = change_pct(revenues.iloc[1], revenues.iloc[2])
    revenue_growth = (
        change_pct(revenue, previous_revenue) if previous_revenue > 0 else np.nan
    )
    result = {
        **current,
        "free_cash_flow": fcf,
        "gross_margin_pct": (
            ratio(current["gross_profit"], revenue) * 100 if revenue > 0 else np.nan
        ),
        "operating_margin_pct": (
            ratio(current["operating_income"], revenue) * 100 if revenue > 0 else np.nan
        ),
        "net_margin_pct": (
            ratio(current["net_income"], revenue) * 100 if revenue > 0 else np.nan
        ),
        "operating_margin_change_1y": (
            (
                ratio(current["operating_income"], revenue)
                - ratio(previous["operating_income"], previous_revenue)
            )
            * 100
            if revenue > 0 and previous_revenue > 0
            else np.nan
        ),
        "revenue_growth_yoy": revenue_growth,
        "growth_acceleration": (
            revenue_growth - prior_growth
            if pd.notna(revenue_growth) and pd.notna(prior_growth)
            else np.nan
        ),
        "fcf_margin_pct": (
            ratio(fcf, revenue) * 100 if pd.notna(fcf) and revenue > 0 else np.nan
        ),
        "fcf_growth_yoy": (
            change_pct(fcf, previous_fcf)
            if pd.notna(previous_fcf) and abs(previous_fcf) > 1e-9
            else np.nan
        ),
        "cash_conversion": (
            ratio(fcf, current["net_income"])
            if number(current["net_income"]) > 0
            else np.nan
        ),
        "net_debt": current["total_debt"] - current["cash"],
        "debt_to_equity": np.nan if negative_equity else ratio(current["total_debt"], equity),
        "net_debt_to_ebitda": (
            np.nan
            if financial or negative_ebitda or pd.isna(ebitda)
            else ratio(current["total_debt"] - current["cash"], ebitda)
        ),
        "interest_coverage": np.nan if negative_operating_profit else ratio(ebit, abs(current["interest_expense"])),
        "return_on_equity": np.nan if negative_equity else ratio(current["net_income"], equity) * 100,
        "return_on_assets": ratio(current["net_income"], current["assets"]) * 100,
        "earnings_growth_yoy": change_pct(
            current["net_income"], previous["net_income"]
        ),
        "roic_applicable": not financial,
        "roic": (
            np.nan
            if financial
            or pd.isna(invested_capital)
            or invested_capital <= 0
            or pd.isna(tax_rate)
            else current["ebit"] * (1 - tax_rate) / invested_capital * 100
        ),
        "is_financial": financial,
        "negative_equity_flag": bool(negative_equity),
        "negative_ebitda_flag": bool(negative_ebitda),
        "negative_operating_profit_flag": bool(negative_operating_profit),
        "net_cash_flag": bool(
            pd.notna(current["total_debt"])
            and pd.notna(current["cash"])
            and current["total_debt"] - current["cash"] <= 0
        ),
        "financial_diagnostics": "|".join(
            name for flag, name in (
                (negative_equity, "NEGATIVE_EQUITY"),
                (negative_ebitda, "NEGATIVE_EBITDA"),
                (negative_operating_profit, "NEGATIVE_OPERATING_PROFIT"),
            ) if flag
        ),
    }
    result["revenue_cagr_3y"] = (
        ((revenues.iloc[0] / revenues.iloc[3]) ** (1 / 3) - 1) * 100
        if len(revenues) >= 4 and revenues.iloc[0] > 0 and revenues.iloc[3] > 0
        else np.nan
    )
    return result


# Level metrics taken from trailing-twelve-month statements when available.
TTM_LEVEL_KEYS = (
    "revenue", "gross_profit", "operating_income", "net_income", "ebit", "ebitda",
    "interest_expense", "operating_cash_flow", "capital_expenditure", "cash",
    "total_debt", "equity", "assets", "free_cash_flow", "gross_margin_pct",
    "operating_margin_pct", "net_margin_pct", "fcf_margin_pct", "cash_conversion",
    "net_debt", "debt_to_equity", "net_debt_to_ebitda", "interest_coverage",
    "return_on_equity", "return_on_assets", "roic", "negative_equity_flag",
    "negative_ebitda_flag", "negative_operating_profit_flag", "net_cash_flag",
    "financial_diagnostics",
)
# Year-over-year metrics that need two full trailing-twelve-month windows.
TTM_GROWTH_KEYS = (
    "revenue_growth_yoy", "operating_margin_change_1y", "fcf_growth_yoy",
    "earnings_growth_yoy",
)


def _consecutive_quarters(frame: Any, count: int) -> pd.DataFrame:
    """Newest ``count`` quarterly columns, only if they are evenly spaced."""
    working = _ordered_frame(frame)
    if working.shape[1] < count:
        return pd.DataFrame()
    working = working.iloc[:, :count]
    dates = pd.to_datetime(working.columns, errors="coerce", utc=True)
    if dates.isna().any():
        return pd.DataFrame()
    gaps = np.asarray((dates[:-1] - dates[1:]).days)
    if len(gaps) and not ((gaps >= 75) & (gaps <= 105)).all():
        return pd.DataFrame()
    return working


def _ttm_frame(quarterly: Any, windows: int) -> pd.DataFrame:
    """Sum flow rows over ``windows`` trailing-four-quarter blocks.

    A block is produced only when every one of its four quarters is present, so
    a missing quarter can never be silently skipped over.
    """
    quarters = _consecutive_quarters(quarterly, 4 * windows)
    if quarters.empty:
        return pd.DataFrame()
    numeric = quarters.apply(pd.to_numeric, errors="coerce")
    columns = {}
    for block in range(windows):
        part = numeric.iloc[:, block * 4:(block + 1) * 4]
        columns[part.columns[0]] = part.sum(axis=1, min_count=4).where(
            part.notna().all(axis=1)
        )
    return pd.DataFrame(columns)


def _balance_frame(quarterly: Any, windows: int) -> pd.DataFrame:
    """Quarter-end balances matching each trailing-twelve-month window."""
    quarters = _consecutive_quarters(quarterly, 4 * (windows - 1) + 1)
    if quarters.empty:
        return pd.DataFrame()
    return quarters.iloc[:, [block * 4 for block in range(windows)]]


def derive_ttm_fundamentals(
    income: Any, balance: Any, cashflow: Any, sector: Any = ""
) -> dict[str, Any]:
    """Trailing-twelve-month features from quarterly statements.

    Two windows (eight quarters) also yield TTM-over-TTM growth; one window
    yields only current levels.  An empty dict means no usable TTM data.
    """
    for windows in (2, 1):
        ttm_income = _ttm_frame(income, windows)
        ttm_cashflow = _ttm_frame(cashflow, windows)
        quarter_balance = _balance_frame(balance, windows)
        if ttm_income.empty:
            continue
        derived = derive_fundamentals(ttm_income, quarter_balance, ttm_cashflow, sector)
        if pd.isna(number(derived.get("revenue"))):
            continue
        keys = TTM_LEVEL_KEYS + (TTM_GROWTH_KEYS if windows == 2 else ())
        output = {key: derived.get(key, np.nan) for key in keys}
        output["ttm_windows"] = windows
        output["ttm_period_end"] = str(ttm_income.columns[0])
        return output
    return {}


def latest_quarter_revenue_growth(income: Any) -> float:
    """Latest quarter revenue versus the same quarter one year earlier."""
    quarters = _consecutive_quarters(income, 5)
    if quarters.empty:
        return np.nan
    lookup = {normalize_statement_label(label): label for label in quarters.index}
    for name in ALIASES["revenue"]:
        label = lookup.get(normalize_statement_label(name))
        if label is not None:
            values = pd.to_numeric(quarters.loc[label], errors="coerce")
            if pd.notna(values.iloc[4]) and values.iloc[4] > 0:
                return change_pct(values.iloc[0], values.iloc[4])
    return np.nan


def merge_fundamentals(
    annual: dict[str, Any], ttm: dict[str, Any], quarter_growth: float = np.nan
) -> dict[str, Any]:
    """Prefer fresher trailing-twelve-month values over the last fiscal year."""
    result = dict(annual)
    replaced = [
        key for key, value in ttm.items()
        if key in TTM_LEVEL_KEYS + TTM_GROWTH_KEYS
        and not (isinstance(value, float) and np.isnan(value))
        and value is not None
    ]
    for key in replaced:
        result[key] = ttm[key]
    result["revenue_growth_latest_quarter_yoy"] = quarter_growth
    result["fundamentals_basis"] = (
        "TTM_WITH_TTM_GROWTH" if ttm.get("ttm_windows") == 2
        else "TTM_LEVELS_ANNUAL_GROWTH" if ttm
        else "ANNUAL"
    )
    result["ttm_period_end"] = ttm.get("ttm_period_end")
    return result
