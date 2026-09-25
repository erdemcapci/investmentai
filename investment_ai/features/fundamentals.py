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
        "debt_to_equity": ratio(current["total_debt"], current["equity"]),
        "net_debt_to_ebitda": (
            np.nan
            if financial
            else ratio(current["total_debt"] - current["cash"], current["ebitda"])
        ),
        "interest_coverage": ratio(current["ebit"], abs(current["interest_expense"])),
        "return_on_equity": ratio(current["net_income"], current["equity"]) * 100,
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
    }
    result["revenue_cagr_3y"] = (
        ((revenues.iloc[0] / revenues.iloc[3]) ** (1 / 3) - 1) * 100
        if len(revenues) >= 4 and revenues.iloc[0] > 0 and revenues.iloc[3] > 0
        else np.nan
    )
    return result
