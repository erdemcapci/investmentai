from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from investment_ai.scoring.common import number, ratio
FINANCIAL_SECTORS={"financials","financial services","banks","insurance"}
ALIASES={
"revenue":["Total Revenue","Operating Revenue"],"gross_profit":["Gross Profit"],"operating_income":["Operating Income","EBIT"],"net_income":["Net Income","Net Income Common Stockholders"],"ebit":["EBIT","Operating Income"],"ebitda":["EBITDA","Normalized EBITDA"],"interest_expense":["Interest Expense","Interest Expense Non Operating"],"tax_provision":["Tax Provision"],"pretax_income":["Pretax Income"],
"operating_cash_flow":["Operating Cash Flow","Total Cash From Operating Activities"],"capital_expenditure":["Capital Expenditure","Capital Expenditures"],
"cash":["Cash Cash Equivalents And Short Term Investments","Cash And Cash Equivalents"],"total_debt":["Total Debt"],"equity":["Stockholders Equity","Total Equity Gross Minority Interest"],"assets":["Total Assets"]}
def is_financial(sector:Any)->bool: return str(sector).strip().lower() in FINANCIAL_SECTORS or 'bank' in str(sector).lower()
def _series(frame:Any,names:list[str])->pd.Series:
    if not isinstance(frame,pd.DataFrame) or frame.empty:return pd.Series(dtype=float)
    lookup={str(i).lower():i for i in frame.index}
    for name in names:
        if name.lower() in lookup:return pd.to_numeric(frame.loc[lookup[name.lower()]],errors='coerce').dropna()
    return pd.Series(dtype=float)
def derive_fundamentals(income:Any,balance:Any,cashflow:Any,sector:Any="")->dict[str,Any]:
    series={k:_series(income if k in {'revenue','gross_profit','operating_income','net_income','ebit','ebitda','interest_expense','tax_provision','pretax_income'} else cashflow if k in {'operating_cash_flow','capital_expenditure'} else balance,v) for k,v in ALIASES.items()}
    cur={k:(s.iloc[0] if len(s) else np.nan) for k,s in series.items()}; previous={k:(s.iloc[1] if len(s)>1 else np.nan) for k,s in series.items()}
    capex=cur['capital_expenditure']; fcf=cur['operating_cash_flow'] + capex if pd.notna(capex) and capex<0 else cur['operating_cash_flow']-capex if pd.notna(capex) else np.nan
    prev_capex=previous['capital_expenditure']; prev_fcf=previous['operating_cash_flow'] + prev_capex if pd.notna(prev_capex) and prev_capex<0 else previous['operating_cash_flow']-prev_capex if pd.notna(prev_capex) else np.nan
    financial=is_financial(sector); invested=cur['total_debt']+cur['equity']-cur['cash']; tax=ratio(cur['tax_provision'],cur['pretax_income']); tax=np.clip(tax,0,0.35) if pd.notna(tax) else np.nan
    result={**cur,"free_cash_flow":fcf,"gross_margin_pct":ratio(cur['gross_profit'],cur['revenue'])*100,"operating_margin_pct":ratio(cur['operating_income'],cur['revenue'])*100,"net_margin_pct":ratio(cur['net_income'],cur['revenue'])*100,"operating_margin_change_1y":(ratio(cur['operating_income'],cur['revenue'])-ratio(previous['operating_income'],previous['revenue']))*100,"revenue_growth_yoy":ratio(cur['revenue']-previous['revenue'],abs(previous['revenue']))*100,"fcf_margin_pct":ratio(fcf,cur['revenue'])*100,"fcf_growth_yoy":ratio(fcf-prev_fcf,abs(prev_fcf))*100,"cash_conversion":ratio(fcf,cur['net_income']) if number(cur['net_income'])>0 else np.nan,"net_debt":cur['total_debt']-cur['cash'],"debt_to_equity":ratio(cur['total_debt'],cur['equity']),"net_debt_to_ebitda":np.nan if financial else ratio(cur['total_debt']-cur['cash'],cur['ebitda']),"interest_coverage":ratio(cur['ebit'],abs(cur['interest_expense'])),"return_on_equity":ratio(cur['net_income'],cur['equity'])*100,"return_on_assets":ratio(cur['net_income'],cur['assets'])*100,"roic_applicable":not financial,"roic":np.nan if financial or invested<=0 or pd.isna(tax) else cur['ebit']*(1-tax)/invested*100,"is_financial":financial}
    rev=series['revenue']; result['revenue_cagr_3y']=(rev.iloc[0]/rev.iloc[3])**(1/3)*100-100 if len(rev)>=4 and rev.iloc[0]>0 and rev.iloc[3]>0 else np.nan
    return result
