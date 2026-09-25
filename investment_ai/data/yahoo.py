from __future__ import annotations
import random,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from typing import Any,Callable
import pandas as pd
import yfinance as yf
from investment_ai.config import MAX_WORKERS,FORCE_REFRESH,ANALYST_TTL_HOURS,VALUATION_TTL_HOURS,FUNDAMENTALS_TTL_HOURS
from investment_ai.data.cache import JsonCache
from investment_ai.features.analyst import parse_recommendations,parse_eps_trend,parse_revisions,parse_estimates,parse_targets,parse_actions,parse_surprises
from investment_ai.features.fundamentals import derive_fundamentals
from investment_ai.features.valuation import parse_valuation
RATE_MARKERS=('rate limit','too many requests','429','yfratelimit')
def call_with_retry(fn:Callable[[],Any],attempts:int=3,base:float=.5):
 last=None
 for attempt in range(attempts):
  try:return fn()
  except Exception as exc:
   last=exc
   if attempt+1<attempts:time.sleep(base*2**attempt+random.uniform(0,.25))
 raise last
def _safe(fn):
 try:return call_with_retry(fn)
 except Exception:return None
class YahooClient:
 def __init__(self,cache:JsonCache):self.cache=cache
 def fetch_symbol(self,symbol:str,sector:str='')->dict[str,Any]:
  ticker=yf.Ticker(symbol) # exactly one object for all symbol endpoints
  analyst,asource=self.cache.get_or_fetch('analyst',symbol,ANALYST_TTL_HOURS,lambda:self._analyst(ticker),FORCE_REFRESH)
  valuation,vsource=self.cache.get_or_fetch('valuation',symbol,VALUATION_TTL_HOURS,lambda:parse_valuation(call_with_retry(lambda:ticker.get_valuation_measures(freq='quarterly',periods=12))),FORCE_REFRESH)
  fundamentals,fsource=self.cache.get_or_fetch('fundamentals',symbol,FUNDAMENTALS_TTL_HOURS,lambda:self._fundamentals(ticker,sector),FORCE_REFRESH)
  return {'symbol':symbol,**analyst.get('data',{}),**valuation.get('data',{}),**fundamentals.get('data',{}),'analyst_cache_status':asource,'valuation_cache_status':vsource,'fundamental_cache_status':fsource,'provider_success':float(not any(x=='error' for x in (asource,vsource,fsource))),'data_errors':' | '.join(x for x in (asource,vsource,fsource) if 'error' in x)}
 def _analyst(self,t):
  targets=parse_targets(_safe(t.get_analyst_price_targets)); rec=parse_recommendations(_safe(t.get_recommendations_summary)); eps=parse_eps_trend(_safe(t.get_eps_trend)); rev=parse_revisions(_safe(t.get_eps_revisions)); earn=parse_estimates(_safe(t.get_earnings_estimate),'eps'); revenue=parse_estimates(_safe(t.get_revenue_estimate),'revenue'); actions=parse_actions(_safe(t.get_upgrades_downgrades)); surprise=parse_surprises(_safe(t.get_earnings_history)); growth=_safe(t.get_growth_estimates); dates=_safe(lambda:t.get_earnings_dates(limit=12)); info=_safe(t.get_info) or {}
  result={**targets,**rec,**eps,**rev,**earn,**revenue,**actions,**surprise,'sector_yahoo':info.get('sector'),'current_price_provider':info.get('currentPrice',info.get('regularMarketPrice'))}
  if isinstance(growth,pd.DataFrame) and not growth.empty:
   for idx,row in growth.iterrows(): result[f"growth_context_{str(idx).replace('+','plus_')}_stock"]=row.get('stock')
  if isinstance(dates,pd.DataFrame) and not dates.empty:
   now=pd.Timestamp.now(tz='UTC'); ds=pd.to_datetime(dates.index,utc=True,errors='coerce');future=ds[ds>=now];past=ds[ds<now];result['next_earnings_date']=str(future.min()) if len(future) else None;result['days_to_next_earnings']=(future.min()-now).total_seconds()/86400 if len(future) else None;result['days_since_last_earnings']=(now-past.max()).total_seconds()/86400 if len(past) else None
  return result
 def _fundamentals(self,t,sector):
  # Both frequencies are fetched/cached; annual drives stable v3 ratios.
  annual_i=call_with_retry(lambda:t.get_income_stmt(freq='yearly'));_safe(lambda:t.get_income_stmt(freq='quarterly'));annual_b=call_with_retry(lambda:t.get_balance_sheet(freq='yearly'));_safe(lambda:t.get_balance_sheet(freq='quarterly'));annual_c=call_with_retry(lambda:t.get_cash_flow(freq='yearly'));_safe(lambda:t.get_cash_flow(freq='quarterly'))
  return derive_fundamentals(annual_i,annual_b,annual_c,sector)
 def fetch_many(self,universe:pd.DataFrame)->pd.DataFrame:
  rows=[]
  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
   jobs={pool.submit(self.fetch_symbol,r.symbol,getattr(r,'sector','')):r.symbol for r in universe.itertuples()}
   for future in as_completed(jobs):
    try:rows.append(future.result())
    except Exception as exc:rows.append({'symbol':jobs[future],'data_errors':str(exc),'provider_success':0})
  return pd.DataFrame(rows)
