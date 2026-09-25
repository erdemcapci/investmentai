from __future__ import annotations
import pandas as pd

def rank_results(frame:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
 result=frame.copy()
 lt=result[result.long_term_score.notna()].sort_values(['long_term_score','confidence_score','expectations_score','quality_score'],ascending=False,kind='stable').copy();lt['long_term_rank']=range(1,len(lt)+1)
 st=result[result.short_term_score.notna()].sort_values(['short_term_score','confidence_score','expectations_score','short_rs_score'],ascending=False,kind='stable').copy();st['short_term_rank']=range(1,len(st)+1)
 ltr=lt.set_index('symbol').long_term_rank;strank=st.set_index('symbol').short_term_rank
 result['long_term_rank']=result.symbol.map(ltr);result['short_term_rank']=result.symbol.map(strank)
 lt=result[result.long_term_score.notna()].sort_values('long_term_rank');st=result[result.short_term_score.notna()].sort_values('short_term_rank')
 return lt,st
