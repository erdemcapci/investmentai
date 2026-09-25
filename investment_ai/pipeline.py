from __future__ import annotations
import numpy as np
import pandas as pd
from investment_ai.features.analyst import expectations_score
from investment_ai.features.valuation import add_peer_percentiles
from investment_ai.features.technical import add_relative_strength
from investment_ai.scoring.long_term import score_long_term
from investment_ai.scoring.short_term import score_short_term
from investment_ai.features.risk import risk_and_confidence
from investment_ai.scoring.ranking import rank_results

def _drivers(r):
 positives=[];negatives=[]
 if pd.notna(r.get('eps_0q_change_30d_pct')): (positives if r['eps_0q_change_30d_pct']>0 else negatives).append(f"EPS estimates {r['eps_0q_change_30d_pct']:+.1f}% over 30d")
 if pd.notna(r.get('forward_revenue_growth')): positives.append(f"Forward revenue growth {r['forward_revenue_growth']*100:.1f}%") if r['forward_revenue_growth']>0 else negatives.append('Forward revenue expected to contract')
 if r.get('rs_126d_percentile',0)>=80:positives.append(f"6m relative strength {r['rs_126d_percentile']:.0f}th percentile")
 if pd.notna(r.get('days_to_next_earnings')) and r['days_to_next_earnings']<=7:negatives.append(f"Earnings in {r['days_to_next_earnings']:.0f} days")
 if r.get('risk_score',0)>=70:negatives.append('Elevated observed risk')
 return (positives or ['No dominant positive driver'])[0],(negatives or ['No dominant negative driver'])[0]
def build_analysis(universe:pd.DataFrame,prices:pd.DataFrame,provider:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
 frame=universe.merge(prices,on='symbol',how='left').merge(provider,on='symbol',how='left');frame['sector']=frame.get('sector',frame.get('gics_sector')).fillna(frame.get('sector_yahoo'))
 frame['target_upside_pct']=(frame.target_median.fillna(frame.target_mean)/frame.current_price-1)*100
 frame['fcf_yield']=frame.free_cash_flow/frame.market_cap
 frame=add_peer_percentiles(frame,{'forward_pe':False,'ev_ebitda':False,'fcf_yield':True,'peg':False,'price_book':False});frame=add_relative_strength(frame)
 expectation=pd.DataFrame([expectations_score(r) for r in frame.to_dict('records')]);frame=pd.concat([frame.reset_index(drop=True),expectation],axis=1)
 short=pd.DataFrame([score_short_term(r) for r in frame.to_dict('records')]);frame=pd.concat([frame,short],axis=1)
 long=pd.DataFrame([score_long_term(r) for r in frame.to_dict('records')]);frame=pd.concat([frame,long],axis=1)
 rc=pd.DataFrame([risk_and_confidence(r) for r in frame.to_dict('records')]);frame=pd.concat([frame,rc],axis=1)
 lt,st=rank_results(frame);rankmaps=(lt.set_index('symbol').long_term_rank,st.set_index('symbol').short_term_rank);frame['long_term_rank']=frame.symbol.map(rankmaps[0]);frame['short_term_rank']=frame.symbol.map(rankmaps[1]);drivers=frame.apply(_drivers,axis=1,result_type='expand');frame[['top_positive_driver','top_negative_driver']]=drivers
 lt=frame[frame.long_term_rank.notna()].sort_values('long_term_rank');st=frame[frame.short_term_rank.notna()].sort_values('short_term_rank');return frame,lt,st
