from __future__ import annotations
import numpy as np
from investment_ai.scoring.common import curve,weighted

def risk_and_confidence(row:dict)->dict:
 market=np.nanmean([curve(row.get('volatility_60d'),[(10,10),(30,45),(60,85),(100,100)]),curve(abs(row.get('max_drawdown_1y',np.nan)),[(5,10),(20,45),(50,100)])])
 balance=np.nanmean([curve(row.get('debt_to_equity'),[(0,5),(1,30),(3,75),(6,100)]),curve(row.get('interest_coverage'),[(0,100),(3,55),(10,10)])])
 event=100-(row.get('event_timing_score') if row.get('event_timing_score') is not None else np.nan)
 disagreement=np.nanmean([curve(row.get('eps_plus_1y_dispersion_pct'),[(5,5),(30,55),(80,100)]),curve(row.get('negative_rating_pct_current'),[(0,0),(20,50),(50,100)])])
 liquidity=curve(row.get('average_dollar_volume_20d'),[(1e6,100),(1e7,55),(1e8,10)])
 risk,_,_=weighted({'market':market,'balance':balance,'event':event,'disagreement':disagreement,'liquidity':liquidity},{'market':.30,'balance':.25,'event':.15,'disagreement':.15,'liquidity':.10},0)
 cover=np.nanmean([row.get('long_term_coverage',np.nan),row.get('short_term_coverage',np.nan),row.get('expectations_coverage',np.nan)])*100
 analyst=curve(row.get('rating_count'),[(0,20),(5,50),(15,85),(30,100)]); freshness=100 if not any('stale' in str(row.get(k,'')) for k in ('analyst_cache_status','valuation_cache_status','fundamental_cache_status')) else 65; provider=row.get('provider_success',0)*100
 confidence,_,_=weighted({'coverage':cover,'analyst':analyst,'freshness':freshness,'provider':provider},{'coverage':.45,'analyst':.20,'freshness':.20,'provider':.15},0)
 return {'risk_score':risk,'confidence_score':confidence,'market_risk_score':market,'balance_sheet_risk_score':balance,'analyst_disagreement_score':disagreement}
