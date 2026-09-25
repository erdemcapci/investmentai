from __future__ import annotations
import numpy as np
from investment_ai.scoring.common import curve,weighted
from investment_ai.features.technical import setup_scores,event_timing_score

def score_short_term(row:dict)->dict:
 setup=setup_scores(row)
 rs,rsc,rss=weighted({'rs20':row.get('rs_20d_percentile'),'rs60':row.get('rs_60d_percentile'),'rs126':row.get('rs_126d_percentile')},{'rs20':.50,'rs60':.35,'rs126':.15},.60)
 volume=curve(row.get('relative_volume_1d'),[(.3,20),(1,55),(1.8,100),(4,70)])
 technical=np.nanmean([curve(row.get('price_vs_ma20_pct'),[(-15,0),(0,60),(8,100),(25,40)]),curve(row.get('price_vs_ma50_pct'),[(-20,0),(0,60),(15,100),(40,40)])])
 event=event_timing_score(row.get('days_to_next_earnings'),row.get('days_since_last_earnings'))
 pillars={'relative_strength':rs,'setup':setup['setup_quality_score'],'expectations':row.get('expectations_score'),'volume':volume,'technical':technical,'event':event}
 score,cov,status=weighted(pillars,{'relative_strength':.20,'setup':.25,'expectations':.20,'volume':.10,'technical':.15,'event':.10},.65)
 return {**setup,'short_rs_score':rs,'short_rs_coverage':rsc,'volume_confirmation_score':volume,'technical_trend_score':technical,'event_timing_score':event,'short_term_score':score,'short_term_coverage':cov,'short_term_status':status}
