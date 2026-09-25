from __future__ import annotations
import numpy as np
import pandas as pd
from investment_ai.scoring.common import curve,weighted

def score_long_term(row:dict)->dict:
 financial=bool(row.get('is_financial',False))
 quality={'fcf_margin':np.nan if financial else curve(row.get('fcf_margin_pct'),[(-10,0),(0,35),(10,70),(25,100)]),'operating_margin':curve(row.get('operating_margin_pct'),[(-10,0),(0,30),(15,75),(35,100)]),'margin_trend':curve(row.get('operating_margin_change_1y'),[(-10,0),(0,50),(10,100)]),'fcf_growth':np.nan if financial else curve(row.get('fcf_growth_yoy'),[(-50,0),(0,50),(50,100)]),'cash_conversion':np.nan if financial else curve(row.get('cash_conversion'),[(0,0),(.8,70),(1.5,100),(3,65)]),'returns':curve(np.nanmean([row.get('return_on_equity',np.nan),row.get('roic',np.nan)]),[(-10,0),(0,30),(15,75),(30,100)])}
 q,qc,qs=weighted(quality,{'fcf_margin':.20,'operating_margin':.15,'margin_trend':.15,'fcf_growth':.15,'cash_conversion':.15,'returns':.20},.60)
 growth={'forward_revenue':curve(row.get('forward_revenue_growth'),[(-.2,0),(0,40),(.3,100)]),'forward_eps':curve(row.get('forward_eps_growth'),[(-.3,0),(0,40),(.4,100)]),'historical_revenue':curve(row.get('revenue_growth_yoy'),[(-20,0),(0,40),(30,100)]),'acceleration':curve(row.get('growth_acceleration'),[(-20,0),(0,50),(20,100)])}
 g,gc,gs=weighted(growth,{'forward_revenue':.30,'forward_eps':.30,'historical_revenue':.20,'acceleration':.20},.60)
 valuation={'forward_pe':row.get('forward_pe_peer_percentile'),'ev_ebitda':np.nan if financial else row.get('ev_ebitda_peer_percentile'),'fcf_yield':np.nan if financial else row.get('fcf_yield_peer_percentile'),'peg':row.get('peg_peer_percentile'),'own_history':row.get('forward_pe_own_history_percentile')}
 if financial:valuation['price_book']=row.get('price_book_peer_percentile')
 vw={'forward_pe':.25,'ev_ebitda':.25,'fcf_yield':.20,'peg':.15,'own_history':.15,'price_book':.25};v,vc,vs=weighted(valuation,vw,.50)
 trend={'rs126':row.get('rs_126d_percentile'),'rs252':row.get('rs_252d_percentile'),'ma200':curve(row.get('price_vs_ma200_pct'),[(-30,0),(0,55),(20,100),(50,60)])};lt,ltc,lts=weighted(trend,{'rs126':.35,'rs252':.35,'ma200':.30},.60)
 safety={'leverage':curve(row.get('debt_to_equity'),[(0,100),(1,75),(3,25),(6,0)]),'coverage':curve(row.get('interest_coverage'),[(0,0),(3,50),(10,100)]),'profitability':curve(row.get('net_margin_pct'),[(-10,0),(0,40),(20,100)])};safe,sc,ss=weighted(safety,{'leverage':.4,'coverage':.35,'profitability':.25},.40)
 pillars={'quality':q,'growth':g,'valuation':v,'expectations':row.get('expectations_score'),'long_trend':lt,'financial_safety':safe};score,cov,status=weighted(pillars,{'quality':.25,'growth':.20,'valuation':.20,'expectations':.20,'long_trend':.10,'financial_safety':.05},.65)
 return {'quality_score':q,'quality_coverage':qc,'quality_status':qs,'growth_score':g,'growth_coverage':gc,'growth_status':gs,'valuation_score':v,'valuation_coverage':vc,'valuation_status':vs,'long_trend_score':lt,'long_trend_coverage':ltc,'financial_safety_score':safe,'long_term_score':score,'long_term_coverage':cov,'long_term_status':status}
