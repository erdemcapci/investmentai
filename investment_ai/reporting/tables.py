from __future__ import annotations
import pandas as pd
LT_COLUMNS=['long_term_rank','short_term_rank','symbol','company_name','index_name','sector','long_term_score','quality_score','growth_score','valuation_score','expectations_score','long_trend_score','risk_score','confidence_score','short_term_score','current_price','target_upside_pct','eps_0q_change_30d_pct','forward_eps_growth','forward_revenue_growth','top_positive_driver','top_negative_driver']
ST_COLUMNS=['short_term_rank','long_term_rank','symbol','company_name','index_name','short_term_score','short_term_setup','long_term_score','short_rs_score','setup_quality_score','expectations_score','volume_confirmation_score','technical_trend_score','event_timing_score','risk_score','confidence_score','return_1d_pct','return_5d_pct','return_20d_pct','drawdown_from_20d_high_pct','days_to_next_earnings','top_positive_driver','top_negative_driver']
def _show(frame,columns,top):
 cols=[c for c in columns if c in frame]; display=frame[cols].head(top).copy(); return display.to_string(index=False,max_colwidth=38,float_format=lambda x:f'{x:.1f}')
def print_rankings(lt:pd.DataFrame,st:pd.DataFrame,top:int):
 print('\nLONG-TERM INVESTMENT RANKING\n');print(_show(lt,LT_COLUMNS,top));print('\nSHORT-TERM OPPORTUNITY RANKING\n');print(_show(st,ST_COLUMNS,top))
