"""Investment AI v3 entry point. Run only: ``python main.py``."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import uuid
import pandas as pd
import yfinance as yf
from investment_ai.config import *
from investment_ai.data.constituents import fetch_index_constituents
from investment_ai.data.prices import download_prices,build_price_features
from investment_ai.data.cache import JsonCache
from investment_ai.data.yahoo import YahooClient
from investment_ai.data.history_store import HistoryStore
from investment_ai.pipeline import build_analysis
from investment_ai.reporting.tables import print_rankings
from investment_ai.reporting.export import export_run

def download_combined_constituents()->pd.DataFrame:
 frame=fetch_index_constituents(CACHE_DIR/'constituents')
 required={'S&P 500','STOXX Europe 600'}
 present=set('|'.join(frame.index_name.dropna()).split(' | '))
 if not required.issubset(present):raise RuntimeError('Both constituent sources are required for the combined universe')
 return frame

def methodology()->pd.DataFrame:
 return pd.DataFrame([
  {'score':'Long term','formula':'25% Quality + 20% Growth + 20% Valuation + 20% Expectations + 10% Long Trend + 5% Financial Safety'},
  {'score':'Short term','formula':'20% Relative Strength + 25% Setup + 20% Expectations + 10% Volume + 15% Technical Trend + 10% Event Timing'},
  {'score':'Risk','formula':'Separate observed-risk measure; never multiplied into attractiveness'},
  {'score':'Confidence','formula':'Separate coverage/freshness/provider measure; never multiplied into attractiveness'},])
def main()->None:
 started=datetime.now(timezone.utc);run_id=f"{started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
 print('Loading combined S&P 500 + STOXX Europe 600 universe...');universe=download_combined_constituents();symbols=universe.symbol.tolist()
 print(f'Downloading two years of daily OHLCV for {len(symbols)} symbols in one batch...');prices=build_price_features(download_prices(symbols,PRICE_PERIOD),symbols)
 print('Loading cached/fresh Yahoo expectations, valuation, and fundamentals...');provider=YahooClient(JsonCache(CACHE_DIR)).fetch_many(universe)
 store=HistoryStore(HISTORY_DB)
 history_rows=[]
 for row in provider.to_dict('records'):
  target_change,status=store.historical_change(row['symbol'],'target_median',30,row.get('target_median'));row['target_change_30d_pct']=target_change;row['target_history_status']=status;store.upsert_analyst(row['symbol'],row);history_rows.append(row)
 provider=pd.DataFrame(history_rows)
 full,long_term,short_term=build_analysis(universe,prices,provider);print_rankings(long_term,short_term,TOP_N)
 store.save_rankings(run_id,started.isoformat(),full[full.long_term_rank.notna()|full.short_term_rank.notna()].to_dict('records'));store.close()
 metadata={'run_id':run_id,'run_timestamp_utc':started.isoformat(),'scoring_model_version':SCORING_MODEL_VERSION,'yfinance_version':yf.__version__,'universe_count':len(universe),'sp500_count':universe.index_name.str.contains('S&P 500',regex=False).sum(),'stoxx600_count':universe.index_name.str.contains('STOXX Europe 600',regex=False).sum(),'price_coverage':prices.current_price.notna().mean(),'analyst_coverage':provider.rating_count.notna().mean() if 'rating_count' in provider else 0,'fundamental_coverage':provider.revenue.notna().mean() if 'revenue' in provider else 0,'valuation_coverage':provider.forward_pe.notna().mean() if 'forward_pe' in provider else 0,'ranked_long_term_count':len(long_term),'ranked_short_term_count':len(short_term),'provider_error_count':provider.data_errors.fillna('').astype(bool).sum(),'freshness_notes':'prices every run; expectations 18h; valuation 24h; statements 7d'}
 if EXPORT_RESULTS:
  directory=Path('investment_ai_runs')/run_id;export_run(directory,full,long_term,short_term,metadata,methodology());print(f'Exported {directory}')
 print('\nResearch support only: no orders, guarantees, or automated buy/sell advice.')
if __name__=='__main__':main()
