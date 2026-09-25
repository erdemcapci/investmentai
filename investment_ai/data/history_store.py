from __future__ import annotations
import sqlite3
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import numpy as np

SNAPSHOT_FIELDS=['target_low','target_mean','target_median','target_high','strong_buy','buy','hold','sell','strong_sell','positive_rating_pct','rating_count','eps_0q_current','eps_plus_1q_current','eps_0y_current','eps_plus_1y_current','revenue_0q_avg','revenue_plus_1q_avg','revenue_0y_avg','revenue_plus_1y_avg','forward_eps_growth','forward_revenue_growth']
class HistoryStore:
 def __init__(self,path:Path):
  Path(path).parent.mkdir(parents=True,exist_ok=True);self.db=sqlite3.connect(path);self.db.row_factory=sqlite3.Row;self._schema()
 def _schema(self):
  cols=', '.join(f'{x} REAL' for x in SNAPSHOT_FIELDS)
  self.db.execute(f'''CREATE TABLE IF NOT EXISTS analyst_snapshots(snapshot_date TEXT NOT NULL,fetched_at_utc TEXT NOT NULL,symbol TEXT NOT NULL,{cols},UNIQUE(snapshot_date,symbol))''')
  self.db.execute('''CREATE TABLE IF NOT EXISTS ranking_history(run_id TEXT,run_timestamp TEXT,symbol TEXT,long_term_score REAL,long_term_rank INTEGER,short_term_score REAL,short_term_rank INTEGER,risk_score REAL,confidence_score REAL,short_term_setup TEXT,UNIQUE(run_id,symbol))''');self.db.commit()
 def upsert_analyst(self,symbol:str,data:dict[str,Any],when:datetime|None=None):
  when=when or datetime.now(timezone.utc); fields=['snapshot_date','fetched_at_utc','symbol']+SNAPSHOT_FIELDS; values=[when.date().isoformat(),when.isoformat(),symbol]+[data.get(k) for k in SNAPSHOT_FIELDS]
  updates=', '.join(f'{x}=excluded.{x}' for x in fields[1:]); q=f"INSERT INTO analyst_snapshots({','.join(fields)}) VALUES ({','.join('?'*len(fields))}) ON CONFLICT(snapshot_date,symbol) DO UPDATE SET {updates}";self.db.execute(q,values);self.db.commit()
 def historical_change(self,symbol:str,field:str,days:int,current:float):
  if field not in SNAPSHOT_FIELDS:return np.nan,'HISTORY_NOT_YET_AVAILABLE'
  cutoff=(date.today()-timedelta(days=days)).isoformat(); row=self.db.execute(f'SELECT {field} FROM analyst_snapshots WHERE symbol=? AND snapshot_date<=? AND {field} IS NOT NULL ORDER BY snapshot_date DESC LIMIT 1',(symbol,cutoff)).fetchone()
  if not row or not row[0] or current is None:return np.nan,'HISTORY_NOT_YET_AVAILABLE'
  return (current-row[0])/abs(row[0])*100,'AVAILABLE'
 def save_rankings(self,run_id:str,ts:str,rows:list[dict[str,Any]]):
  keys=['long_term_score','long_term_rank','short_term_score','short_term_rank','risk_score','confidence_score','short_term_setup']
  self.db.executemany(f"INSERT OR REPLACE INTO ranking_history VALUES ({','.join('?'*10)})",[[run_id,ts,r['symbol']]+[r.get(k) for k in keys] for r in rows]);self.db.commit()
 def changes(self,symbol:str,days:int=7):
  cutoff=(datetime.now(timezone.utc)-timedelta(days=days)).isoformat(); row=self.db.execute('SELECT * FROM ranking_history WHERE symbol=? AND run_timestamp<=? ORDER BY run_timestamp DESC LIMIT 1',(symbol,cutoff)).fetchone();return dict(row) if row else None
 def close(self):self.db.close()
