from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd
from investment_ai.scoring.common import change_pct, number, ratio, curve, weighted

PERIODS=("0q","+1q","0y","+1y")

def _frame(value:Any)->pd.DataFrame: return value.copy() if isinstance(value,pd.DataFrame) else pd.DataFrame()
def _row(frame:pd.DataFrame,label:str)->pd.Series:
    if frame.empty: return pd.Series(dtype=object)
    labels={str(i).lower():i for i in frame.index}; key=labels.get(label.lower())
    return frame.loc[key] if key is not None else pd.Series(dtype=object)
def _get(row:pd.Series,*names:str):
    norm={str(k).replace(' ','').lower():v for k,v in row.items()}
    for name in names:
        if name.replace(' ','').lower() in norm: return number(norm[name.replace(' ','').lower()])
    return np.nan

def parse_recommendations(frame:Any)->dict[str,Any]:
    frame=_frame(frame); out={}
    for period in ("0m","-1m","-2m","-3m"):
        row=_row(frame,period); counts=[_get(row,n) for n in ("strongBuy","buy","hold","sell","strongSell")]
        valid=[0 if pd.isna(x) else max(x,0) for x in counts]; total=sum(valid)
        prefix=period.replace('-','minus_').replace('0m','current')
        out[f"rating_count_{prefix}"]=total if total else np.nan
        out[f"positive_rating_pct_{prefix}"]=(valid[0]+valid[1])/total*100 if total else np.nan
        out[f"negative_rating_pct_{prefix}"]=(valid[3]+valid[4])/total*100 if total else np.nan
        out[f"hold_pct_{prefix}"]=valid[2]/total*100 if total else np.nan
        out[f"recommendation_strength_{prefix}"]=sum(a*b for a,b in zip(valid,(100,80,50,20,0)))/total if total else np.nan
        if period=='0m': out.update(dict(zip(("strong_buy","buy","hold","sell","strong_sell"),valid)))
    out["positive_rating_pct"]=out.get("positive_rating_pct_current",np.nan); out["rating_count"]=out.get("rating_count_current",np.nan)
    for months,key in ((1,"minus_1m"),(3,"minus_3m")):
        out[f"positive_rating_change_{months}m_pp"]=number(out.get("positive_rating_pct_current"))-number(out.get(f"positive_rating_pct_{key}"))
        out[f"recommendation_strength_change_{months}m"]=number(out.get("recommendation_strength_current"))-number(out.get(f"recommendation_strength_{key}"))
    return out

def parse_eps_trend(frame:Any)->dict[str,Any]:
    out={}
    for period in PERIODS:
        row=_row(_frame(frame),period); prefix=period.replace('+','plus_')
        current=_get(row,"current"); out[f"eps_{prefix}_current"]=current
        for days in (7,30,60,90): out[f"eps_{prefix}_change_{days}d_pct"]=change_pct(current,_get(row,f"{days}daysAgo"))
    return out

def parse_revisions(frame:Any)->dict[str,Any]:
    out={}; frame=_frame(frame)
    for period in PERIODS:
        row=_row(frame,period); prefix=period.replace('+','plus_')
        for days in (7,30):
            up=_get(row,f"upLast{days}days"); down=_get(row,f"downLast{days}days"); total=up+down
            out[f"eps_revision_breadth_{prefix}_{days}d"]=(up-down)/total if pd.notna(total) and total>0 else np.nan
            out[f"eps_up_{prefix}_{days}d"],out[f"eps_down_{prefix}_{days}d"]=up,down
    return out

def parse_estimates(frame:Any,kind:str)->dict[str,Any]:
    out={}; frame=_frame(frame)
    names=("numberOfAnalysts","avg","low","high","yearAgoEps" if kind=='eps' else "yearAgoRevenue","growth")
    for period in PERIODS:
        row=_row(frame,period); prefix=period.replace('+','plus_')
        for name in names: out[f"{kind}_{prefix}_{name}"]=_get(row,name)
        avg=out[f"{kind}_{prefix}_avg"]; out[f"{kind}_{prefix}_dispersion_pct"]=ratio(out[f"{kind}_{prefix}_high"]-out[f"{kind}_{prefix}_low"],abs(avg))*100
    out[f"forward_{kind}_growth"]=out.get(f"{kind}_plus_1y_growth",out.get(f"{kind}_0y_growth",np.nan))
    return out

def parse_targets(value:Any)->dict[str,Any]:
    if isinstance(value,pd.DataFrame): source=value.iloc[:,0].to_dict() if not value.empty else {}
    else: source=value if isinstance(value,dict) else {}
    low=number(source.get("low",source.get("targetLowPrice"))); mean=number(source.get("mean",source.get("targetMeanPrice"))); median=number(source.get("median",source.get("targetMedianPrice"))); high=number(source.get("high",source.get("targetHighPrice")))
    return {"target_low":low,"target_mean":mean,"target_median":median,"target_high":high}

def parse_actions(frame:Any, now:pd.Timestamp|None=None)->dict[str,Any]:
    frame=_frame(frame); now=now or pd.Timestamp.now(tz='UTC'); out={}
    if frame.empty: return out
    dates=pd.to_datetime(frame.index,utc=True,errors='coerce'); actions=frame.get('Action',frame.get('action',pd.Series('',index=frame.index))).astype(str).str.lower()
    for days in (7,30,90):
        mask=dates>=now-pd.Timedelta(days=days); selected=actions[mask]
        up=selected.str.contains('up').sum(); down=selected.str.contains('down').sum()
        out.update({f"rating_actions_{days}d":len(selected),f"upgrades_{days}d":up,f"downgrades_{days}d":down,f"rating_action_balance_{days}d":(up-down)/(up+down) if up+down else np.nan})
    last=frame.iloc[0]; out.update({"latest_rating_date":str(dates[0]),"latest_rating_firm":last.get('Firm',last.get('firm','')),"latest_rating_action":last.get('Action',last.get('action','')),"latest_rating_from":last.get('FromGrade',''),"latest_rating_to":last.get('ToGrade','')})
    return out

def parse_surprises(frame:Any)->dict[str,Any]:
    frame=_frame(frame); col=next((c for c in frame.columns if str(c).lower() in {'surprisepercent','surprise(%)','surprise_pct'}),None)
    vals=pd.to_numeric(frame[col],errors='coerce').dropna().tail(4) if col else pd.Series(dtype=float)
    if len(vals)<2:return {}
    if vals.abs().max()<=2: vals=vals*100
    return {"positive_surprise_rate_4q":(vals>0).mean()*100,"median_eps_surprise_4q":vals.median(),"mean_eps_surprise_4q":vals.mean(),"eps_surprise_volatility_4q":vals.std(ddof=0)}

def expectations_score(row:dict[str,Any])->dict[str,Any]:
    momentum=np.nanmean([curve(row.get(f"eps_0q_change_{d}d_pct"),[(-20,0),(0,50),(20,100)]) for d in (7,30,60,90)])
    breadth=curve(np.nanmean([row.get('eps_revision_breadth_0q_7d',np.nan),row.get('eps_revision_breadth_0q_30d',np.nan)]),[(-1,0),(0,50),(1,100)])
    recommendation=curve(np.nanmean([row.get('positive_rating_change_1m_pp',np.nan),row.get('positive_rating_change_3m_pp',np.nan)]),[(-20,0),(0,50),(20,100)])
    action=curve(row.get('rating_action_balance_30d'),[(-1,0),(0,50),(1,100)])
    evidence=curve(np.nanmean([row.get('forward_eps_growth',np.nan),row.get('forward_revenue_growth',np.nan)])*100,[(-20,0),(0,50),(30,100)])
    target=curve(row.get('target_upside_pct'),[(-30,0),(0,40),(30,80),(60,100)])
    surprise=curve(row.get('positive_surprise_rate_4q'),[(0,0),(50,50),(100,100)])
    comps={"eps_momentum":momentum,"revision_breadth":breadth,"recommendation_trend":recommendation,"rating_actions":action,"forward_evidence":evidence,"target_signal":target,"earnings_execution":surprise,"target_momentum":row.get('target_momentum_score',np.nan)}
    score,cov,status=weighted(comps,{"eps_momentum":.30,"revision_breadth":.20,"recommendation_trend":.15,"rating_actions":.10,"forward_evidence":.10,"target_signal":.05,"earnings_execution":.05,"target_momentum":.05},.60)
    return {**{f"expectations_{k}_score":v for k,v in comps.items()},"expectations_score":score,"expectations_coverage":cov,"expectations_status":status}
