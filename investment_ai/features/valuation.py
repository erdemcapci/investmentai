from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from investment_ai.scoring.common import number, percentile
ALIASES={"trailing_pe":["Trailing P/E","TrailingPe"],"forward_pe":["Forward P/E","ForwardPe"],"peg":["PEG Ratio","PegRatio"],"price_sales":["Price/Sales","PsRatio"],"price_book":["Price/Book","PbRatio"],"ev_revenue":["Enterprise Value/Revenue","EnterprisesValueEBITDARatio"],"ev_ebitda":["Enterprise Value/EBITDA","EnterprisesValueEBITDARatio"],"market_cap":["Market Cap","MarketCap"]}
def parse_valuation(frame:Any)->dict[str,Any]:
    if not isinstance(frame,pd.DataFrame) or frame.empty:return {}
    out={}; lookup={str(i).replace(' ','').lower():i for i in frame.index}
    for key,names in ALIASES.items():
        idx=next((lookup[n.replace(' ','').lower()] for n in names if n.replace(' ','').lower() in lookup),None)
        vals=pd.to_numeric(frame.loc[idx],errors='coerce').dropna() if idx is not None else pd.Series(dtype=float)
        if key in {'trailing_pe','forward_pe','peg','ev_ebitda'}: vals=vals[vals>0]
        out[key]=number(vals.iloc[0]) if len(vals) else np.nan
        out[f"{key}_history"]=vals.iloc[1:].tolist()
        out[f"{key}_own_history_percentile"]=(vals.iloc[1:]>=vals.iloc[0]).mean()*100 if len(vals)>=5 else np.nan
    return out
def add_peer_percentiles(frame:pd.DataFrame,metrics:dict[str,bool],minimum_peers:int=15)->pd.DataFrame:
    result=frame.copy(); result['peer_group_method']='universe'
    for metric,higher in metrics.items(): result[f'{metric}_peer_percentile']=np.nan
    for _,idx in result.groupby(result.get('sector',pd.Series('',index=result.index)).fillna('')).groups.items():
        use=idx if len(idx)>=minimum_peers else result.index
        method='sector' if len(idx)>=minimum_peers else 'universe'
        result.loc[idx,'peer_group_method']=method
        for metric,higher in metrics.items(): result.loc[idx,f'{metric}_peer_percentile']=percentile(result.loc[use,metric],higher).reindex(idx)
    return result
