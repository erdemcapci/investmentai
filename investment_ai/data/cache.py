from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

class JsonCache:
    """Atomic success-only cache: a failed refresh never destroys good data."""
    def __init__(self, root: Path): self.root=Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def _path(self, tier: str, symbol: str) -> Path:
        safe=symbol.replace('/','_'); p=self.root/tier; p.mkdir(parents=True, exist_ok=True); return p/f"{safe}.json"
    def read(self,tier:str,symbol:str)->dict[str,Any]|None:
        p=self._path(tier,symbol)
        try: return json.loads(p.read_text())
        except (OSError,json.JSONDecodeError): return None
    def fresh(self, item:dict[str,Any]|None, ttl_hours:float)->bool:
        if not item or not item.get("fetched_at_utc"): return False
        try: age=(datetime.now(timezone.utc)-datetime.fromisoformat(item["fetched_at_utc"])).total_seconds()/3600
        except (ValueError,TypeError): return False
        return age <= ttl_hours
    def get_or_fetch(self,tier:str,symbol:str,ttl_hours:float,fetch:Callable[[],dict[str,Any]],force=False):
        old=self.read(tier,symbol)
        if not force and self.fresh(old,ttl_hours): return old, "cache"
        try:
            data=fetch()
            if not data: raise ValueError("empty provider response")
            item={"fetched_at_utc":datetime.now(timezone.utc).isoformat(),"data":data}
            p=self._path(tier,symbol); tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(item,default=str)); tmp.replace(p)
            return item,"provider"
        except Exception as exc:
            if old: return old,f"stale_cache_after_error:{exc}"
            return {"fetched_at_utc":None,"data":{},"error":str(exc)},"error"
