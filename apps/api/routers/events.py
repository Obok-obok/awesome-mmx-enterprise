from __future__ import annotations
# MMX_SYS_PATH_GUARD: ensure repo root/src are importable when running via uvicorn or direct import
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / 'src'
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
from pathlib import Path
from apps.api.deps import get_paths

router = APIRouter(tags=['events'])

class EventIn(BaseModel):
    event_type: str
    payload: Dict[str, Any]

@router.post('/events')
def ingest_event(ev: EventIn):
    paths = get_paths()
    base = paths.data_raw / f'events/{ev.event_type}'
    base.mkdir(parents=True, exist_ok=True)
    # append per-day shard file
    ts = ev.payload.get('event_time') or ev.payload.get('occurred_at') or ev.payload.get('attempt_at') or ev.payload.get('connected_at') or ev.payload.get('contract_signed_at') or ev.payload.get('premium_recognized_at')
    day = pd.to_datetime(ts).strftime('%Y-%m-%d') if ts else 'unknown'
    fp = base / f'{day}.csv'
    df = pd.DataFrame([ev.payload])
    if fp.exists():
        df.to_csv(fp, mode='a', header=False, index=False)
    else:
        df.to_csv(fp, index=False)
    return {'status': 'ok', 'file': str(fp)}
