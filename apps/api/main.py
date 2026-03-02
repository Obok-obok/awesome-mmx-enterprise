from __future__ import annotations
# MMX_SYS_PATH_GUARD: ensure repo root/src are importable when running via uvicorn or direct import
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / 'src'
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI
from apps.api.routers import events, pipeline, recommend, evaluate, monitoring

app = FastAPI(title='MMx Enterprise API', version='2.0')

app.include_router(events.router, prefix='/v1')
app.include_router(pipeline.router, prefix='/v1')
app.include_router(recommend.router, prefix='/v1')
app.include_router(evaluate.router, prefix='/v1')
app.include_router(monitoring.router, prefix='/v1')
