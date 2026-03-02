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
from apps.api.deps import get_paths
from mmx.usecases.build_mart import build_daily_channel_fact
from mmx.usecases.train_model import train_model
from mmx.config.settings import load_settings

router = APIRouter(tags=['pipeline'])

@router.post('/pipeline/build-mart')
def build_mart():
    s = load_settings()
    paths = get_paths()
    return build_daily_channel_fact(paths, s).__dict__

@router.post('/pipeline/train')
def train():
    s = load_settings()
    paths = get_paths()
    res = train_model(
        paths,
        backend=s.inference_backend,
        method=s.inference_method,
        reporting_delay=s.reporting_delay,
        reporting_delay_max=s.reporting_delay_max,
    )
    return res.__dict__
