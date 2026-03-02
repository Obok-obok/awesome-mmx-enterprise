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
from mmx.usecases.data_quality import detect_weekend_reporting_delay

router = APIRouter(tags=['monitoring'])

@router.get('/monitoring/data-quality/reporting-delay')
def reporting_delay_signal():
    paths = get_paths()
    res = detect_weekend_reporting_delay(paths)
    return res.__dict__
