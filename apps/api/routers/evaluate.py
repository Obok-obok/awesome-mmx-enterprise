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
from typing import Dict
from apps.api.deps import get_paths
from mmx.config.settings import load_settings
from mmx.governance.registry import load_registry
from mmx.usecases.evaluate_shadow import evaluate_shadow

router = APIRouter(tags=['evaluate'])

class ShadowIn(BaseModel):
    ai_plan: Dict[str, float]
    human_plan: Dict[str, float]

@router.post('/evaluate/shadow')
def shadow(inp: ShadowIn):
    s = load_settings()
    paths = get_paths()
    reg = load_registry(paths)
    res = evaluate_shadow(paths, reg.production_version, inp.ai_plan, inp.human_plan, policy_lambda=s.policy_lambda)
    return res.__dict__
