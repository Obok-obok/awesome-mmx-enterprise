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
from typing import Optional, Dict
from apps.api.deps import get_paths
from mmx.config.settings import load_settings
from mmx.domain.types import Policy, ObjectiveMode
from mmx.governance.registry import load_registry
from mmx.usecases.recommend_budget import recommend_budget

router = APIRouter(tags=['recommend'])

class RecommendIn(BaseModel):
    period_start: str
    period_end: str
    total_budget: float
    experiment_id: Optional[str] = None
    prev_budget_by_channel: Optional[Dict[str, float]] = None

@router.post('/recommend')
def recommend(inp: RecommendIn):
    s = load_settings()
    paths = get_paths()
    reg = load_registry(paths)
    policy = Policy(ObjectiveMode.RISK_ADJUSTED, s.policy_lambda, s.policy_delta)
    res = recommend_budget(paths, inp.period_start, inp.period_end, inp.total_budget, reg.production_version, policy, inp.experiment_id, inp.prev_budget_by_channel)
    return {'decision': res.decision.__dict__, 'decision_json_path': res.decision_json_path}
