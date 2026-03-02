from __future__ import annotations
# MMX_SYS_PATH_GUARD: ensure repo root/src are importable when running this script directly
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / 'src'
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
from mmx.config.settings import load_settings
from mmx.data.paths import resolve_paths
from mmx.domain.types import Policy, ObjectiveMode
from mmx.governance.registry import load_registry
from mmx.usecases.recommend_budget import recommend_budget

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--period-start', required=True)
    ap.add_argument('--period-end', required=True)
    ap.add_argument('--total-budget', required=True, type=float)
    args = ap.parse_args()

    s = load_settings()
    p = resolve_paths(s)
    reg = load_registry(p)
    if reg.production_version == 'none':
        raise SystemExit('No production model. Run promote_model.py first.')

    policy = Policy(
        objective_mode=ObjectiveMode.RISK_ADJUSTED,
        policy_lambda=s.policy_lambda,
        policy_delta=s.policy_delta,
    )

    res = recommend_budget(
        p,
        args.period_start,
        args.period_end,
        args.total_budget,
        model_version=reg.production_version,
        policy=policy,
        ramp_up_cap_share=s.ramp_up_cap_share,
        ramp_up_cap_abs=s.ramp_up_cap_abs,
    )
    print('decision_json:', res.decision_json_path)
    print('risk_adjusted_premium:', res.decision.risk_adjusted_premium)
