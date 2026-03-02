from __future__ import annotations
from dataclasses import dataclass
from typing import List
from mmx.domain.types import RolloutMode

@dataclass(frozen=True)
class RolloutDecision:
    mode: RolloutMode
    allowed_ai_share: float
    reason_codes: List[str]

def apply_rollout_policy(p_ai_better: float, data_coverage: float) -> RolloutDecision:
    if data_coverage < 0.7:
        return RolloutDecision(RolloutMode.CONSERVATIVE, 0.1, ['COVERAGE_LOW'])
    if p_ai_better < 0.6:
        return RolloutDecision(RolloutMode.CONSERVATIVE, 0.1, ['P_AI_SUPERIOR_LOW'])
    if p_ai_better < 0.75:
        return RolloutDecision(RolloutMode.NORMAL, 0.2, ['P_AI_SUPERIOR_OK'])
    return RolloutDecision(RolloutMode.NORMAL, 0.3, ['P_AI_SUPERIOR_HIGH'])
