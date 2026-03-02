from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class Constraints:
    total_budget: float
    min_by_channel: Dict[str, float]
    max_by_channel: Dict[str, float]
    prev_by_channel: Optional[Dict[str, float]] = None
    delta: float = 0.0
    ramp_up_cap_share: float = 0.0
    ramp_up_cap_abs: float = 0.0

def default_constraints(channels: list[str], total_budget: float) -> Constraints:
    return Constraints(
        total_budget=float(total_budget),
        min_by_channel={c: 0.0 for c in channels},
        max_by_channel={c: float(total_budget) for c in channels},
        prev_by_channel=None,
        delta=0.0,
        ramp_up_cap_share=0.0,
        ramp_up_cap_abs=0.0,
    )
