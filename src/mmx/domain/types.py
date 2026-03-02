from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class ObjectiveMode(str, Enum):
    RISK_ADJUSTED = 'RISK_ADJUSTED'
    RISK_NEUTRAL = 'RISK_NEUTRAL'

class RolloutMode(str, Enum):
    NORMAL = 'NORMAL'
    CONSERVATIVE = 'CONSERVATIVE'
    STOP = 'STOP'

@dataclass(frozen=True)
class Policy:
    objective_mode: ObjectiveMode
    policy_lambda: float
    policy_delta: float
