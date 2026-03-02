from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ObjectiveResult:
    expected: float
    std: float
    ci_low: float
    ci_high: float
    risk_adjusted: float

def compute(samples: np.ndarray, policy_lambda: float) -> ObjectiveResult:
    s = np.asarray(samples, dtype=float)
    expected = float(np.mean(s))
    std = float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
    ci_low = float(np.quantile(s, 0.025))
    ci_high = float(np.quantile(s, 0.975))
    ra = expected - policy_lambda * std
    return ObjectiveResult(expected, std, ci_low, ci_high, float(ra))
