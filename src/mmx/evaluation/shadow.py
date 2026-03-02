from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ShadowResult:
    delta_ra_mean: float
    p_ai_better: float
    ci_low: float
    ci_high: float

def shadow_eval(ra_ai: np.ndarray, ra_human: np.ndarray) -> ShadowResult:
    delta = ra_ai - ra_human
    return ShadowResult(
        delta_ra_mean=float(np.mean(delta)),
        p_ai_better=float(np.mean(delta > 0)),
        ci_low=float(np.quantile(delta, 0.025)),
        ci_high=float(np.quantile(delta, 0.975)),
    )
