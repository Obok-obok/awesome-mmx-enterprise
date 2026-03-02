from __future__ import annotations
import numpy as np
from mmx.engine.sem.lag import apply_lag_kernel, default_kernel

def apply_reporting_delay(leads_true: np.ndarray, max_delay: int = 3, decay: float = 0.6) -> np.ndarray:
    w = default_kernel(max_delay, decay=decay)
    return apply_lag_kernel(leads_true, w, max_lag=max_delay)
