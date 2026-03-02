from __future__ import annotations
import numpy as np
import math

def adstock_geometric(x: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock: a_t = x_t + decay*a_{t-1}."""
    a = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        a[t] = x[t] + (decay * a[t-1] if t > 0 else 0.0)
    return a

def half_life(decay: float) -> float:
    if not (0 < decay < 1):
        return float('nan')
    return math.log(0.5) / math.log(decay)
