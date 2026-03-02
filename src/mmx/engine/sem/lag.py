from __future__ import annotations
import numpy as np

def apply_lag_kernel(x: np.ndarray, w: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    y = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        s = 0.0
        for k in range(max_lag+1):
            if t-k >= 0:
                s += w[k] * x[t-k]
        y[t] = s
    return y

def default_kernel(max_lag: int, decay: float = 0.5) -> np.ndarray:
    w = np.array([decay**k for k in range(max_lag+1)], dtype=float)
    return w / (w.sum() + 1e-9)
