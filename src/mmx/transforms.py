from __future__ import annotations
import numpy as np

def adstock(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + alpha * carry
        out[i] = carry
    return out

def saturation(x: np.ndarray, k: float) -> np.ndarray:
    return np.log1p(k * np.maximum(x, 0.0))
