from __future__ import annotations

"""Shared transforms for the synthetic data generator."""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def adstock(spend: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    """Geometric adstock.

    Parameters
    ----------
    spend:
        Daily spend series.
    theta:
        Carryover decay in (0,1). Larger means longer memory.
    """

    out = np.zeros_like(spend, dtype=float)
    carry = 0.0
    t = float(theta)
    for i, x in enumerate(spend.astype(float)):
        carry = float(x) + t * carry
        out[i] = carry
    return out


def hill_saturation(x: NDArray[np.float64], alpha: float, k: float) -> NDArray[np.float64]:
    """Hill-type saturation curve in [0,1]."""

    a = float(alpha)
    kk = float(max(k, 1e-9))
    xa = np.power(np.clip(x, 0.0, None), a)
    ka = np.power(kk, a)
    return xa / (xa + ka)


def apply_lag_kernel(x: NDArray[np.float64], kernel: Sequence[float]) -> NDArray[np.float64]:
    """Apply a discrete lag kernel (convolution) to create delayed effects."""

    w = np.asarray(list(kernel), dtype=float)
    if w.size == 0:
        return x.astype(float)
    w = w / max(1e-12, float(np.sum(w)))
    # causal conv: y[t] = sum_{k=0..K-1} w[k] * x[t-k]
    y = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        s = 0.0
        for k in range(len(w)):
            j = t - k
            if j < 0:
                break
            s += float(w[k]) * float(x[j])
        y[t] = s
    return y
