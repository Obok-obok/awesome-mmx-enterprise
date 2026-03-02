from __future__ import annotations
import numpy as np

def hill(x: np.ndarray, alpha: float, ec50: float, hill: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    num = alpha * np.power(np.clip(x, 0, None), hill)
    den = np.power(np.clip(x, 0, None), hill) + (ec50 ** hill)
    return num / (den + 1e-9)

def saturation_ratio(x: float, ec50: float) -> float:
    return float(x / (x + ec50)) if (x + ec50) > 0 else 0.0
