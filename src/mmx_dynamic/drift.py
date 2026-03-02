from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriftConfig:
    # z-score threshold for residual drift alert
    z_threshold: float = 3.0
    # rolling window for residual mean/std
    window: int = 21
    # if drift persists for N consecutive days -> hard drift
    persist_days: int = 3


class ResidualDriftDetector:
    """Simple ops-safe drift detector.

    Detects drift on a single scalar residual series using rolling z-score.
    """

    def __init__(self, cfg: DriftConfig):
        self.cfg = cfg
        self.residuals: list[float] = []
        self._persist = 0

    def update(self, resid: float) -> dict:
        self.residuals.append(float(resid))
        w = self.cfg.window
        if len(self.residuals) < max(10, w):
            return {"drift": False, "z": 0.0, "persist": 0}

        tail = self.residuals[-w:]
        mean = sum(tail) / len(tail)
        var = sum((x - mean) ** 2 for x in tail) / max(1, (len(tail) - 1))
        std = (var ** 0.5) if var > 1e-12 else 1e-6
        z = (tail[-1] - mean) / std

        drift_today = abs(z) >= self.cfg.z_threshold
        if drift_today:
            self._persist += 1
        else:
            self._persist = 0

        hard = self._persist >= self.cfg.persist_days
        return {"drift": bool(drift_today), "hard_drift": bool(hard), "z": float(z), "persist": int(self._persist)}
