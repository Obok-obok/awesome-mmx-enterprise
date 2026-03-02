from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class BudgetBounds:
    min_share: float = 0.05
    max_share: float = 0.80


def normalize_with_bounds(weights: np.ndarray, bounds: BudgetBounds) -> np.ndarray:
    w = np.clip(weights, 1e-9, None)
    w = w / w.sum()

    # Project to box-simplex approximately by iterative clipping + renormalization
    lo, hi = bounds.min_share, bounds.max_share
    for _ in range(30):
        w = np.clip(w, lo, hi)
        w = w / w.sum()
        if np.all((w >= lo - 1e-6) & (w <= hi + 1e-6)):
            break
    return w


def recommend_budget_thompson(
    channels: List[str],
    total_budget: float,
    score_samples: Dict[str, np.ndarray],
    bounds: BudgetBounds,
) -> Dict[str, float]:
    """Thompson-style allocation.

    score_samples[ch]: samples of (expected revenue per 1 unit spend) or any comparable score.
    We draw one sample per channel, allocate proportionally, then enforce bounds.
    """
    rng = np.random.default_rng(0)
    draws = np.array([float(rng.choice(score_samples[ch])) for ch in channels])
    draws = np.clip(draws, 1e-9, None)
    w = normalize_with_bounds(draws, bounds)
    return {ch: float(total_budget * w[i]) for i, ch in enumerate(channels)}


def recommend_budget_thompson_explore(
    channels: List[str],
    total_budget: float,
    score_samples: Dict[str, np.ndarray],
    bounds: BudgetBounds,
    exploration_eps: float = 0.10,
    temperature: float = 1.0,
    seed: int = 0,
) -> Dict[str, float]:
    """Ops-friendly adaptive allocator.

    - (1-eps) budget: Thompson-style exploitation
    - eps budget: random exploration (uniform weights)
    - temperature: softens/hardens exploitation weights
    """
    rng = np.random.default_rng(seed)
    draws = np.array([float(rng.choice(score_samples[ch])) for ch in channels])
    draws = np.clip(draws, 1e-9, None)
    # temperature on log-scale to avoid extreme allocations
    draws = np.exp(np.log(draws) / max(temperature, 1e-6))
    w_exploit = normalize_with_bounds(draws, bounds)
    w_explore = np.ones(len(channels), dtype=float) / float(len(channels))
    eps = float(np.clip(exploration_eps, 0.0, 0.5))
    w = (1.0 - eps) * w_exploit + eps * w_explore
    w = normalize_with_bounds(w, bounds)
    return {ch: float(total_budget * w[i]) for i, ch in enumerate(channels)}
