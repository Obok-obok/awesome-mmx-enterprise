from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List
import numpy as np
from scipy.optimize import minimize
from mmx.optimization.constraints import Constraints

@dataclass(frozen=True)
class SolveResult:
    budget: Dict[str, float]
    success: bool
    message: str

def solve_slsqp(channels: List[str], constraints: Constraints, objective_fn: Callable[[Dict[str, float]], float]) -> SolveResult:
    n = len(channels)

    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = np.clip(x, 0.0, None)
        s = float(np.sum(x))
        if s <= 0:
            return np.array([constraints.total_budget / max(1, n)] * n, dtype=float)
        return x * (constraints.total_budget / s)

    # Multi-start to avoid getting stuck at a symmetric/flat initial point.
    # Start points: uniform, (optional) prev allocation, and a few random perturbations.
    starts: List[np.ndarray] = []
    starts.append(np.array([constraints.total_budget / max(1, n)] * n, dtype=float))
    if constraints.prev_by_channel:
        prev = np.array([float(constraints.prev_by_channel.get(c, 0.0)) for c in channels], dtype=float)
        if float(np.sum(prev)) > 0:
            starts.append(_normalize(prev))
    rng = np.random.default_rng(17)
    for _ in range(4):
        z = rng.lognormal(mean=0.0, sigma=0.7, size=n)
        starts.append(_normalize(z))

    # Apply optional ramp-up caps for channels with prev=0 (risk control).
    prev = constraints.prev_by_channel
    cap_share = float(constraints.ramp_up_cap_share)
    cap_abs = float(constraints.ramp_up_cap_abs)
    share_cap_value = constraints.total_budget * cap_share if cap_share > 0 else 0.0

    bounds = []
    for c in channels:
        lo = float(constraints.min_by_channel[c])
        hi = float(constraints.max_by_channel[c])
        if prev is not None and float(prev.get(c, 0.0)) <= 0.0:
            cap = 0.0
            if cap_abs > 0:
                cap = cap_abs
            if share_cap_value > 0:
                cap = cap if cap > 0 else share_cap_value
                cap = min(cap, share_cap_value)
            if cap > 0:
                hi = min(hi, cap)
        bounds.append((lo, hi))

    cons = [{'type': 'eq', 'fun': lambda x: float(np.sum(x) - constraints.total_budget)}]

    if constraints.prev_by_channel and constraints.delta > 0:
        prev = constraints.prev_by_channel
        delta = float(constraints.delta)
        for i, ch in enumerate(channels):
            p = float(prev.get(ch, 0.0))
            if p > 0:
                lim = delta * p
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, p=p, lim=lim: (p + lim) - x[i]})
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, p=p, lim=lim: x[i] - (p - lim)})

    def fun(x: np.ndarray) -> float:
        plan = {channels[i]: float(x[i]) for i in range(n)}
        return -float(objective_fn(plan))

    best_x: np.ndarray | None = None
    best_val: float = float("inf")
    best_res = None
    for x0 in starts:
        res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 400, 'ftol': 1e-9})
        val = float(res.fun) if np.isfinite(res.fun) else float("inf")
        if val < best_val:
            best_val = val
            best_res = res
            best_x = res.x

    if best_x is None:
        best_x = starts[0]
        ok = False
        msg = "optimization failed: no result"
    else:
        ok = bool(best_res.success)
        msg = str(best_res.message)

    plan = {channels[i]: float(best_x[i]) for i in range(n)}
    return SolveResult(plan, ok, msg)
