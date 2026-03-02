from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class ChannelConstraint:
    """Per-channel budget constraints relative to current spend."""

    lock: bool = False
    min_ratio: float = 0.0
    max_ratio: float = 10.0


@dataclass(frozen=True)
class OptResult:
    recommended_spend: Dict[str, float]
    predicted_value: float
    predicted_premium: float
    predicted_leads: float
    predicted_contracts: float
    budget_cap: float
    unspent_budget: float
    target_premium: float
    reached_target: bool
    objective: str


def _finite_diff_marginal(
    pred_premium: Callable[[str, float], float],
    ch: str,
    x: float,
    step: float,
) -> float:
    """Approximate dPremium/dSpend with a safe finite difference."""
    x0 = max(0.0, float(x))
    x1 = max(0.0, float(x0 + step))
    p0 = float(pred_premium(ch, x0))
    p1 = float(pred_premium(ch, x1))
    return max(0.0, (p1 - p0) / max(1e-9, step))


def optimize_min_spend_for_target(
    *,
    channels: Iterable[str],
    current_spend: Dict[str, float],
    constraints: Dict[str, ChannelConstraint],
    pred_leads: Callable[[str, float], float],
    pred_contracts: Callable[[str, float], float],
    pred_premium: Callable[[str, float], float],
    premium_target: float,
    step: float,
    budget_cap: float,
) -> OptResult:
    """Greedy min-cost allocation to reach Premium target.

    - Start from per-channel minimums (or lock at current)
    - Iteratively add `step` to the channel with max marginal premium per spend
    - Stop when target reached or budget/max constraints exhausted

    This is intentionally simple + deterministic for operational explainability.
    """

    chs = list(channels)
    step = float(max(1.0, step))
    budget_cap = float(max(0.0, budget_cap))
    premium_target = float(max(0.0, premium_target))

    # build bounds
    lb: Dict[str, float] = {}
    ub: Dict[str, float] = {}
    for ch in chs:
        cur = float(max(0.0, current_spend.get(ch, 0.0)))
        c = constraints.get(ch, ChannelConstraint())
        if c.lock:
            lb[ch] = cur
            ub[ch] = cur
        else:
            lb[ch] = max(0.0, cur * float(max(0.0, c.min_ratio)))
            ub[ch] = max(lb[ch], cur * float(max(lb[ch] / max(cur, 1e-9), c.max_ratio)))

    # start at lower bounds
    x: Dict[str, float] = {ch: float(lb[ch]) for ch in chs}
    spend_sum = float(sum(x.values()))
    # If lower bounds exceed cap, scale down unlocked channels proportionally (locks remain)
    if spend_sum > budget_cap and spend_sum > 0:
        locked = [ch for ch in chs if lb[ch] == ub[ch]]
        unlocked = [ch for ch in chs if ch not in locked]
        locked_sum = float(sum(x[ch] for ch in locked))
        remaining_cap = max(0.0, budget_cap - locked_sum)
        unlocked_sum = float(sum(x[ch] for ch in unlocked))
        if unlocked_sum > 0:
            scale = min(1.0, remaining_cap / unlocked_sum)
            for ch in unlocked:
                x[ch] = x[ch] * scale
        spend_sum = float(sum(x.values()))

    def total_premium(xx: Dict[str, float]) -> float:
        return float(sum(pred_premium(ch, float(xx[ch])) for ch in chs))

    # Greedy allocate until target or cap
    cur_prem = total_premium(x)
    while cur_prem + 1e-9 < premium_target:
        # find best marginal
        best = None
        for ch in chs:
            if x[ch] + step > ub[ch] + 1e-9:
                continue
            if spend_sum + step > budget_cap + 1e-9:
                continue
            m = _finite_diff_marginal(pred_premium, ch, x[ch], step)
            if best is None or m > best[0]:
                best = (m, ch)
        if best is None or best[0] <= 0:
            break
        _, ch_star = best
        x[ch_star] = float(min(ub[ch_star], x[ch_star] + step))
        spend_sum = float(sum(x.values()))
        cur_prem = total_premium(x)

    # predicted funnel outputs for reporting
    pred_leads_tot = float(sum(pred_leads(ch, x[ch]) for ch in chs))
    pred_contracts_tot = float(sum(pred_contracts(ch, x[ch]) for ch in chs))
    pred_prem_tot = float(sum(pred_premium(ch, x[ch]) for ch in chs))

    return OptResult(
        recommended_spend={ch: float(x[ch]) for ch in chs},
        predicted_value=pred_prem_tot,  # value is premium by definition in this mode
        predicted_premium=pred_prem_tot,
        predicted_leads=pred_leads_tot,
        predicted_contracts=pred_contracts_tot,
        budget_cap=budget_cap,
        unspent_budget=float(max(0.0, budget_cap - spend_sum)),
        target_premium=premium_target,
        reached_target=bool(pred_prem_tot + 1e-9 >= premium_target),
        objective="MIN_SPEND_FOR_TARGET_PREMIUM",
    )
