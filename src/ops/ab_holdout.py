from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class HoldoutConfig:
    holdout_channels: List[str]
    delta: float = 0.20  # +/- 20%


def _stable_hash_int(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def assign_weekly_groups(date_str: str, channels: List[str], holdout_channels: List[str], seed: int = 42) -> Dict[str, str]:
    """Assign CONTROL/TREATMENT for holdout channels, stable within a week.

    date_str: YYYY-MM-DD
    """
    # Use ISO week to keep stable for 7 days
    import datetime as _dt

    d = _dt.date.fromisoformat(date_str)
    iso_year, iso_week, _ = d.isocalendar()
    groups: Dict[str, str] = {}
    for ch in channels:
        if ch not in holdout_channels:
            groups[ch] = "NA"
            continue
        key = _stable_hash_int(str(seed), ch, str(iso_year), str(iso_week))
        groups[ch] = "TREATMENT" if (key % 2 == 0) else "CONTROL"
    return groups


def apply_holdout_multipliers(
    budgets: Dict[str, float],
    groups: Dict[str, str],
    delta: float,
    min_share: float,
    max_share: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Apply +/-delta to holdout channels and renormalize back to original total.

    Returns (new_budgets, multipliers).
    """
    total = float(sum(budgets.values()))
    new = {k: float(v) for k, v in budgets.items()}
    mults: Dict[str, float] = {k: 1.0 for k in budgets}

    for ch, g in groups.items():
        if g == "TREATMENT":
            mults[ch] = 1.0 + float(delta)
            new[ch] *= mults[ch]
        elif g == "CONTROL":
            mults[ch] = max(1e-9, 1.0 - float(delta))
            new[ch] *= mults[ch]

    # renormalize to keep total spend constant
    s = sum(new.values())
    if s <= 0:
        return budgets, mults
    scale = total / s
    for ch in new:
        new[ch] *= scale

    # enforce bounds and renormalize iteratively (simple projection)
    channels = list(new.keys())
    for _ in range(5):
        total = float(sum(new.values()))
        changed = False
        for ch in channels:
            mn = min_share * total
            mx = max_share * total
            if new[ch] < mn:
                new[ch] = mn
                changed = True
            elif new[ch] > mx:
                new[ch] = mx
                changed = True
        if not changed:
            break
        # renormalize back to original total
        s2 = sum(new.values())
        if s2 > 0:
            k = total / s2
            for ch in new:
                new[ch] *= k

    # final fix for numeric drift
    s3 = sum(new.values())
    if s3 > 0:
        k = (sum(budgets.values())) / s3
        for ch in new:
            new[ch] *= k

    return new, mults
