from __future__ import annotations

"""Retention-based LTV helper.

Goal
----
Replace naive LTV = premium * factor with a simple retention/churn model that:
- uses per-channel monthly retention (or churn) if provided
- falls back to global defaults if not

Interpretation
--------------
We treat observed `premium` as the first-period premium from newly written contracts.
Then expected LTV is approximated as:

  LTV_total ≈ premium_total * annuity_factor

where

  annuity_factor = Σ_{m=0..H-1} (retention^m) / (1+discount)^m

This is a lightweight proxy suitable for ops monitoring.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class RetentionLTVConfig:
    horizon_months: int = 36
    monthly_retention_default: float = 0.92   # r
    monthly_discount_default: float = 0.01    # d
    retention_path: str = "data/input_channel_retention.csv"  # optional


def _annuity_factor(r: float, d: float, H: int) -> float:
    r = float(np.clip(r, 0.0, 0.999999))
    d = float(max(d, 0.0))
    H = int(max(H, 1))
    # Σ (r/(1+d))^m
    q = r / (1.0 + d)
    if abs(1.0 - q) < 1e-10:
        return float(H)
    return float((1.0 - q**H) / (1.0 - q))


def load_channel_retention_table(path: str) -> Optional[pd.DataFrame]:
    """Optional table with columns: channel, monthly_retention, monthly_discount (optional)."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    need = {"channel", "monthly_retention"}
    if not need.issubset(set(df.columns)):
        return None
    out = df.copy()
    if "monthly_discount" not in out.columns:
        out["monthly_discount"] = np.nan
    out["channel"] = out["channel"].astype(str)
    return out


def compute_ltv_total(df: pd.DataFrame, cfg: RetentionLTVConfig = RetentionLTVConfig()) -> pd.DataFrame:
    """Return df with an added/overwritten 'ltv' column."""
    g = df.copy()
    if "premium" not in g.columns:
        g["ltv"] = 0.0
        return g

    tab = load_channel_retention_table(str(getattr(cfg, "retention_path", "data/input_channel_retention.csv")))
    if tab is not None:
        tab = tab.set_index("channel")

    H = int(getattr(cfg, "horizon_months", 36))
    r0 = float(getattr(cfg, "monthly_retention_default", 0.92))
    d0 = float(getattr(cfg, "monthly_discount_default", 0.01))

    # Precompute per-channel factors if possible
    factors: Dict[str, float] = {}

    if "channel" in g.columns and tab is not None:
        for ch in g["channel"].astype(str).unique().tolist():
            if ch in tab.index:
                r = float(tab.loc[ch, "monthly_retention"])
                d = tab.loc[ch, "monthly_discount"]
                d = d0 if (pd.isna(d)) else float(d)
            else:
                r, d = r0, d0
            factors[ch] = _annuity_factor(r, d, H)

        g["ltv"] = g.apply(lambda row: float(row.get("premium", 0.0)) * float(factors.get(str(row.get("channel","")), _annuity_factor(r0, d0, H))), axis=1)
    else:
        fac = _annuity_factor(r0, d0, H)
        g["ltv"] = g["premium"].astype(float) * float(fac)

    return g
