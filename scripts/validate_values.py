#!/usr/bin/env python3
"""Deep value validation for MMX Enterprise.

This script validates that:
- Period length (n_days) is consistent with period_start/period_end.
- Warm-start (carryover/backlog) changes predictions when there is historical spend.
- Funnel invariants hold (premium equals contracts * ppc sample when sigma=0).

It is designed to be runnable without a trained model (uses a synthetic posterior).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from mmx.engine.sem.inference import PosteriorSummary, posterior_predict_premium, posterior_predict_funnel


def _kernel(L: int, decay: float = 0.7) -> str:
    w = np.array([decay**k for k in range(L + 1)], dtype=float)
    w = w / (w.sum() + 1e-9)
    return json.dumps([float(x) for x in w.tolist()])


def make_synthetic_posterior(channels: list[str]) -> PosteriorSummary:
    params: Dict[str, Dict[str, Any]] = {}
    for ch in channels:
        params[ch] = {
            "decay": 0.9,
            "ec50": 50.0,
            "hill": 1.2,
            "alpha_lead": 80.0,
            "rate_attempt_per_lead": 0.8,
            "rate_connected_per_attempt": 0.5,
            "rate_contract_per_connected": 0.25,
            "w_lead_to_attempt": _kernel(7, decay=0.6),
            "w_connected_to_contract": _kernel(14, decay=0.7),
        }
    globals_ = {
        "mu_log_premium_per_contract": float(np.log(1000.0)),
        "sigma_log_premium_per_contract": 0.0,  # force deterministic premium-per-contract for invariant test
    }
    return PosteriorSummary(model_version="synthetic", channel_params=params, globals=globals_)


def make_history_df(channel: str, start: str, days: int, spend: float) -> pd.DataFrame:
    dts = pd.date_range(start=pd.to_datetime(start), periods=days, freq="D")
    return pd.DataFrame(
        {
            "date": dts,
            "channel": [channel] * len(dts),
            "spend": [spend] * len(dts),
            "leads": [0.0] * len(dts),
            "call_attempt": [0.0] * len(dts),
            "call_connected": [0.0] * len(dts),
            "contracts": [0.0] * len(dts),
            "premium": [0.0] * len(dts),
        }
    )


def main() -> None:
    # --- Setup synthetic inputs
    channels = ["kakao"]
    posterior = make_synthetic_posterior(channels)

    # History window: 40 days of spend prior to the forecast period
    hist = make_history_df("kakao", start="2026-01-01", days=40, spend=100.0)
    period_start = pd.Timestamp("2026-02-10")
    period_end = pd.Timestamp("2026-02-24")
    n_days = int((period_end - period_start).days) + 1
    assert n_days == 15, f"n_days expected 15, got {n_days}"

    # Future plan: 0 spend in period -> carryover should still produce some premium if warm-start is enabled
    plan = {"kakao": 0.0}

    s_warm = posterior_predict_premium(
        hist,
        posterior,
        plan,
        period_start=period_start,
        n_days=n_days,
        warm_start=True,
        n_samples=2000,
        seed=1,
    )
    s_cold = posterior_predict_premium(
        hist,
        posterior,
        plan,
        period_start=period_start,
        n_days=n_days,
        warm_start=False,
        n_samples=2000,
        seed=1,
    )

    warm_mean = float(np.mean(s_warm))
    cold_mean = float(np.mean(s_cold))
    assert warm_mean > 0.0, "Warm-start mean premium should be > 0 when history exists."
    assert cold_mean == 0.0, "Cold-start mean premium should be 0 when period spend is 0."

    # Funnel invariant: premium = contracts * ppc, and since sigma=0, premium std should be 0 given deterministic pipeline
    funnel = posterior_predict_funnel(
        posterior,
        {"kakao": 1000.0},
        df_daily=hist,
        period_start=period_start,
        n_days=n_days,
        warm_start=True,
        n_samples=500,
        seed=2,
    )
    # sigma=0 => premium_per_contract constant => premium samples vary only by ctr_total (deterministic in this simplified predictor)
    prem = funnel["premium"]
    ctr = funnel["contracts"]
    ratio = prem / np.maximum(ctr, 1e-9)
    # ratio should be constant ~ exp(mu)=1000, except when ctr is 0
    nonzero = ctr > 1e-6
    if np.any(nonzero):
        assert float(np.std(ratio[nonzero])) < 1e-6, "Premium/contract ratio should be constant when sigma=0."

    print("PASS: period-length sync, warm-start effect, and funnel invariant checks.")
    print(f"Warm-start premium mean={warm_mean:,.2f} vs Cold-start mean={cold_mean:,.2f}")


if __name__ == "__main__":
    main()
