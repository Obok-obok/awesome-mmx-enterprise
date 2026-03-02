from __future__ import annotations

"""Geo holdout + matched market utilities.

This module implements an ops-friendly geo holdout experiment layer.

v5 upgrades
-----------
1) Assignment is no longer limited to *pairs*. We support:
   - 1 treated geo + multiple control geos (true synthetic control).
2) Matching quality filters include:
   - R² (pre-fit)
   - RMSE (pre-fit)
   - MAPE (pre-fit)
3) Matching is ridge-weighted synthetic control (fast approximation):
   - Fit donor weights with ridge regression
   - Project to non-negative simplex (weights >= 0, sum = 1)

Expected input (optional)
-------------------------
data/input_daily_geo_channel.csv with columns:
  date, geo, channel, spend, leads, tm_attempts, tm_connected, contracts, premium
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class GeoHoldoutConfig:
    enabled: bool = False
    holdout_channels: List[str] | None = None
    delta: float = 0.15  # +/- 15% within holdout channels
    min_geo_share: float = 0.02  # within-channel geo share floor (after renorm)
    max_geo_share: float = 0.80

    # assignment
    seed: int = 42
    treated_frac: float = 0.50  # share of eligible geos to treat (weekly-stable)
    min_control_geos: int = 5   # require at least this many control geos for synth

    # matching / synth control
    prewindow_days: int = 56
    ridge_lambda: float = 1.0
    mm_max_geos: int = 40  # cap geos for matching to save memory

    # pre-fit quality filters
    prefit_min_r2: float = 0.50
    prefit_max_rmse: float = 1e18
    prefit_max_mape: float = 1e18  # in fraction (0.20 = 20%)


def _stable_hash_int(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _project_to_simplex_nonneg(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        return np.ones_like(w) / float(len(w))
    return w / s


def _project_simplex_with_bounds(w: np.ndarray, lo: float, hi: float) -> np.ndarray:
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    for _ in range(50):
        w2 = np.clip(w, lo, hi)
        w2 = w2 / w2.sum()
        if np.max(np.abs(w2 - w)) < 1e-9:
            w = w2
            break
        w = w2
    return w


def assign_geo_groups_weekly_unpaired(date_str: str, geos: List[str], seed: int = 42, treated_frac: float = 0.50) -> Dict[str, str]:
    """Weekly-stable assignment CONTROL/TREATMENT without requiring pairs.

    Uses a stable hash per (geo, iso_year, iso_week) so the assignment is
    deterministic and repeatable.
    """
    import datetime as _dt

    if not geos:
        return {}
    d = _dt.date.fromisoformat(date_str)
    iso_year, iso_week, _ = d.isocalendar()

    scores = []
    for g in sorted(set(geos)):
        key = _stable_hash_int(str(seed), g, str(iso_year), str(iso_week))
        scores.append((key, g))
    scores.sort()  # deterministic ordering

    n = len(scores)
    k = int(round(float(treated_frac) * n))
    k = max(1, min(n - 1, k))  # ensure both groups exist

    treat = set([g for _, g in scores[:k]])
    groups = {g: ("TREATMENT" if g in treat else "CONTROL") for _, g in scores}
    return groups


def _ridge_synth_weights_matrix(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Ridge weights -> non-negative simplex projection."""
    XtX = X.T @ X
    Xty = X.T @ y
    A = XtX + float(ridge_lambda) * np.eye(X.shape[1])
    try:
        w = np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, Xty, rcond=None)[0]
    return _project_to_simplex_nonneg(w)


def _fit_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / max(1, len(y))))
    denom = np.clip(np.abs(y), 1e-6, None)
    mape = float(np.mean(np.abs(y - yhat) / denom))
    return {"r2": float(r2), "rmse": float(rmse), "mape": float(mape)}


def build_synth_controls(
    geo_daily: pd.DataFrame,
    groups: Dict[str, str],
    value_col: str = "premium",
    prewindow_days: int = 56,
    ridge_lambda: float = 1.0,
    mm_max_geos: int = 40,
    prefit_min_r2: float = 0.50,
    prefit_max_rmse: float = 1e18,
    prefit_max_mape: float = 1e18,
    min_control_geos: int = 5,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Build *true* synthetic controls: treated geo -> donor weights over CONTROL geos.

    Returns:
      weights_by_treat: {treated_geo: {control_geo: weight}}
      quality_df: per-treated metrics (r2/rmse/mape, n_controls, notes)
    """
    g = geo_daily.copy()
    g["date"] = pd.to_datetime(g["date"])
    end = g["date"].max()
    start = end - pd.Timedelta(days=int(prewindow_days))
    g = g[(g["date"] >= start) & (g["date"] <= end)]

    piv = (
        g.pivot_table(index="date", columns="geo", values=value_col, aggfunc="sum")
        .fillna(0.0)
        .astype("float32")
    )
    if piv.shape[1] < 2:
        return {}, pd.DataFrame()

    # cap geos by total value to save memory
    geo_totals = piv.sum(axis=0).sort_values(ascending=False)
    keep = list(geo_totals.index[: int(mm_max_geos)])
    piv = piv[keep]

    geos = list(piv.columns)
    treated = [x for x in geos if groups.get(x) == "TREATMENT"]
    control = [x for x in geos if groups.get(x) == "CONTROL"]

    if len(control) < int(min_control_geos) or len(treated) < 1:
        q = pd.DataFrame([{
            "treated_geo": "",
            "r2": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "n_controls": int(len(control)),
            "notes": "insufficient_controls_or_treated"
        }])
        return {}, q

    Yc = piv[control].values.astype(float)  # (T, C)
    weights_by_treat: Dict[str, Dict[str, float]] = {}
    qrows = []

    for tg in treated:
        y = piv[tg].values.astype(float)  # (T,)
        w = _ridge_synth_weights_matrix(Yc, y, ridge_lambda=float(ridge_lambda))
        yhat = Yc @ w
        m = _fit_metrics(y, yhat)

        ok = (m["r2"] >= float(prefit_min_r2)) and (m["rmse"] <= float(prefit_max_rmse)) and (m["mape"] <= float(prefit_max_mape))
        if ok:
            weights_by_treat[tg] = {control[i]: float(w[i]) for i in range(len(control)) if float(w[i]) > 0}
            note = "ok"
        else:
            note = "filtered"

        qrows.append({
            "treated_geo": tg,
            "r2": m["r2"],
            "rmse": m["rmse"],
            "mape": m["mape"],
            "n_controls": int(len(control)),
            "notes": note,
        })

    qdf = pd.DataFrame(qrows).sort_values(["notes","r2"], ascending=[True, False]).reset_index(drop=True)
    return weights_by_treat, qdf


def apply_geo_holdout_within_channel(
    geo_weights: Dict[str, float],
    geo_groups: Dict[str, str],
    delta: float,
    min_share: float,
    max_share: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Apply +/-delta on geo weights and renormalize (weights sum to 1)."""
    geos = list(geo_weights.keys())
    w = np.array([float(geo_weights[g]) for g in geos], dtype=float)
    mult = np.ones_like(w)
    for i, g in enumerate(geos):
        grp = geo_groups.get(g, "NA")
        if grp == "TREATMENT":
            mult[i] = 1.0 + float(delta)
        elif grp == "CONTROL":
            mult[i] = max(1e-6, 1.0 - float(delta))
    w = w * mult
    w = _project_simplex_with_bounds(w, float(min_share), float(max_share))
    new = {geos[i]: float(w[i]) for i in range(len(geos))}
    mults = {geos[i]: float(mult[i]) for i in range(len(geos))}
    return new, mults


def build_geo_spend_plan(
    next_date: str,
    reco_channel_budgets: Dict[str, float],
    geo_daily: pd.DataFrame,
    cfg: GeoHoldoutConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a geo-level spend plan for next day (geo x channel).

    Returns:
      plan_df: date, geo, channel, budget, geo_weight, geo_group, geo_multiplier
      assign_df: date, channel, geo, geo_group, geo_multiplier
    """
    if not cfg.enabled:
        return pd.DataFrame(), pd.DataFrame()
    holdout = set(cfg.holdout_channels or [])

    g = geo_daily.copy()
    g["date"] = pd.to_datetime(g["date"])
    end = g["date"].max()
    recent = g[g["date"] >= (end - pd.Timedelta(days=14))].copy()
    if len(recent) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Use eligible geos from recent activity
    eligible_geos = sorted(recent["geo"].dropna().astype(str).unique().tolist())
    geo_groups = assign_geo_groups_weekly_unpaired(
        next_date,
        eligible_geos,
        seed=int(cfg.seed),
        treated_frac=float(cfg.treated_frac),
    )

    rows = []
    assigns = []
    for ch, bud in reco_channel_budgets.items():
        sub = recent[recent["channel"] == ch]
        if len(sub) == 0:
            continue
        geo_sp = sub.groupby("geo")["spend"].sum().astype(float)
        if float(geo_sp.sum()) <= 0:
            geos = sorted(sub["geo"].unique().tolist())
            geo_weights = {gg: 1.0 / len(geos) for gg in geos}
        else:
            geo_weights = (geo_sp / float(geo_sp.sum())).to_dict()

        mults = {gg: 1.0 for gg in geo_weights}
        if ch in holdout:
            geo_weights, mults = apply_geo_holdout_within_channel(
                geo_weights,
                geo_groups,
                delta=float(cfg.delta),
                min_share=float(cfg.min_geo_share),
                max_share=float(cfg.max_geo_share),
            )

        for geo, w in geo_weights.items():
            rows.append(
                {
                    "date": next_date,
                    "geo": geo,
                    "channel": ch,
                    "budget": float(bud) * float(w),
                    "geo_weight": float(w),
                    "geo_group": geo_groups.get(str(geo), "NA"),
                    "geo_multiplier": float(mults.get(geo, 1.0)),
                }
            )
        if ch in holdout:
            for geo in geo_weights.keys():
                assigns.append(
                    {
                        "date": next_date,
                        "channel": ch,
                        "geo": geo,
                        "geo_group": geo_groups.get(str(geo), "NA"),
                        "geo_multiplier": float(mults.get(geo, 1.0)),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(assigns)
