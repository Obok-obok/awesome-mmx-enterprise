from __future__ import annotations

"""Matched market estimator for geo holdout (weekly).

v5 upgrades
-----------
- Use *true* synthetic control: 1 treated geo + weighted basket of control geos
- Filter treated geos using multiple pre-fit metrics: R², RMSE, MAPE
- Support multiple outcome metrics: premium, contracts, ltv
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from .geo_holdout import build_synth_controls, assign_geo_groups_weekly_unpaired
from .ltv_model import RetentionLTVConfig, compute_ltv_total


@dataclass
class MatchedMarketConfig:
    pre_days: int = 28
    post_days: int = 7

    # synth matching
    ridge_lambda: float = 1.0
    mm_max_geos: int = 40
    treated_frac: float = 0.50
    min_control_geos: int = 5

    # quality filters
    prefit_min_r2: float = 0.50
    prefit_max_rmse: float = 1e18
    prefit_max_mape: float = 1e18

    # retention-based ltv
    ltv: RetentionLTVConfig = RetentionLTVConfig()


def _window_split(g: pd.DataFrame, week_end: str, pre_days: int, post_days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    g = g.copy()
    g["date"] = pd.to_datetime(g["date"])
    end = pd.to_datetime(week_end)
    post_start = end - pd.Timedelta(days=int(post_days) - 1)
    pre_end = post_start - pd.Timedelta(days=1)
    pre_start = pre_end - pd.Timedelta(days=int(pre_days) - 1)
    pre = g[(g["date"] >= pre_start) & (g["date"] <= pre_end)]
    post = g[(g["date"] >= post_start) & (g["date"] <= end)]
    return pre, post, pre_start, pre_end, post_start


def _synth_sum(df: pd.DataFrame, weights: Dict[str, float], value_col: str) -> float:
    if not weights:
        return 0.0
    s = 0.0
    for geo, w in weights.items():
        s += float(w) * float(df.loc[df["geo"] == geo, value_col].sum())
    return float(s)


def _treated_sum(df: pd.DataFrame, geo: str, value_col: str) -> float:
    return float(df.loc[df["geo"] == geo, value_col].sum())


def _effect_did(pre: pd.DataFrame, post: pd.DataFrame, treated_geo: str, donor_w: Dict[str, float], value_col: str) -> float:
    """(T_post - S_post) - (T_pre - S_pre)"""
    t_pre = _treated_sum(pre, treated_geo, value_col)
    t_post = _treated_sum(post, treated_geo, value_col)
    s_pre = _synth_sum(pre, donor_w, value_col)
    s_post = _synth_sum(post, donor_w, value_col)
    return float((t_post - s_post) - (t_pre - s_pre))


def estimate_weekly_lift(
    geo_daily: pd.DataFrame,
    week_end: str,
    holdout_channels: Optional[List[str]] = None,
    value_col: str = "premium",
    cfg: MatchedMarketConfig = MatchedMarketConfig(),
    groups: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Weekly lift using synthetic control.

    groups:
      Optional mapping geo -> {CONTROL,TREATMENT}. If not provided, assigns via stable hash.
    """
    g = geo_daily.copy()
    g["date"] = pd.to_datetime(g["date"])

    # Add/overwrite ltv using retention model (if value_col needs it)
    if value_col == "ltv" or ("ltv" in (holdout_channels or [])):  # harmless guard
        g = compute_ltv_total(g, cfg.ltv)

    pre, post, pre_start, pre_end, post_start = _window_split(g, week_end, int(cfg.pre_days), int(cfg.post_days))
    if len(pre) == 0 or len(post) == 0:
        return pd.DataFrame([{
            "week_end": str(pd.to_datetime(week_end).date()),
            "value_col": value_col,
            "lift_total": 0.0,
            "n_treated_used": 0,
            "notes": "insufficient_data",
        }])

    # Determine groups
    if groups is None:
        # Use geos with activity in either window
        geos = sorted(set(pd.concat([pre["geo"], post["geo"]]).dropna().astype(str).unique().tolist()))
        groups = assign_geo_groups_weekly_unpaired(
            str(pd.to_datetime(week_end).date()),
            geos,
            seed=42,
            treated_frac=float(cfg.treated_frac),
        )

    # Build synth controls on PRE window only (stronger identification)
    weights_by_treat, qdf = build_synth_controls(
        pre,
        groups=groups,
        value_col=value_col,
        prewindow_days=int(cfg.pre_days),
        ridge_lambda=float(cfg.ridge_lambda),
        mm_max_geos=int(cfg.mm_max_geos),
        prefit_min_r2=float(cfg.prefit_min_r2),
        prefit_max_rmse=float(cfg.prefit_max_rmse),
        prefit_max_mape=float(cfg.prefit_max_mape),
        min_control_geos=int(cfg.min_control_geos),
    )

    treated_used = [tg for tg, w in weights_by_treat.items() if w]
    if not treated_used:
        return pd.DataFrame([{
            "week_end": str(pd.to_datetime(week_end).date()),
            "value_col": value_col,
            "lift_total": 0.0,
            "n_treated_used": 0,
            "notes": "no_valid_treated_after_filter",
        }])

    # Total lift (sum over treated geos)
    lifts = []
    for tg in treated_used:
        lifts.append(_effect_did(pre, post, tg, weights_by_treat[tg], value_col=value_col))
    row = {
        "week_end": str(pd.to_datetime(week_end).date()),
        "value_col": value_col,
        "lift_total": float(np.sum(lifts)),
        "n_treated_used": int(len(treated_used)),
        "notes": "ok",
    }

    # Add prefit diagnostics (aggregated)
    q_ok = qdf[qdf["notes"] == "ok"].copy() if qdf is not None and len(qdf) else pd.DataFrame()
    if len(q_ok):
        row["prefit_r2_mean"] = float(q_ok["r2"].mean())
        row["prefit_rmse_mean"] = float(q_ok["rmse"].mean())
        row["prefit_mape_mean"] = float(q_ok["mape"].mean())

    # Optional: channel-scoped lift
    hold = set(holdout_channels or [])
    if hold:
        for ch in sorted(hold):
            pre_ch = pre[pre["channel"] == ch]
            post_ch = post[post["channel"] == ch]
            lifts_ch = []
            for tg in treated_used:
                lifts_ch.append(_effect_did(pre_ch, post_ch, tg, weights_by_treat[tg], value_col=value_col))
            row[f"lift_{ch}"] = float(np.sum(lifts_ch))

    return pd.DataFrame([row])


def estimate_weekly_lift_multi(
    geo_daily: pd.DataFrame,
    week_end: str,
    holdout_channels: Optional[List[str]] = None,
    value_cols: Optional[List[str]] = None,
    cfg: MatchedMarketConfig = MatchedMarketConfig(),
    groups: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    g = geo_daily.copy()
    g["date"] = pd.to_datetime(g["date"])

    cols = value_cols or ["premium", "contracts", "ltv"]
    rows = []
    for c in cols:
        if c not in g.columns and c != "ltv":
            continue
        # ltv is computed inside estimate_weekly_lift
        res = estimate_weekly_lift(
            g,
            week_end=week_end,
            holdout_channels=holdout_channels,
            value_col=c,
            cfg=cfg,
            groups=groups,
        )
        rows.append(res)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
