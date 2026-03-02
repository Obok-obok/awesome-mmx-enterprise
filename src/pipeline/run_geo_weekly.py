from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.ops.matched_market import MatchedMarketConfig, estimate_weekly_lift_multi
from src.ops.ltv_model import RetentionLTVConfig
from src.utils.logging_utils import silence_noise


logger = logging.getLogger(__name__)


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/mmx.yaml") -> None:
    silence_noise()
    cfg = _load_cfg(config_path)

    geo_cfg = cfg.get("geo_holdout", {})
    if not bool(geo_cfg.get("enabled", False)):
        logger.info("Geo holdout disabled. Skip.")
        return

    geo_path = Path(geo_cfg.get("input_geo_channel_path", "data/input_daily_geo_channel.csv"))
    if not geo_path.exists():
        logger.info("Missing geo input: %s. Skip.", geo_path)
        return

    out_root = Path(cfg["data"]["out_dir"]) / "ops"
    out_root.mkdir(parents=True, exist_ok=True)

    geo = pd.read_csv(geo_path, parse_dates=["date"])
    if len(geo) == 0:
        logger.info("Geo input empty. Skip.")
        return

    # Memory guard: keep only the recent window needed for pre/post + matching
    dmax = geo["date"].max()
    keep_days = int(geo_cfg.get("mm_pre_days", 28)) + int(geo_cfg.get("mm_post_days", 7)) + int(geo_cfg.get("prewindow_days", 56))
    geo = geo[geo["date"] >= (dmax - pd.Timedelta(days=keep_days + 14))].copy()

    week_end = str(geo["date"].max().date())

    mm_cfg = MatchedMarketConfig(
        pre_days=int(geo_cfg.get("mm_pre_days", 28)),
        post_days=int(geo_cfg.get("mm_post_days", 7)),
        ridge_lambda=float(geo_cfg.get("mm_ridge_lambda", 1.0)),
        mm_max_geos=int(geo_cfg.get("mm_max_geos", 40)),
        treated_frac=float(geo_cfg.get("treated_frac", 0.50)),
        min_control_geos=int(geo_cfg.get("min_control_geos", 5)),
        prefit_min_r2=float(geo_cfg.get("mm_prefit_min_r2", 0.50)),
        prefit_max_rmse=float(geo_cfg.get("mm_prefit_max_rmse", 1e18)),
        prefit_max_mape=float(geo_cfg.get("mm_prefit_max_mape", 1e18)),
        ltv=RetentionLTVConfig(
            horizon_months=int(geo_cfg.get("ltv_horizon_months", 36)),
            monthly_retention_default=float(geo_cfg.get("ltv_monthly_retention_default", 0.92)),
            monthly_discount_default=float(geo_cfg.get("ltv_monthly_discount_default", 0.01)),
            retention_path=str(geo_cfg.get("ltv_retention_path", "data/input_channel_retention.csv")),
        ),
    )
    holdout_channels = list(geo_cfg.get("holdout_channels", []))

    # Optional: use explicit geo assignments (CONTROL/TREATMENT) if available
    groups = None
    assign_path = out_root / "geo_assignments.csv"
    if assign_path.exists():
        try:
            assigns = pd.read_csv(assign_path, parse_dates=["date"])
            end_dt = pd.to_datetime(week_end)
            post_start = end_dt - pd.Timedelta(days=int(mm_cfg.post_days) - 1)
            a = assigns[(assigns["date"] >= post_start) & (assigns["date"] <= end_dt)].copy()
            if len(a) > 0 and "geo_group" in a.columns:
                d_last = a["date"].max()
                a = a[a["date"] == d_last]
                groups = {
                    str(r["geo"]): str(r["geo_group"]).upper()
                    for _, r in a.iterrows()
                    if str(r.get("geo_group", "")).upper() in ("CONTROL", "TREATMENT")
                }
        except Exception:
            groups = None

    res = estimate_weekly_lift_multi(
        geo,
        week_end=week_end,
        holdout_channels=holdout_channels,
        value_cols=list(geo_cfg.get("mm_value_cols", ["premium", "contracts", "ltv"])),
        cfg=mm_cfg,
        groups=groups,
    )

    out_path = out_root / "geo_mm_results_weekly.csv"
    if out_path.exists():
        old = pd.read_csv(out_path)
        allx = pd.concat([old, res], ignore_index=True)
        allx.drop_duplicates(subset=["week_end", "value_col"], keep="last", inplace=True)
    else:
        allx = res
    allx.to_csv(out_path, index=False)

    logger.info("Geo matched-market weekly result saved: %s", out_path)


if __name__ == "__main__":
    main()
