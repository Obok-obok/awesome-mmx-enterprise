from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/mmx.yaml") -> None:
    cfg = _load_cfg(config_path)
    out_root = Path(cfg["data"]["out_dir"]) / "ops"

    assign_path = out_root / "ab_assignments.csv"
    perf_path = out_root / "performance_daily.csv"

    if not assign_path.exists() or not perf_path.exists():
        logger.info("A/B weekly skipped (missing logs).")
        return

    assign = pd.read_csv(assign_path)
    assign["date"] = pd.to_datetime(assign["date"])
    perf = pd.read_csv(perf_path)
    perf["date"] = pd.to_datetime(perf["date"])

    # Use last full 7-day window ending on latest perf date
    end = perf["date"].max()
    start = end - pd.Timedelta(days=6)

    a = assign[(assign["date"] >= start) & (assign["date"] <= end)].copy()
    p = perf[(perf["date"] >= start) & (perf["date"] <= end)].copy()

    # Join on date; premium is total, so we do a simple attribution proxy:
    # compare total premium on treatment vs control days for holdout channels.
    # (In production you'd do geo split / partial pooling; this is a light-weight starter.)
    merged = a.merge(p[["date", "actual_premium"]], on="date", how="inner")
    if merged.empty:
        logger.info("A/B weekly skipped (no overlap).")
        return

    # Aggregate by group across holdout channels
    # A date may have multiple channels; we treat it as treatment day if any holdout channel is treatment.
    day_group = (
        merged.groupby(["date"])  # per day
        .agg(
            any_treat=("group", lambda s: (s == "TREATMENT").any()),
            any_control=("group", lambda s: (s == "CONTROL").any()),
            premium=("actual_premium", "first"),
        )
        .reset_index()
    )

    treat = day_group[day_group["any_treat"] & ~day_group["any_control"]]
    ctrl = day_group[day_group["any_control"] & ~day_group["any_treat"]]

    # If mixed days exist, we exclude them to keep it simple.
    lift = float(treat["premium"].mean() - ctrl["premium"].mean()) if (len(treat) > 0 and len(ctrl) > 0) else 0.0

    holdout_channels = sorted(set(a["channel"].tolist()))
    notes = "Simple diff of means on pure treatment vs pure control days; excludes mixed days."

    out = pd.DataFrame(
        [
            {
                "week_start": str(start.date()),
                "week_end": str(end.date()),
                "channels": ",".join(holdout_channels),
                "n_treat_days": int(len(treat)),
                "n_control_days": int(len(ctrl)),
                "lift_premium": float(lift),
                "notes": notes,
            }
        ]
    )

    out_path = out_root / "ab_results_weekly.csv"
    if out_path.exists():
        old = pd.read_csv(out_path)
        all_ = pd.concat([old, out], ignore_index=True)
        all_.drop_duplicates(subset=["week_end"], keep="last", inplace=True)
    else:
        all_ = out
    all_.to_csv(out_path, index=False)

    logger.info("A/B weekly results updated: %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
