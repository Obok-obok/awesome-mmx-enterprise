"""Pre-release validation for MMX engine artifacts.

Runs lightweight consistency checks that should pass in free-tier VM environments.
- Verifies required columns
- Verifies non-negativity
- Verifies ROI 5-factor recomposition (daily + monthly)
- Verifies key artifacts existence

Usage:
  python scripts/validate_engine.py --out out
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


REQUIRED_BASE_COLS = [
    "date",
    "channel",
    "spend",
    "leads",
    "tm_attempts",
    "tm_connected",
    "contracts",
    "premium",
]

REQUIRED_DERIVED_COLS = [
    "leads_per_spend",
    "attempts_per_lead",
    "connected_per_attempt",
    "contracts_per_connected",
    "premium_per_contract",
    "roi",
    "roi_implied",
]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _check_panel(path: str, name: str) -> None:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_BASE_COLS + REQUIRED_DERIVED_COLS if c not in df.columns]
    _assert(not missing, f"{name}: missing columns: {missing}")

    # non-negativity
    for c in ["spend", "leads", "tm_attempts", "tm_connected", "contracts", "premium"]:
        bad = (pd.to_numeric(df[c], errors="coerce") < 0).sum()
        _assert(bad == 0, f"{name}: negative values found in {c} (count={bad})")

    # funnel monotonicity (soft constraints; allow equals)
    bad1 = (df["tm_connected"] > df["tm_attempts"]).sum()
    bad2 = (df["contracts"] > df["tm_connected"]).sum()
    _assert(bad1 == 0, f"{name}: tm_connected > tm_attempts rows={bad1}")
    _assert(bad2 == 0, f"{name}: contracts > tm_connected rows={bad2}")

    # ROI recomposition check where all denominators are positive
    mask = (
        (df["spend"] > 0)
        & (df["leads"] > 0)
        & (df["tm_attempts"] > 0)
        & (df["tm_connected"] > 0)
        & (df["contracts"] > 0)
    )
    if mask.any():
        diff = (df.loc[mask, "roi"] - df.loc[mask, "roi_implied"]).abs()
        mx = float(diff.max())
        _assert(mx < 1e-9, f"{name}: ROI mismatch max_abs_diff={mx}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="output directory (default: out)")
    args = ap.parse_args()

    out_dir = args.out
    needed = [
        "panel_daily_channel.csv",
        "panel_monthly_channel.csv",
        "panel_daily_campaign.csv",
        "panel_monthly_campaign.csv",
        "posterior_summary_channel.csv",
        "counterfactuals_channel.csv",
        "budget_recommendation.csv",
        "data_quality_report.json",
        "metric_lineage.csv",
    ]
    missing = [f for f in needed if not os.path.exists(os.path.join(out_dir, f))]
    _assert(not missing, f"Missing artifacts in {out_dir}: {missing}")

    _check_panel(os.path.join(out_dir, "panel_daily_channel.csv"), "panel_daily_channel")
    _check_panel(os.path.join(out_dir, "panel_monthly_channel.csv"), "panel_monthly_channel")

    # campaign panels may not always have channel/campaign dims but should have the core columns
    _check_panel(os.path.join(out_dir, "panel_daily_campaign.csv"), "panel_daily_campaign")
    _check_panel(os.path.join(out_dir, "panel_monthly_campaign.csv"), "panel_monthly_campaign")

    print("OK: engine artifacts validated")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise
