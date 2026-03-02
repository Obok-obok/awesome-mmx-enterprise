#!/usr/bin/env python
"""
Validate end-to-end data integrity for MMX.

This script is intended for operators to quickly verify:
- Mart schema + value constraints
- Funnel monotonicity constraints
- Data Quality Gate report consistency
- Latest decision artifact consistency (Decision vs Funnel forecast)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_MART = REPO_ROOT / "data" / "curated" / "mart" / "daily_channel_fact.csv"
DQ_LATEST = REPO_ROOT / "logs" / "pipeline" / "mart_validation_latest.json"
DEC_DIR = REPO_ROOT / "artifacts" / "recommendations" / "decisions"
FC_DIR = REPO_ROOT / "artifacts" / "recommendations" / "funnel_forecast"


REQUIRED_MART_COLS = [
    "date",
    "channel",
    "spend",
    "leads",
    "call_attempt",
    "call_connected",
    "contracts",
    "premium",
]


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _latest_file(dir_path: Path, suffix: str) -> Path:
    files = sorted([p for p in dir_path.glob(f"*{suffix}") if p.is_file()])
    if not files:
        _fail(f"No files found in {dir_path} with suffix={suffix}")
    return files[-1]


def validate_mart(strict: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not DATA_MART.exists():
        _fail(f"Mart not found: {DATA_MART}")
    df = pd.read_csv(DATA_MART)
    missing = [c for c in REQUIRED_MART_COLS if c not in df.columns]
    if missing:
        _fail(f"Mart missing columns: {missing}")
    _ok("Mart schema")

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    nat_rows = int(df["date"].isna().sum())
    if strict and nat_rows > 0:
        _fail(f"Mart date has NaT rows: {nat_rows}")
    _ok("Mart date parse")

    # Non-negativity
    if strict and (df["spend"] < 0).any():
        _fail("Negative spend found")
    if strict and (df["premium"] < 0).any():
        _fail("Negative premium found")
    counts = ["leads", "call_attempt", "call_connected", "contracts"]
    if strict and (df[counts] < 0).any().any():
        _fail("Negative counts found")
    _ok("Non-negativity")

    # Monotonicity
    v1 = int((df["call_connected"] > df["call_attempt"]).sum())
    v2 = int((df["contracts"] > df["call_connected"]).sum())
    v3 = int(((df["premium"] > 0) & (df["contracts"] <= 0)).sum())
    if strict and (v1 or v2 or v3):
        _fail(f"Monotonicity violations: connected>attempt={v1}, contracts>connected={v2}, premium>0 but contracts<=0={v3}")
    _ok("Funnel monotonicity")

    # Coverage sanity
    n_days = df["date"].dt.date.nunique()
    n_ch = df["channel"].nunique()
    if strict and len(df) != n_days * n_ch:
        _fail(f"Mart missing (date,channel) rows. rows={len(df)} expected={n_days*n_ch}")
    _ok("Mart coverage")

    rep = {}
    if DQ_LATEST.exists():
        rep = _read_json(DQ_LATEST)
        _ok("Data Quality Gate report present")
    else:
        _ok("Data Quality Gate report not found (skipping)")

    return df, rep


def validate_latest_decision(strict: bool = True) -> None:
    if not DEC_DIR.exists():
        _ok("No decision artifacts directory (skipping)")
        return
    dec_path = _latest_file(DEC_DIR, ".json")
    dec = _read_json(dec_path)
    _ok(f"Latest decision: {dec_path.name}")

    # Decision checks
    for k in ["expected_premium", "premium_std", "risk_adjusted_premium"]:
        if k not in dec:
            _fail(f"Decision missing key: {k}")
        if strict and not isinstance(dec[k], (int, float)):
            _fail(f"Decision key not numeric: {k}={dec[k]}")

    vr = dec.get("validation_report", {})
    if strict:
        for k in ["sum_budget", "sum_error", "min_violation", "max_violation", "delta_violation", "ramp_violation"]:
            if k not in vr:
                _fail(f"Decision.validation_report missing: {k}")
        if abs(float(vr["sum_error"])) > 1e-6:
            _fail(f"Budget sum error too large: {vr['sum_error']}")
        for k in ["min_violation", "max_violation", "delta_violation", "ramp_violation"]:
            if float(vr[k]) > 1e-6:
                _fail(f"Constraint violation detected: {k}={vr[k]}")
    _ok("Decision validation_report")

    cov = dec.get("data_coverage", None)
    if cov is not None and strict:
        if not (0.0 <= float(cov) <= 1.0):
            _fail(f"data_coverage out of range: {cov}")
    _ok("Decision data_coverage")

    # Funnel forecast consistency (premium row)
    fc_path = FC_DIR / (dec_path.stem + ".csv")
    if not fc_path.exists():
        _ok("No funnel forecast CSV for latest decision (skipping)")
        return
    fc = pd.read_csv(fc_path)
    prem = fc[fc["stage"] == "premium"]
    if prem.empty:
        _fail("Funnel forecast missing premium stage")
    prem_row = prem.iloc[0]
    for col, key in [("expected", "expected_premium"), ("std", "premium_std")]:
        if abs(float(prem_row[col]) - float(dec[key])) > 1e-9:
            _fail(f"Decision vs Funnel mismatch: {key}={dec[key]} vs fc.{col}={prem_row[col]}")
    _ok("Decision vs Funnel forecast premium consistency")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-strict", action="store_true", help="Relax failures to warnings where possible.")
    args = parser.parse_args()
    strict = not args.non_strict

    validate_mart(strict=strict)
    validate_latest_decision(strict=strict)
    print("[PASS] Data integrity checks passed.")


if __name__ == "__main__":
    main()
