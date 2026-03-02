from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    level: str  # PASS/WARN/FAIL
    check: str
    detail: str


def _result(level: str, check: str, detail: str) -> dict:
    return {"level": level, "check": check, "detail": detail}


def _is_missing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c not in df.columns]


def _pct_missing(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    return float(pd.isna(s).mean())


def _outlier_rate(x: pd.Series, p: float = 0.99, mult: float = 5.0) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 30:
        return 0.0
    thr = float(np.nanquantile(x, p) * mult)
    if thr <= 0:
        return 0.0
    return float((x > thr).mean())


def run_data_quality(df_ch: pd.DataFrame, df_camp: pd.DataFrame) -> str:
    """Stronger data-quality gate.

    Covers:
      - Missing required columns
      - NaNs / invalid dates
      - Negative values
      - Basic funnel consistency constraints
      - Extreme outliers (warn)
    Returns a JSON string for easy writing.
    """
    results: list[dict] = []

    # Required columns
    req_ch = ["date", "channel", "spend", "leads", "tm_attempts", "tm_connected", "contracts", "premium"]
    req_camp = [
        "date",
        "channel",
        "campaign_id",
        "message_type",
        "spend",
        "leads",
        "tm_attempts",
        "tm_connected",
        "contracts",
        "premium",
    ]

    miss = _is_missing(df_ch, req_ch)
    results.append(
        _result("FAIL" if miss else "PASS", "required_columns.channel", f"missing={miss}" if miss else "ok")
    )
    miss = _is_missing(df_camp, req_camp)
    results.append(
        _result("FAIL" if miss else "PASS", "required_columns.campaign", f"missing={miss}" if miss else "ok")
    )

    # Date parsing
    for name, df in [("channel", df_ch), ("campaign", df_camp)]:
        d = pd.to_datetime(df.get("date"), errors="coerce")
        bad = int(d.isna().sum())
        results.append(
            _result("FAIL" if bad else "PASS", f"date_parse.{name}", f"bad_rows={bad}")
        )

    # Missingness thresholds (warn)
    for name, df, cols in [
        ("channel", df_ch, ["spend", "leads", "tm_attempts", "tm_connected", "contracts", "premium"]),
        ("campaign", df_camp, ["spend", "leads", "tm_attempts", "tm_connected", "contracts", "premium"]),
    ]:
        for c in cols:
            pct = _pct_missing(df[c])
            lvl = "WARN" if pct > 0.01 else "PASS"
            results.append(_result(lvl, f"missingness.{name}.{c}", f"missing_pct={pct:.3f}"))

    # Negatives (fail)
    for name, df in [("channel", df_ch), ("campaign", df_camp)]:
        for c in ["spend", "leads", "tm_attempts", "tm_connected", "contracts", "premium"]:
            x = pd.to_numeric(df[c], errors="coerce")
            neg = int((x < 0).sum())
            results.append(_result("FAIL" if neg else "PASS", f"non_negative.{name}.{c}", f"neg_rows={neg}"))

    # Funnel consistency (fail or warn)
    def _consistency(df: pd.DataFrame, key: str) -> None:
        a = pd.to_numeric(df["tm_attempts"], errors="coerce").fillna(0)
        c = pd.to_numeric(df["tm_connected"], errors="coerce").fillna(0)
        k = pd.to_numeric(df["contracts"], errors="coerce").fillna(0)
        p = pd.to_numeric(df["premium"], errors="coerce").fillna(0)
        bad1 = int((c > a).sum())
        bad2 = int((k > c).sum())
        bad3 = int(((k == 0) & (p > 0)).sum())
        bad4 = int(((k > 0) & (p <= 0)).sum())
        results.append(_result("FAIL" if bad1 else "PASS", f"funnel.tm_connected_le_attempts.{key}", f"bad_rows={bad1}"))
        results.append(_result("FAIL" if bad2 else "PASS", f"funnel.contracts_le_connected.{key}", f"bad_rows={bad2}"))
        # premium consistency is warn (real data may have accounting delays)
        lvl3 = "WARN" if bad3 else "PASS"
        lvl4 = "WARN" if bad4 else "PASS"
        results.append(_result(lvl3, f"premium.zero_contracts.{key}", f"bad_rows={bad3}"))
        results.append(_result(lvl4, f"premium.missing_when_contracts.{key}", f"bad_rows={bad4}"))

    _consistency(df_ch, "channel")
    _consistency(df_camp, "campaign")

    # Outliers (warn)
    for name, df in [("channel", df_ch), ("campaign", df_camp)]:
        for c in ["spend", "leads", "contracts", "premium"]:
            rate = _outlier_rate(df[c])
            lvl = "WARN" if rate > 0.01 else "PASS"
            results.append(_result(lvl, f"outliers.{name}.{c}", f"outlier_rate={rate:.3f}"))

    # Summaries
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        counts[r["level"]] += 1

    payload = {
        "summary": counts,
        "results": results,
        "policy": {
            "missing_warn_threshold": 0.01,
            "outlier_rule": "x > p99*5",
            "funnel_constraints": [
                "tm_connected <= tm_attempts",
                "contracts <= tm_connected",
                "premium should be 0 when contracts=0 (warn)",
            ],
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
