from __future__ import annotations
import pandas as pd
import numpy as np

def load_channel_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    expected = {"date", "channel", "spend", "leads", "tm_attempts", "tm_connected", "contracts", "premium"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in channel file: {missing}")
    return df


def load_campaign_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    expected = {
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
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in campaign file: {missing}")
    return df


def _safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe division returning float with NaN when denominator is 0 or missing."""
    den2 = den.replace(0, np.nan)
    out = num / den2
    # Ensure numpy NaN (not pd.NA) so .astype(float) never crashes.
    return pd.to_numeric(out, errors="coerce").astype(float)


def build_panel_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["cpl"] = _safe_rate(out["spend"], out["leads"])
    out["cpa_contract"] = _safe_rate(out["spend"], out["contracts"])
    out["connect_rate"] = _safe_rate(out["tm_connected"], out["tm_attempts"])
    out["close_rate"] = _safe_rate(out["contracts"], out["tm_connected"])

    # --- ROI factorization (Premium/Spend) ---
    # Premium/Spend = (Leads/Spend) * (Attempts/Leads) * (Connected/Attempts) * (Contracts/Connected) * (Premium/Contract)
    out["leads_per_spend"] = _safe_rate(out["leads"], out["spend"])
    out["attempts_per_lead"] = _safe_rate(out["tm_attempts"], out["leads"])
    out["connected_per_attempt"] = _safe_rate(out["tm_connected"], out["tm_attempts"])
    out["contracts_per_connected"] = _safe_rate(out["contracts"], out["tm_connected"])
    out["premium_per_contract"] = _safe_rate(out["premium"], out["contracts"])
    out["roi"] = _safe_rate(out["premium"], out["spend"])
    out["roi_implied"] = (
        out["leads_per_spend"]
        * out["attempts_per_lead"]
        * out["connected_per_attempt"]
        * out["contracts_per_connected"]
        * out["premium_per_contract"]
    )

    return out
