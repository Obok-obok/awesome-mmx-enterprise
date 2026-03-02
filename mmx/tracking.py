from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


# -----------------------------------------------------------------------------
# Plan vs Actual Tracking
# -----------------------------------------------------------------------------

# Funnel tracking metrics (enterprise ops)
# Premium = Spend × (1/CPL) × AttemptRate × ConnectRate × CloseRate × APS
#   CPL         = Spend / Leads
#   AttemptRate = Attempts / Leads
#   ConnectRate = Connected / Attempts
#   CloseRate   = Contracts / Connected
#   APS         = Premium / Contracts
REQUIRED_METRICS = ["spend", "leads", "attempts", "connected", "contracts", "premium"]


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root_dir: str

    @property
    def run_dir(self) -> str:
        return os.path.join(self.root_dir, self.run_id)

    @property
    def plan_json(self) -> str:
        return os.path.join(self.run_dir, "plan_summary.json")

    @property
    def plan_by_channel_csv(self) -> str:
        return os.path.join(self.run_dir, "plan_by_channel.csv")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def validate_actuals_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate/standardize actuals schema.

    Expected columns:
      - month (YYYY-MM) OR date (YYYY-MM-DD)
      - channel
      - spend, leads, contracts, premium

    Adds standardized month column (YYYY-MM).
    """
    if df is None or df.empty:
        raise ValueError("Actuals data is empty.")

    cols = set(df.columns)
    if "channel" not in cols:
        raise ValueError("Actuals must include 'channel' column.")

    if "month" in cols:
        m = df["month"].astype(str)
        df = df.copy()
        df["month"] = m.str.slice(0, 7)
    elif "date" in cols:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if dt.isna().all():
            raise ValueError("Actuals 'date' column could not be parsed.")
        df = df.copy()
        df["month"] = dt.dt.to_period("M").astype(str)
    else:
        raise ValueError("Actuals must include either 'month' or 'date'.")

    df = df.copy()
    df["channel"] = df["channel"].astype(str)

    # Backward-compatible aliases
    if "tm_attempts" in df.columns and "attempts" not in df.columns:
        df["attempts"] = df["tm_attempts"]
    if "tm_connected" in df.columns and "connected" not in df.columns:
        df["connected"] = df["tm_connected"]

    for c in REQUIRED_METRICS:
        if c not in df.columns:
            raise ValueError(f"Actuals must include '{c}' column.")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        if (df[c] < 0).any():
            raise ValueError(f"Actuals column '{c}' contains negative values.")
    return df


def plan_from_opt_simulation(opt_simulation: dict) -> pd.DataFrame:
    if not isinstance(opt_simulation, dict) or not opt_simulation:
        raise ValueError("opt_simulation is missing. Run the optimizer first.")
    by_ch = opt_simulation.get("by_channel", [])
    if not isinstance(by_ch, list) or len(by_ch) == 0:
        raise ValueError("opt_simulation.by_channel is empty.")

    df = pd.DataFrame(by_ch)
    df = df.rename(
        columns={
            "채널": "channel",
            "권장(최적화)": "recommended_spend",
            "예측 Leads": "pred_leads",
            "예측 Contracts": "pred_contracts",
            "예측 Premium": "pred_premium",
        }
    )
    need = {"channel", "recommended_spend", "pred_leads", "pred_contracts", "pred_premium"}
    if not need.issubset(set(df.columns)):
        missing = sorted(list(need - set(df.columns)))
        raise ValueError(f"opt_simulation.by_channel missing fields: {missing}")

    df["channel"] = df["channel"].astype(str)
    for c in ["recommended_spend", "pred_leads", "pred_contracts", "pred_premium"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df[["channel", "recommended_spend", "pred_leads", "pred_contracts", "pred_premium"]]


def save_plan_run(
    *,
    opt_simulation: dict,
    month: str,
    history_start: str | None = None,
    history_end: str | None = None,
    forecast_start: str | None = None,
    forecast_end: str | None = None,
    root_dir: str = "out/runs",
    run_id: str | None = None,
) -> RunPaths:
    run_id = run_id or new_run_id()
    paths = RunPaths(run_id=run_id, root_dir=root_dir)
    _ensure_dir(paths.run_dir)

    plan_df = plan_from_opt_simulation(opt_simulation).copy()
    plan_df.insert(0, "month", str(month)[:7])
    plan_df.to_csv(paths.plan_by_channel_csv, index=False, encoding="utf-8")

    totals = opt_simulation.get("totals", {}) if isinstance(opt_simulation.get("totals"), dict) else {}
    summary = {
        "run_id": run_id,
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "month": str(month)[:7],
        "periods": {
            "history_start": history_start,
            "history_end": history_end,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
        },
        "objective": str(opt_simulation.get("objective", "")),
        "ycol": str(opt_simulation.get("ycol", "")),
        "total_budget": float(opt_simulation.get("total_budget", totals.get("budget", 0.0)) or 0.0),
        "totals": {
            "budget": float(totals.get("budget", 0.0) or 0.0),
            "leads": float(totals.get("leads", 0.0) or 0.0),
            "contracts": float(totals.get("contracts", 0.0) or 0.0),
            "premium": float(totals.get("premium", 0.0) or 0.0),
        },
    }
    with open(paths.plan_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return paths


def list_runs(root_dir: str = "out/runs") -> list[str]:
    if not os.path.isdir(root_dir):
        return []
    runs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    runs.sort(reverse=True)
    return runs


def load_plan_by_channel(run_id: str, root_dir: str = "out/runs") -> pd.DataFrame:
    p = RunPaths(run_id=run_id, root_dir=root_dir)
    if not os.path.exists(p.plan_by_channel_csv):
        raise FileNotFoundError(f"Plan file not found: {p.plan_by_channel_csv}")
    df = pd.read_csv(p.plan_by_channel_csv)
    need = {"month", "channel", "recommended_spend", "pred_leads", "pred_contracts", "pred_premium"}
    if not need.issubset(set(df.columns)):
        raise ValueError("Plan csv schema mismatch.")
    df["month"] = df["month"].astype(str).str.slice(0, 7)
    df["channel"] = df["channel"].astype(str)
    for c in ["recommended_spend", "pred_leads", "pred_attempts", "pred_connected", "pred_contracts", "pred_premium"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def compare_plan_vs_actual(
    plan_by_channel: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    month: str,
) -> tuple[pd.DataFrame, dict]:
    if plan_by_channel is None or plan_by_channel.empty:
        raise ValueError("Plan data is empty.")

    actuals = validate_actuals_df(actuals)
    month = str(month)[:7]

    plan_m = plan_by_channel.copy()
    if "month" in plan_m.columns:
        plan_m = plan_m[plan_m["month"].astype(str).str.slice(0, 7) == month].copy()
    plan_m["month"] = month

    act_m = actuals[actuals["month"].astype(str).str.slice(0, 7) == month].copy()
    if act_m.empty:
        raise ValueError(f"No actuals rows for month={month}.")

    act_by = act_m.groupby("channel", as_index=False).agg(
        actual_spend=("spend", "sum"),
        actual_leads=("leads", "sum"),
        actual_attempts=("attempts", "sum"),
        actual_connected=("connected", "sum"),
        actual_contracts=("contracts", "sum"),
        actual_premium=("premium", "sum"),
    )

    plan_by = plan_m.groupby("channel", as_index=False).agg(
        recommended_spend=("recommended_spend", "sum"),
        pred_leads=("pred_leads", "sum"),
        pred_contracts=("pred_contracts", "sum"),
        pred_premium=("pred_premium", "sum"),
    )

    # If the plan does not carry attempts/connected, default to a neutral decomposition:
    # pred_attempts = pred_leads, pred_connected = pred_leads (=> AttemptRate=1, ConnectRate=1, CloseRate=Contracts/Leads)
    if "pred_attempts" not in plan_by.columns:
        plan_by["pred_attempts"] = plan_by["pred_leads"]
    if "pred_connected" not in plan_by.columns:
        plan_by["pred_connected"] = plan_by["pred_leads"]

    merged = pd.merge(plan_by, act_by, on="channel", how="outer").fillna(0.0)
    merged.insert(0, "month", month)

    merged["var_spend"] = merged["actual_spend"] - merged["recommended_spend"]
    merged["var_leads"] = merged["actual_leads"] - merged["pred_leads"]
    merged["var_attempts"] = merged["actual_attempts"] - merged.get("pred_attempts", 0.0)
    merged["var_connected"] = merged["actual_connected"] - merged.get("pred_connected", 0.0)
    merged["var_contracts"] = merged["actual_contracts"] - merged["pred_contracts"]
    merged["var_premium"] = merged["actual_premium"] - merged["pred_premium"]

    merged["roi_plan"] = merged.apply(
        lambda r: (r["pred_premium"] / r["recommended_spend"]) if r["recommended_spend"] > 0 else 0.0,
        axis=1,
    )
    merged["roi_actual"] = merged.apply(
        lambda r: (r["actual_premium"] / r["actual_spend"]) if r["actual_spend"] > 0 else 0.0,
        axis=1,
    )

    totals = {
        "month": month,
        "recommended_spend": float(merged["recommended_spend"].sum()),
        "actual_spend": float(merged["actual_spend"].sum()),
        "pred_leads": float(merged["pred_leads"].sum()),
        "actual_leads": float(merged["actual_leads"].sum()),
        "pred_contracts": float(merged["pred_contracts"].sum()),
        "actual_contracts": float(merged["actual_contracts"].sum()),
        "pred_premium": float(merged["pred_premium"].sum()),
        "actual_premium": float(merged["actual_premium"].sum()),
    }
    return merged, totals

# -----------------------------------------------------------------------------
# Period-based compare + driver decomposition (CPL/RR/APS/ROI)
# -----------------------------------------------------------------------------

from itertools import permutations
from datetime import date

def months_inclusive(start: date, end: date) -> int:
    """Count calendar months between start and end, inclusive.

    Example: 2026-03-01 ~ 2026-03-31 => 1
             2026-03-01 ~ 2026-04-01 => 2 (Mar, Apr)
    """
    if start is None or end is None:
        return 0
    if start > end:
        return 0
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def _safe_div(a: float, b: float) -> float:
    try:
        a = float(a)
        b = float(b)
        if b == 0:
            return 0.0
        return a / b
    except Exception:
        return 0.0


def _metrics_from_row(spend: float, leads: float, attempts: float, connected: float, contracts: float, premium: float) -> dict:
    spend = float(spend or 0.0)
    leads = float(leads or 0.0)
    attempts = float(attempts or 0.0)
    connected = float(connected or 0.0)
    contracts = float(contracts or 0.0)
    premium = float(premium or 0.0)
    cpl = _safe_div(spend, leads) if leads > 0 else 0.0
    attempt_rate = _safe_div(attempts, leads) if leads > 0 else 0.0
    connect_rate = _safe_div(connected, attempts) if attempts > 0 else 0.0
    close_rate = _safe_div(contracts, connected) if connected > 0 else 0.0
    aps = _safe_div(premium, contracts) if contracts > 0 else 0.0
    roi = _safe_div(premium, spend) if spend > 0 else 0.0
    return {
        "spend": spend,
        "leads": leads,
        "attempts": attempts,
        "connected": connected,
        "contracts": contracts,
        "premium": premium,
        "cpl": cpl,
        "attempt_rate": attempt_rate,
        "connect_rate": connect_rate,
        "close_rate": close_rate,
        "aps": aps,
        "roi": roi,
    }


def scale_plan_by_months(plan_df: pd.DataFrame, n_months: int) -> pd.DataFrame:
    """Scale monthly plan to match a multi-month forecast window."""
    n = int(n_months) if n_months and n_months > 0 else 1
    out = plan_df.copy()
    for c in ["recommended_spend", "pred_leads", "pred_attempts", "pred_connected", "pred_contracts", "pred_premium"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0) * float(n)
    return out


def aggregate_actuals_period(actuals: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    actuals = validate_actuals_df(actuals)
    # use month (YYYY-MM) column, compare by month boundaries
    m_start = f"{start.year:04d}-{start.month:02d}"
    m_end = f"{end.year:04d}-{end.month:02d}"
    df = actuals.copy()
    df["month"] = df["month"].astype(str).str.slice(0, 7)
    df = df[(df["month"] >= m_start) & (df["month"] <= m_end)].copy()
    if df.empty:
        raise ValueError(f"No actuals rows for period={m_start}~{m_end}.")
    by = df.groupby("channel", as_index=False).agg(
        actual_spend=("spend", "sum"),
        actual_leads=("leads", "sum"),
        actual_attempts=("attempts", "sum"),
        actual_connected=("connected", "sum"),
        actual_contracts=("contracts", "sum"),
        actual_premium=("premium", "sum"),
    )
    return by


def compare_plan_vs_actual_period(
    plan_by_channel: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    start: date,
    end: date,
    scale_by_forecast_months: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Compare plan vs actual aggregated over a period.

    If scale_by_forecast_months=True, the plan is assumed to be a 1-month plan and is
    multiplied by the number of months in [start, end] inclusive.
    """
    if plan_by_channel is None or plan_by_channel.empty:
        raise ValueError("Plan data is empty.")
    n_months = months_inclusive(start, end) or 1
    plan = plan_by_channel.copy()
    if scale_by_forecast_months:
        plan = scale_plan_by_months(plan, n_months)

    plan_by = plan.groupby("channel", as_index=False).agg(
        recommended_spend=("recommended_spend", "sum"),
        pred_leads=("pred_leads", "sum"),
        pred_attempts=("pred_attempts", "sum") if "pred_attempts" in plan.columns else ("pred_leads", "sum"),
        pred_connected=("pred_connected", "sum") if "pred_connected" in plan.columns else ("pred_leads", "sum"),
        pred_contracts=("pred_contracts", "sum"),
        pred_premium=("pred_premium", "sum"),
    )

    act_by = aggregate_actuals_period(actuals, start, end)

    merged = pd.merge(plan_by, act_by, on="channel", how="outer").fillna(0.0)
    merged.insert(0, "period", f"{start}~{end}")
    merged["n_months"] = int(n_months)

    merged["var_spend"] = merged["actual_spend"] - merged["recommended_spend"]
    merged["var_leads"] = merged["actual_leads"] - merged["pred_leads"]
    merged["var_attempts"] = merged["actual_attempts"] - merged["pred_attempts"]
    merged["var_connected"] = merged["actual_connected"] - merged["pred_connected"]
    merged["var_contracts"] = merged["actual_contracts"] - merged["pred_contracts"]
    merged["var_premium"] = merged["actual_premium"] - merged["pred_premium"]

    merged["roi_plan"] = merged.apply(lambda r: _safe_div(r["pred_premium"], r["recommended_spend"]), axis=1)
    merged["roi_actual"] = merged.apply(lambda r: _safe_div(r["actual_premium"], r["actual_spend"]), axis=1)

    totals = {
        "period": f"{start}~{end}",
        "n_months": int(n_months),
        "recommended_spend": float(merged["recommended_spend"].sum()),
        "actual_spend": float(merged["actual_spend"].sum()),
        "pred_leads": float(merged["pred_leads"].sum()),
        "actual_leads": float(merged["actual_leads"].sum()),
        "pred_contracts": float(merged["pred_contracts"].sum()),
        "actual_contracts": float(merged["actual_contracts"].sum()),
        "pred_premium": float(merged["pred_premium"].sum()),
        "actual_premium": float(merged["actual_premium"].sum()),
    }
    return merged, totals


def _model_premium(f: dict) -> float:
    # premium = spend * (1/CPL) * AttemptRate * ConnectRate * CloseRate * APS
    spend = float(f.get("spend", 0.0))
    inv_cpl = float(f.get("inv_cpl", 0.0))
    attempt = float(f.get("attempt", 0.0))
    connect = float(f.get("connect", 0.0))
    close = float(f.get("close", 0.0))
    aps = float(f.get("aps", 0.0))
    return spend * inv_cpl * attempt * connect * close * aps


def shapley_decompose_premium(
    plan_metrics: dict,
    actual_metrics: dict,
) -> dict:
    """Shapley decomposition of premium gap into Spend, CPL, Attempt, Connect, Close, APS factors.

    Returns dict with contributions (sum == actual_premium - plan_premium).
    """
    # Build factor representations
    # Use inv_cpl to keep multiplicative form.
    p = {
        "spend": plan_metrics["spend"],
        "inv_cpl": _safe_div(1.0, plan_metrics["cpl"]) if plan_metrics["cpl"] > 0 else 0.0,
        "attempt": plan_metrics["attempt_rate"],
        "connect": plan_metrics["connect_rate"],
        "close": plan_metrics["close_rate"],
        "aps": plan_metrics["aps"],
    }
    a = {
        "spend": actual_metrics["spend"],
        "inv_cpl": _safe_div(1.0, actual_metrics["cpl"]) if actual_metrics["cpl"] > 0 else 0.0,
        "attempt": actual_metrics["attempt_rate"],
        "connect": actual_metrics["connect_rate"],
        "close": actual_metrics["close_rate"],
        "aps": actual_metrics["aps"],
    }

    keys = ["spend", "inv_cpl", "attempt", "connect", "close", "aps"]
    contrib = {"Spend": 0.0, "CPL": 0.0, "Attempt": 0.0, "Connect": 0.0, "Close": 0.0, "APS": 0.0}

    # If plan premium is 0 and actual premium is 0, all 0
    base_val = _model_premium(p)
    new_val = _model_premium(a)
    if abs(new_val - base_val) < 1e-12:
        return {k: 0.0 for k in contrib}

    # Shapley: average marginal contribution over all permutations
    for perm in permutations(keys):
        cur = dict(p)
        prev_val = _model_premium(cur)
        for k in perm:
            cur[k] = a[k]
            now_val = _model_premium(cur)
            delta = now_val - prev_val
            if k == "spend":
                contrib["Spend"] += delta
            elif k == "inv_cpl":
                contrib["CPL"] += delta
            elif k == "attempt":
                contrib["Attempt"] += delta
            elif k == "connect":
                contrib["Connect"] += delta
            elif k == "close":
                contrib["Close"] += delta
            elif k == "aps":
                contrib["APS"] += delta
            prev_val = now_val

    n_perm = float(len(list(permutations(keys))))
    for k in list(contrib.keys()):
        contrib[k] = contrib[k] / n_perm

    # Small numerical drift: force exact sum
    total = sum(contrib.values())
    gap = new_val - base_val
    drift = gap - total
    contrib["APS"] += drift
    return contrib


def explain_gap_table(cmp_df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-channel explanation table with factor contributions and ranking."""
    rows = []
    for _, r in cmp_df.iterrows():
        ch = r.get("channel")
        plan_m = _metrics_from_row(
            r.get("recommended_spend", 0.0),
            r.get("pred_leads", 0.0),
            r.get("pred_attempts", 0.0),
            r.get("pred_connected", 0.0),
            r.get("pred_contracts", 0.0),
            r.get("pred_premium", 0.0),
        )
        act_m = _metrics_from_row(
            r.get("actual_spend", 0.0),
            r.get("actual_leads", 0.0),
            r.get("actual_attempts", 0.0),
            r.get("actual_connected", 0.0),
            r.get("actual_contracts", 0.0),
            r.get("actual_premium", 0.0),
        )
        contrib = shapley_decompose_premium(plan_m, act_m)
        gap = float(act_m["premium"] - plan_m["premium"])
        # rank by abs contribution
        ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_reason = ranked[0][0] if ranked else "-"
        rows.append({
            "channel": ch,
            "premium_gap": gap,
            "contrib_spend": contrib["Spend"],
            "contrib_cpl": contrib["CPL"],
            "contrib_attempt": contrib["Attempt"],
            "contrib_connect": contrib["Connect"],
            "contrib_close": contrib["Close"],
            "contrib_aps": contrib["APS"],
            "top_reason": top_reason,
            "roi_plan": plan_m["roi"],
            "roi_actual": act_m["roi"],
            "cpl_plan": plan_m["cpl"],
            "cpl_actual": act_m["cpl"],
            "attempt_plan": plan_m["attempt_rate"],
            "attempt_actual": act_m["attempt_rate"],
            "connect_plan": plan_m["connect_rate"],
            "connect_actual": act_m["connect_rate"],
            "close_plan": plan_m["close_rate"],
            "close_actual": act_m["close_rate"],
            "aps_plan": plan_m["aps"],
            "aps_actual": act_m["aps"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("premium_gap", ascending=True)
    return out
