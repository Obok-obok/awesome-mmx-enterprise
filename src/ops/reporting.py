from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class WeeklyReportConfig:
    days: int = 7


def _fmt_money(x: float) -> str:
    try:
        return f"{x:,.0f}"
    except Exception:
        return str(x)


def generate_weekly_pdf(
    out_pdf: Path,
    perf_path: Path,
    reco_hist_path: Path,
    ab_results_path: Optional[Path] = None,
    geo_mm_results_path: Optional[Path] = None,
    cfg: WeeklyReportConfig = WeeklyReportConfig(),
) -> Path:
    """Create a 1-page executive-style PDF summarizing Ops Monitoring for the last N days.

    Layout is intentionally fixed so it is "임원 보고 규격"-friendly.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    if not perf_path.exists():
        raise FileNotFoundError(f"Missing performance log: {perf_path}")

    perf = pd.read_csv(perf_path)
    perf["date"] = pd.to_datetime(perf["date"])
    perf = perf.sort_values("date")
    last = perf.tail(cfg.days)

    # Load latest reco (next-day) if exists
    reco_last = None
    if reco_hist_path.exists():
        reco = pd.read_csv(reco_hist_path)
        if len(reco) > 0:
            reco_last = reco.sort_values("date").tail(20)

    ab_last = None
    if ab_results_path and ab_results_path.exists():
        ab = pd.read_csv(ab_results_path)
        if len(ab) > 0:
            ab_last = ab.sort_values("week_end").tail(1)

    # Fixed 1-page template
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    w, h = A4

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(18 * mm, h - 18 * mm, "MMX Ops Weekly (1-page)")
    c.setFont("Helvetica", 9)
    date_range = f"{last['date'].min().date()} ~ {last['date'].max().date()}"
    c.drawRightString(w - 18 * mm, h - 18 * mm, f"Period: {date_range}")
    c.line(18 * mm, h - 21 * mm, w - 18 * mm, h - 21 * mm)

    # KPI box
    act = float(last["actual_premium"].sum())
    pred = float(last["pred_premium_mean"].sum())
    gap = act - pred
    drift_days = int(last["drift_flag"].astype(int).sum()) if "drift_flag" in last.columns else 0

    c.setFont("Helvetica-Bold", 11)
    c.drawString(18 * mm, h - 32 * mm, "1) KPI Summary (7D)")
    c.setFont("Helvetica", 10)
    c.drawString(22 * mm, h - 40 * mm, f"Actual Premium: {_fmt_money(act)}")
    c.drawString(82 * mm, h - 40 * mm, f"Pred Mean: {_fmt_money(pred)}")
    c.drawString(132 * mm, h - 40 * mm, f"Gap: {_fmt_money(gap)}")
    c.setFont("Helvetica", 9)
    c.drawString(22 * mm, h - 46 * mm, f"Drift days: {drift_days}")

    # Daily table
    c.setFont("Helvetica-Bold", 11)
    c.drawString(18 * mm, h - 58 * mm, "2) Daily Actual vs Prediction")
    headers = ["Date", "Actual", "PredMean", "CVaR10", "DriftZ"]
    colx = [18, 55, 85, 115, 145]
    y = h - 66 * mm
    c.setFont("Helvetica", 8.5)
    for i, hh in enumerate(headers):
        c.drawString(colx[i] * mm, y, hh)
    y -= 4 * mm
    c.line(18 * mm, y, w - 18 * mm, y)
    y -= 6 * mm
    for _, r in last.iterrows():
        c.drawString(colx[0] * mm, y, str(r["date"].date()))
        c.drawRightString((colx[1] + 18) * mm, y, _fmt_money(float(r.get("actual_premium", 0.0))))
        c.drawRightString((colx[2] + 18) * mm, y, _fmt_money(float(r.get("pred_premium_mean", 0.0))))
        c.drawRightString((colx[3] + 18) * mm, y, _fmt_money(float(r.get("pred_premium_cvar10", 0.0))))
        c.drawRightString((colx[4] + 18) * mm, y, f"{float(r.get('drift_z', 0.0)):.2f}")
        y -= 5.3 * mm

    # Experiment results (A/B + Geo MM) in 2 columns
    left_x = 18 * mm
    right_x = (w / 2) + 3 * mm
    top_y = h - 112 * mm
    box_h = 38 * mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(left_x, top_y, "3) Experiments")

    # A/B (left)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(left_x, top_y - 7 * mm, "A/B Holdout")
    c.setFont("Helvetica", 8.5)
    if ab_last is None:
        c.drawString(left_x, top_y - 13 * mm, "No weekly A/B result")
    else:
        row = ab_last.iloc[0].to_dict()
        c.drawString(left_x, top_y - 13 * mm, f"Week end: {row.get('week_end', '')}")
        c.drawString(left_x, top_y - 18 * mm, f"Channels: {row.get('channels', '')}")
        c.drawString(left_x, top_y - 23 * mm, f"Lift(premium): {_fmt_money(float(row.get('lift_premium', 0.0)))}")
        note = str(row.get("notes", ""))
        c.drawString(left_x, top_y - 28 * mm, f"Notes: {note[:38]}")

    # Geo MM (right)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(right_x, top_y - 7 * mm, "Geo Holdout (Matched Market)")
    c.setFont("Helvetica", 8.5)
    geo_rows = None
    if geo_mm_results_path and geo_mm_results_path.exists():
        try:
            geo_mm = pd.read_csv(geo_mm_results_path)
            if len(geo_mm) > 0:
                geo_mm["week_end"] = geo_mm["week_end"].astype(str)
                we = str(pd.to_datetime(geo_mm["week_end"]).max().date())
                geo_rows = geo_mm[geo_mm["week_end"] == we]
        except Exception:
            geo_rows = None

    if geo_rows is None or len(geo_rows) == 0:
        c.drawString(right_x, top_y - 13 * mm, "No geo MM weekly result")
    else:
        # Expect rows per metric in value_col
        def _lift_for(metric: str) -> str:
            sub = geo_rows[geo_rows["value_col"] == metric]
            if len(sub) == 0:
                return "NA"
            v = float(sub.iloc[0]["lift_total"])
            return _fmt_money(v)

        c.drawString(right_x, top_y - 13 * mm, f"week_end={str(geo_rows['week_end'].iloc[0])}  pairs={int(geo_rows['n_pairs'].max())}")
        c.drawString(right_x, top_y - 18 * mm, f"lift premium:  {_lift_for('premium')}")
        c.drawString(right_x, top_y - 23 * mm, f"lift contracts: {_lift_for('contracts')}")
        c.drawString(right_x, top_y - 28 * mm, f"lift LTV:      {_lift_for('ltv')}")
        c.drawString(right_x, top_y - 28 * mm, f"Notes: {str(row.get('notes',''))[:38]}")

    # Latest recommendation snapshot (bottom)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(18 * mm, h - 156 * mm, "4) Latest Budget Recommendation")
    c.setFont("Helvetica", 8.5)
    if reco_last is None:
        c.drawString(18 * mm, h - 162 * mm, "No recommendation history")
    else:
        latest_date = str(pd.to_datetime(reco_last["date"]).max().date())
        latest = reco_last[pd.to_datetime(reco_last["date"]).dt.date == pd.to_datetime(latest_date).date()].copy()
        latest = latest.sort_values("budget", ascending=False)
        c.drawString(18 * mm, h - 162 * mm, f"Reco date: {latest_date}")
        y2 = h - 168 * mm
        for _, rr in latest.head(8).iterrows():
            c.drawString(22 * mm, y2, str(rr["channel"]))
            c.drawRightString(w - 18 * mm, y2, _fmt_money(float(rr["budget"])))
            y2 -= 5 * mm

    c.save()
    return out_pdf
