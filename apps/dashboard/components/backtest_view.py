from __future__ import annotations

"""Backtest section rendering.

UI is *read-only*: it never synthesizes numbers. It only reads artifacts from
artifacts/backtests/latest.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from .ui import fmt_money, fmt_percent, fmt_float, kpi_row, section


@dataclass(frozen=True)
class BacktestViewModel:
    run_id: str
    created_at: str
    train_range: str
    test_range: str
    policy_lambda: float
    overall: dict
    ts_daily: pd.DataFrame
    by_channel: pd.DataFrame
    plan_compare_by_channel: pd.DataFrame
    plan_compare_totals: dict
    plan_compare_monthly_by_channel: pd.DataFrame
    plan_compare_monthly_totals: dict
    period_summary: pd.DataFrame
    lineage: dict
    config: dict


def load_backtest_vm(artifacts_latest_dir: Path) -> BacktestViewModel | None:
    d = Path(artifacts_latest_dir)
    if not d.exists():
        return None
    cfg_p = d / "config.json"
    spl_p = d / "splits.json"
    met_p = d / "metrics_overall.json"
    ts_p = d / "timeseries_daily.csv"
    by_p = d / "metrics_by_channel.csv"
    lin_p = d / "lineage.json"
    pcb_p = d / "plan_compare_by_channel.csv"
    pct_p = d / "plan_compare_totals.json"
    pcm_p = d / "plan_compare_monthly_by_channel.csv"
    pcmt_p = d / "plan_compare_monthly_totals.json"
    ps_p = d / "period_summary.csv"
    if not (cfg_p.exists() and spl_p.exists() and met_p.exists() and ts_p.exists() and by_p.exists() and lin_p.exists()):
        return None
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    spl = json.loads(spl_p.read_text(encoding="utf-8"))
    met = json.loads(met_p.read_text(encoding="utf-8"))
    ts = pd.read_csv(ts_p)
    by = pd.read_csv(by_p)
    lin = json.loads(lin_p.read_text(encoding="utf-8"))
    pcb = pd.read_csv(pcb_p) if pcb_p.exists() else pd.DataFrame()
    pct = json.loads(pct_p.read_text(encoding="utf-8")) if pct_p.exists() else {}
    pcm = pd.read_csv(pcm_p) if pcm_p.exists() else pd.DataFrame()
    pcmt = json.loads(pcmt_p.read_text(encoding="utf-8")) if pcmt_p.exists() else {}
    ps = pd.read_csv(ps_p) if ps_p.exists() else pd.DataFrame()

    train = spl.get("train", {})
    test = spl.get("test", {})
    return BacktestViewModel(
        run_id=str(cfg.get("run_id", d.name)),
        created_at=str(cfg.get("created_at", "")),
        train_range=f"{train.get('start','')} ~ {train.get('end','')}",
        test_range=f"{test.get('start','')} ~ {test.get('end','')}",
        policy_lambda=float(cfg.get("policy", {}).get("lambda", 0.0)),
        overall=met,
        ts_daily=ts,
        by_channel=by,
        plan_compare_by_channel=pcb,
        plan_compare_totals=pct,
        plan_compare_monthly_by_channel=pcm,
        plan_compare_monthly_totals=pcmt,
        period_summary=ps,
        lineage=lin,
        config=cfg,
    )



def render_backtest_section(vm: BacktestViewModel | None) -> None:
    """Render the Backtest tab.

    Design goals:
      - Minimal sections (Spend Allocation -> KPI Amount/Rate table).
      - Metric-first comparison (Actual vs Pred side-by-side).
      - Totals are computed as Σ (amount) or recomputed ratios (rates / unit KPIs).
    """
    section("Backtest (Actual vs Predicted)")

    if vm is None:
        st.info("백테스트 결과 아티팩트가 없습니다. scripts/run_backtest.py 실행 후 다시 확인하세요.")
        return

    st.caption(
        f"Run: {vm.run_id}  |  Train: {vm.train_range}  |  Test: {vm.test_range}  |  λ={vm.policy_lambda}"
    )

    # Debug/audit badges for latent-quality safety rails (kept compact).
    semg = (vm.config or {}).get("sem_globals", {}) or {}
    if semg:
        lq_lo = semg.get("lq_clip_low")
        lq_hi = semg.get("lq_clip_high")
        lq_a = semg.get("lq_alpha_rate")
        gppc = semg.get("gamma_ppc_lq")
        badges = []
        if lq_lo is not None and lq_hi is not None:
            badges.append(f"LQ clip [{fmt_float(lq_lo)}, {fmt_float(lq_hi)}]")
        if lq_a is not None:
            badges.append(f"α={fmt_float(lq_a)}")
        if gppc is not None:
            badges.append(f"γ_ppc={fmt_float(gppc)}")
        if badges:
            st.caption(" · ".join(badges))

    # ---------------------------------------------------------------------
    # Period selector (monthly operation)
    # ---------------------------------------------------------------------
    periods: list[str] = []
    if vm.period_summary is not None and not vm.period_summary.empty and "period" in vm.period_summary.columns:
        periods = sorted([p for p in vm.period_summary["period"].dropna().astype(str).unique().tolist()])
    if not periods and vm.plan_compare_monthly_totals:
        periods = sorted(list((vm.plan_compare_monthly_totals.get("periods", {}) or {}).keys()))
    period_options = ["전체(Test)"] + periods
    selected_period = st.selectbox("기간 선택", options=period_options, index=0)

    show_opt = st.checkbox("모델 배분(Optimized)도 함께 보기", value=True)
    show_roi_sat = st.checkbox("ROI / Saturation(포화도)도 함께 보기", value=False)

    # Resolve totals (actual + predictions) for selected period
    if selected_period == "전체(Test)":
        p = vm.plan_compare_totals or {}
        actual = (p.get("actual") or {}) if isinstance(p, dict) else {}
        pred_a = (p.get("pred_on_actual_plan") or {}) if isinstance(p, dict) else {}
        pred_o = (p.get("pred_on_opt_plan") or {}) if isinstance(p, dict) else {}
        alloc_df = vm.plan_compare_by_channel.copy() if vm.plan_compare_by_channel is not None else pd.DataFrame()
    else:
        p = ((vm.plan_compare_monthly_totals.get("periods", {}) or {}).get(selected_period, {}) or {})
        actual = p.get("actual", {}) or {}
        pred_a = p.get("pred_on_actual_plan", {}) or {}
        pred_o = p.get("pred_on_opt_plan", {}) or {}
        alloc_df = (
            vm.plan_compare_monthly_by_channel.loc[vm.plan_compare_monthly_by_channel["period"].astype(str) == str(selected_period)].copy()
            if vm.plan_compare_monthly_by_channel is not None and not vm.plan_compare_monthly_by_channel.empty
            else pd.DataFrame()
        )

    # ---------------------------------------------------------------------
    # Section A: Spend allocation (channel)
    # ---------------------------------------------------------------------
    st.subheader("1) 채널별 Spend 배분")
    if alloc_df is None or alloc_df.empty:
        st.warning("Spend 배분 아티팩트를 찾지 못했습니다. (plan_compare_by_channel*.csv)")
    else:
        # Keep the table compact: spend + share (and optimized if enabled)
        cols = ["channel", "actual_spend", "actual_share"]
        if show_roi_sat:
            for c in ["actual_roi", "pred_roi_on_actual", "sat_ratio_est_actual"]:
                if c in alloc_df.columns:
                    cols.append(c)
        if show_opt and "opt_spend" in alloc_df.columns and "opt_share" in alloc_df.columns:
            cols += ["opt_spend", "opt_share"]
            if show_roi_sat:
                for c in ["pred_roi_on_opt", "sat_ratio_est_opt"]:
                    if c in alloc_df.columns:
                        cols.append(c)

        t = alloc_df[cols].copy()
        # Totals
        total_row = {
            "channel": "TOTAL",
            "actual_spend": float(t["actual_spend"].sum()) if "actual_spend" in t.columns else 0.0,
            "actual_share": 1.0,
        }
        if show_roi_sat:
            # Total ROI by sums (preferred) or leave blank if insufficient columns
            if "actual_roi" in t.columns and "actual_premium" in alloc_df.columns and "actual_spend" in alloc_df.columns:
                prem_sum = float(alloc_df["actual_premium"].sum())
                spend_sum = float(alloc_df["actual_spend"].sum())
                total_row["actual_roi"] = (prem_sum / spend_sum) if spend_sum > 0 else 0.0
            if "pred_roi_on_actual" in t.columns and "pred_premium_on_actual" in alloc_df.columns and "actual_spend" in alloc_df.columns:
                prem_sum = float(alloc_df["pred_premium_on_actual"].sum())
                spend_sum = float(alloc_df["actual_spend"].sum())
                total_row["pred_roi_on_actual"] = (prem_sum / spend_sum) if spend_sum > 0 else 0.0
            if "sat_ratio_est_actual" in t.columns and "actual_spend" in alloc_df.columns:
                spend = alloc_df["actual_spend"].astype(float)
                sat = alloc_df["sat_ratio_est_actual"].astype(float)
                denom = float(spend.sum())
                total_row["sat_ratio_est_actual"] = float((spend * sat).sum() / denom) if denom > 0 else 0.0
        if show_opt and "opt_spend" in t.columns:
            total_row["opt_spend"] = float(t["opt_spend"].sum())
            total_row["opt_share"] = 1.0
            if show_roi_sat and "pred_roi_on_opt" in t.columns and "pred_premium_on_opt" in alloc_df.columns and "opt_spend" in alloc_df.columns:
                prem_sum = float(alloc_df["pred_premium_on_opt"].sum())
                spend_sum = float(alloc_df["opt_spend"].sum())
                total_row["pred_roi_on_opt"] = (prem_sum / spend_sum) if spend_sum > 0 else 0.0
            if show_roi_sat and "sat_ratio_est_opt" in t.columns and "opt_spend" in alloc_df.columns:
                spend = alloc_df["opt_spend"].astype(float)
                sat = alloc_df["sat_ratio_est_opt"].astype(float)
                denom = float(spend.sum())
                total_row["sat_ratio_est_opt"] = float((spend * sat).sum() / denom) if denom > 0 else 0.0
        t = pd.concat([t, pd.DataFrame([total_row])], ignore_index=True)

        # Formatting
        def _fmt_money(x: float) -> str:
            try:
                return fmt_money(float(x))
            except Exception:
                return str(x)

        def _fmt_pct(x: float) -> str:
            try:
                return fmt_percent(float(x))
            except Exception:
                return str(x)

        fmt_map = {
            "actual_spend": _fmt_money,
            "actual_share": _fmt_pct,
        }
        if show_roi_sat:
            fmt_map.update({
                "actual_roi": lambda x: fmt_float(float(x), 3),
                "pred_roi_on_actual": lambda x: fmt_float(float(x), 3),
                "sat_ratio_est_actual": lambda x: fmt_percent(float(x)),
            })
        if show_opt:
            fmt_map.update({"opt_spend": _fmt_money, "opt_share": _fmt_pct})
            if show_roi_sat:
                fmt_map.update({
                    "pred_roi_on_opt": lambda x: fmt_float(float(x), 3),
                    "sat_ratio_est_opt": lambda x: fmt_percent(float(x)),
                })

        st.dataframe(
            t.style.format(fmt_map),
            use_container_width=True,
            hide_index=True,
        )

    # NOTE: We intentionally avoid extra charts here.
    # Backtest UX is table-first; charts tended to duplicate information and
    # make the page noisy.

    # ---------------------------------------------------------------------
    # Section B: KPI (Amount + Rate) metric-first comparison
    # ---------------------------------------------------------------------
    st.subheader("2) 퍼널 KPI: Actual vs Predicted (지표별 비교)")

    def _safe_div(num: float, den: float) -> float | None:
        if den == 0:
            return None
        return num / den

    # Base amount totals
    spend_a = float(actual.get("spend", 0.0))
    leads_a = float(actual.get("leads", 0.0))
    att_a = float(actual.get("attempts", 0.0))
    con_a = float(actual.get("connected", 0.0))
    ctr_a = float(actual.get("contracts", 0.0))
    prem_a = float(actual.get("premium", 0.0))

    spend_p = float(pred_a.get("spend", spend_a))
    leads_p = float(pred_a.get("leads", 0.0))
    att_p = float(pred_a.get("attempts", 0.0))
    con_p = float(pred_a.get("connected", 0.0))
    ctr_p = float(pred_a.get("contracts", 0.0))
    prem_p = float(pred_a.get("premium", 0.0))

    spend_o = float(pred_o.get("spend", spend_a))
    leads_o = float(pred_o.get("leads", 0.0))
    att_o = float(pred_o.get("attempts", 0.0))
    con_o = float(pred_o.get("connected", 0.0))
    ctr_o = float(pred_o.get("contracts", 0.0))
    prem_o = float(pred_o.get("premium", 0.0))

    def _row(metric: str, unit: str, a: float | None, p: float | None, p_opt: float | None) -> dict:
        a_v = None if a is None else float(a)
        p_v = None if p is None else float(p)
        d_v = None
        e_v = None
        if a_v is not None and p_v is not None:
            d_v = p_v - a_v
            e_v = None if a_v == 0 else (p_v - a_v) / a_v
        row = {
            "Metric": metric,
            "Unit": unit,
            "Actual": a_v,
            "Pred": p_v,
            "Δ": d_v,
            "Error%": e_v,
        }
        if show_opt:
            row["Pred(Opt)"] = None if p_opt is None else float(p_opt)
        return row

    rows: list[dict] = []

    # Amount block (Σ)
    rows.append({"Metric": "— Amount —", "Unit": "", "Actual": None, "Pred": None, "Δ": None, "Error%": None, **({"Pred(Opt)": None} if show_opt else {})})
    rows += [
        _row("Spend", "₩", spend_a, spend_p, spend_o),
        _row("Premium", "₩", prem_a, prem_p, prem_o),
        _row("Premium / Contract", "₩", _safe_div(prem_a, ctr_a), _safe_div(prem_p, ctr_p), _safe_div(prem_o, ctr_o)),
        _row("Leads", "cnt", leads_a, leads_p, leads_o),
        _row("Attempts", "cnt", att_a, att_p, att_o),
        _row("Connected", "cnt", con_a, con_p, con_o),
        _row("Contracts", "cnt", ctr_a, ctr_p, ctr_o),
    ]

    # Rate block (recomputed from totals)
    rows.append({"Metric": "— Rates —", "Unit": "", "Actual": None, "Pred": None, "Δ": None, "Error%": None, **({"Pred(Opt)": None} if show_opt else {})})
    rows += [
        _row("Attempt Rate (Attempts / Leads)", "%", _safe_div(att_a, leads_a), _safe_div(att_p, leads_p), _safe_div(att_o, leads_o)),
        _row("Connect Rate (Connected / Attempts)", "%", _safe_div(con_a, att_a), _safe_div(con_p, att_p), _safe_div(con_o, att_o)),
        _row("Contract Rate (Contracts / Connected)", "%", _safe_div(ctr_a, con_a), _safe_div(ctr_p, con_p), _safe_div(ctr_o, con_o)),
    ]

    # Unit economics block (recomputed)
    rows.append({"Metric": "— Unit Economics —", "Unit": "", "Actual": None, "Pred": None, "Δ": None, "Error%": None, **({"Pred(Opt)": None} if show_opt else {})})
    rows += [
        _row("CPL (Spend / Lead)", "₩", _safe_div(spend_a, leads_a), _safe_div(spend_p, leads_p), _safe_div(spend_o, leads_o)),
        _row("CPA (Spend / Contract)", "₩", _safe_div(spend_a, ctr_a), _safe_div(spend_p, ctr_p), _safe_div(spend_o, ctr_o)),
    ]

    kpi_df = pd.DataFrame(rows)

    # Formatters
    def _fmt_val(metric: str, unit: str, v: float | None) -> str:
        if v is None or (isinstance(v, float) and (pd.isna(v))):
            return "—"
        if unit == "%":
            return fmt_percent(float(v))
        if unit == "₩":
            return fmt_money(float(v))
        return f"{float(v):,.0f}"

    def _fmt_err(v: float | None) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return fmt_percent(float(v))

    # Render as a compact, readable table (pre-formatted strings) to avoid wide layouts.
    disp = kpi_df.copy()
    def _fmt_cell(row_idx: int, col: str, v: float | None) -> str:
        unit = str(disp.loc[row_idx, 'Unit'])
        metric = str(disp.loc[row_idx, 'Metric'])
        if metric.startswith('—'):
            return ''
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return '—'
        if col == 'Error%':
            return fmt_percent(float(v))
        if unit == '%':
            return fmt_percent(float(v))
        if unit == '₩':
            return fmt_money(float(v))
        return f"{float(v):,.0f}"

    for c in ['Actual','Pred','Δ','Error%'] + (['Pred(Opt)'] if show_opt else []):
        disp[c] = [ _fmt_cell(i, c, disp.loc[i, c]) for i in range(len(disp)) ]

    st.dataframe(disp[['Metric','Unit','Actual','Pred'] + (['Pred(Opt)'] if show_opt else []) + ['Δ','Error%']],
                 use_container_width=True, hide_index=True)

    # NOTE: No charts here by design (tables already include the comparison).

    # ---------------------------------------------------------------------
    # Minimal provenance (collapsed)
    # ---------------------------------------------------------------------
    with st.expander("Lineage / Provenance", expanded=False):
        st.json(vm.lineage)