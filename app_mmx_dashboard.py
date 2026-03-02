import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from mmx.executive import render_executive_summary
from mmx.optimizer import render_optimizer_engine
from mmx.metrics_doc import render_metrics_dictionary
from mmx.tracking import (list_runs, load_plan_by_channel, compare_plan_vs_actual, compare_plan_vs_actual_period, validate_actuals_df, explain_gap_table, months_inclusive)
from mmx.formatting import format_won, format_ratio


def read_monthly_premium_target(path: str = "config/business_config.yaml") -> float:
    """Read fixed Premium target from config (no external deps).

    Expected line: monthly_target_premium_default: <number>
    """
    try:
        if not os.path.exists(path):
            return 0.0
        for line in open(path, "r", encoding="utf-8"):
            s = (line or "").strip()
            if s.startswith("monthly_target_premium_default"):
                # monthly_target_premium_default: 23000000
                parts = s.split(":", 1)
                if len(parts) == 2:
                    return float(str(parts[1]).strip())
        return 0.0
    except Exception:
        return 0.0

# =========================================================
# MMX Executive Dashboard (Business Decision Flow)
# - 1) Executive Summary (Decide)
# - 2) Drivers & Decomposition (Diagnose)
# - 3) Model & Confidence (Explain)
# - 4) Budget Optimizer (Act)
# - 5) Monitoring & Data Quality (Learn)
# =========================================================

st.set_page_config(page_title="MMX Executive Dashboard", layout="wide")

# ---------- Styling (clean, standardized) ----------
BCG_CSS = """
<style>
:root{
  --bg:#ffffff; --ink:#0b0f19; --muted:#6b7280; --line:#e5e7eb;
  --accent:#0b1f44; --warn:#b42318; --mid:#b45309; --ok:#0f766e;
}
html, body, [class*="css"] { background: var(--bg); color: var(--ink); }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1320px; }
h1,h2,h3 { letter-spacing: -0.02em; }
.smallcap { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
.hr { border-top: 1px solid var(--line); margin: 14px 0 18px 0; }
.card { border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px; background: #fff; }
.kpi { font-size: 26px; font-weight: 750; line-height: 1.2; }
.kpi_sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
.note { color: var(--muted); font-size: 12px; }
.badge { display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px;
         border: 1px solid var(--line); background:#fff; }
.badge-warn{ border-color: rgba(180,35,24,.35); color: var(--warn); background: rgba(180,35,24,.06); }
.badge-mid { border-color: rgba(180,83,9,.35); color: var(--mid); background: rgba(180,83,9,.06); }
.badge-ok  { border-color: rgba(15,118,110,.35); color: var(--ok);  background: rgba(15,118,110,.06); }
.pill { display:inline-block; padding: 6px 12px; border-radius: 999px; font-size: 12px;
        border: 1px solid var(--line); background:#fff; margin-right: 6px; }
.pill-approve{ border-color: rgba(15,118,110,.35); color: var(--ok); background: rgba(15,118,110,.06); }
.pill-hold{ border-color: rgba(180,83,9,.35); color: var(--mid); background: rgba(180,83,9,.06); }
.pill-reject{ border-color: rgba(180,35,24,.35); color: var(--warn); background: rgba(180,35,24,.06); }
table { font-size: 13px; }
</style>
"""
st.markdown(BCG_CSS, unsafe_allow_html=True)

# ---------- Helpers ----------
def badge(label: str, kind: str = "ok"):
    cls = {"warn":"badge badge-warn","mid":"badge badge-mid","ok":"badge badge-ok"}.get(kind,"badge")
    st.markdown(f"<span class='{cls}'>{label}</span>", unsafe_allow_html=True)

def money(x) -> str:
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "-"
        return f"{float(x):,.0f}"
    except Exception:
        return "-"

def pct(x, d=1) -> str:
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "-"
        return f"{float(x)*100:.{d}f}%"
    except Exception:
        return "-"

def safe_div(a, b):
    b = 0 if b is None else b
    if b == 0:
        return np.nan
    return a / b

def clamp(x, lo, hi):
    try:
        return max(lo, min(hi, x))
    except Exception:
        return x

def pill(label: str, kind: str):
    cls = {
        "approve": "pill pill-approve",
        "hold": "pill pill-hold",
        "reject": "pill pill-reject",
    }.get(kind, "pill")
    st.markdown(f"<span class='{cls}'>{label}</span>", unsafe_allow_html=True)

def read_csv(path: str):
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return None
        return df
    except Exception:
        return None

def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ---------- Data loading (prefer /out artifacts) ----------
PANEL_CH_PATHS = [
    "out/panel_monthly_channel.csv",
    "out/panel_daily_channel.csv",
    "out/dynamic/panel_daily_channel.csv",
    "data/input_daily_channel.csv",
]

EXEC_SUMMARY_PATHS = [
    "out/executive_summary.json",
]

DQ_PATHS = [
    "out/data_quality_report.json",
]

POST_CH_PATHS = [
    "out/posterior_summary_channel.csv",
]

CF_CH_PATHS = [
    "out/counterfactuals_channel.csv",
]

BUDGET_PATHS = [
    "out/budget_recommendation.csv",
]

LINEAGE_PATHS = [
    "out/metric_lineage.csv",
]

def load_first(paths, reader):
    for p in paths:
        obj = reader(p)
        if obj is not None:
            return obj, p
    return None, None

panel_ch, panel_src = load_first(PANEL_CH_PATHS, read_csv)
exec_sum, exec_src = load_first(EXEC_SUMMARY_PATHS, read_json)
dq, dq_src = load_first(DQ_PATHS, read_json)
post_ch, post_src = load_first(POST_CH_PATHS, read_csv)
cf_ch, cf_src = load_first(CF_CH_PATHS, read_csv)
bud, bud_src = load_first(BUDGET_PATHS, read_csv)
lineage, lin_src = load_first(LINEAGE_PATHS, read_csv)

if panel_ch is None:
    st.error("필수 데이터(out/panel_monthly_channel.csv / out/panel_daily_channel.csv 또는 data/input_daily_channel.csv)가 없음. scripts/run_all.sh 실행 후 다시 시도 필요함.")
    st.stop()

# ---------- Normalize types ----------
panel_ch = panel_ch.copy()
panel_ch["date"] = pd.to_datetime(panel_ch["date"]).dt.date
panel_ch["channel"] = panel_ch["channel"].astype(str)

# Detect granularity (monthly vs daily) to set default window / labels
_dates_sorted = sorted(pd.to_datetime(panel_ch["date"]).unique())
_median_gap_days = 1
if len(_dates_sorted) >= 3:
    _deltas = np.diff(np.array(_dates_sorted, dtype="datetime64[D]"))
    try:
        _median_gap_days = int(np.median(_deltas).astype(int))
    except Exception:
        _median_gap_days = 1
is_monthly = _median_gap_days >= 20

# Default analysis window:
# - Daily panel: last 30 days
# - Monthly panel: last 6 months
max_date = pd.to_datetime(panel_ch["date"]).max().date()
default_start = max_date - (timedelta(days=29) if not is_monthly else timedelta(days=183))

# ---------- Top header ----------
st.markdown("# MMX Executive Dashboard")
st.markdown(
    f"<div class='note'>분석 기준일: {max_date} · 기본 윈도우: {'최근 6개월' if is_monthly else '최근 30일'} · 데이터: {panel_src}</div>",
    unsafe_allow_html=True
)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Global periods (shared across all pages) ----------
# Three distinct periods:
# - Reporting period: what you are viewing as "actuals" in the dashboard
# - Optimization history period: the data window used for estimation/optimization
# - Forecast / impact period: the period the optimized plan is assumed to impact (for scaling & tracking)

def _to_date(x):
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None

def _month_str(d):
    # YYYY-MM
    return pd.to_datetime(d).strftime("%Y-%m")

def _month_range(start_d, end_d):
    # inclusive months
    s = pd.to_datetime(start_d).to_period("M")
    e = pd.to_datetime(end_d).to_period("M")
    if e < s:
        return []
    return [str(p) for p in pd.period_range(s, e, freq="M")]

# Defaults based on data
report_default_start = default_start
report_default_end = max_date

# Optimization history defaults (a bit longer than report window when possible)
hist_default_start = max_date - (timedelta(days=365) if is_monthly else timedelta(days=89))
hist_default_end = max_date

# Forecast/impact default: next month (monthly) or next 30 days (daily)
if is_monthly:
    _next_month_start = (pd.to_datetime(max_date).to_period("M") + 1).to_timestamp().date()
    _next_month_end = (pd.to_datetime(_next_month_start).to_period("M")).to_timestamp(how="end").date()
    fc_default_start = _next_month_start
    fc_default_end = _next_month_end
else:
    fc_default_start = max_date + timedelta(days=1)
    fc_default_end = max_date + timedelta(days=30)

# Sidebar: global settings (shared)
st.sidebar.markdown("## Global Settings")
st.sidebar.markdown("<div class='smallcap'>모든 페이지에 공통 적용되는 기간/대상 설정</div>", unsafe_allow_html=True)

# Fixed business target (read-only in UI)
PREMIUM_TARGET = read_monthly_premium_target()
st.sidebar.markdown(
    f"<div class='card'><div class='smallcap'>Fixed Premium Target</div><div class='kpi'>{format_won(PREMIUM_TARGET)}</div><div class='kpi_sub'>고정(설정 파일 기반)</div></div>",
    unsafe_allow_html=True,
)

with st.sidebar.expander("📅 기간 설정 (공통)", expanded=True):
    st.markdown("**1) 이번달/리포트 대상 기간 (Actuals View)**")
    report_start = st.date_input("리포트 기간 시작", value=report_default_start, key="report_start")
    report_end = st.date_input("리포트 기간 종료", value=report_default_end, key="report_end")

    st.markdown("---")
    st.markdown("**2) 옵티마이제이션에 사용하는 과거 데이터 기간 (History for Optimization)**")
    hist_start = st.date_input("최적화 과거기간 시작", value=hist_default_start, key="hist_start")
    hist_end = st.date_input("최적화 과거기간 종료", value=hist_default_end, key="hist_end")

    st.markdown("---")
    st.markdown("**3) 최적화 결과가 반영되는 성과 기간 (Forecast / Impact)**")
    fc_start = st.date_input("성과기간 시작", value=fc_default_start, key="fc_start")
    fc_end = st.date_input("성과기간 종료", value=fc_default_end, key="fc_end")

# Validate periods (fail fast, no tracebacks)
def _assert_periods_ok():
    errs = []
    if report_start > report_end:
        errs.append("리포트 기간 시작이 종료보다 늦음.")
    if hist_start > hist_end:
        errs.append("최적화 과거기간 시작이 종료보다 늦음.")
    if fc_start > fc_end:
        errs.append("성과기간 시작이 종료보다 늦음.")
    # Optional sanity: history should end not after report_end by too much (allow equal)
    # but do not enforce hard since some users may want different.
    if errs:
        st.error("기간 설정 오류: " + " / ".join(errs))
        st.stop()

_assert_periods_ok()

# Channels (global)
channels = sorted(panel_ch["channel"].unique().tolist())
selected_channels = st.sidebar.multiselect("채널 선택", options=channels, default=channels, key="selected_channels")

# Data windows
mask_report = (panel_ch["date"] >= report_start) & (panel_ch["date"] <= report_end) & (panel_ch["channel"].isin(selected_channels))
mask_hist = (panel_ch["date"] >= hist_start) & (panel_ch["date"] <= hist_end) & (panel_ch["channel"].isin(selected_channels))

df_report = panel_ch.loc[mask_report].copy()
df_hist = panel_ch.loc[mask_hist].copy()

# ---------- KPI aggregation (reporting window) ----------
tot_spend = float(df_report["spend"].sum())
tot_prem = float(df_report.get("premium", pd.Series([0]*len(df_report))).sum())
tot_leads = float(df_report.get("leads", pd.Series([0]*len(df_report))).sum())
tot_contracts = float(df_report.get("contracts", pd.Series([0]*len(df_report))).sum())

roi = safe_div(tot_prem, tot_spend)
cpl = safe_div(tot_spend, tot_leads)
cpa = safe_div(tot_spend, tot_contracts)

# Convenience aliases used by legacy render blocks
df = df_report
hist_tot_spend = float(df_hist['spend'].sum()) if len(df_hist) else 0.0
hist_tot_prem = float(df_hist.get('premium', pd.Series([0]*len(df_hist))).sum()) if len(df_hist) else 0.0
hist_roi = safe_div(hist_tot_prem, hist_tot_spend)


# ---------- Page Navigation ----------
PAGES = [
    "MONITOR · Executive Monitor (MTD)",
    "DIAGNOSE · Funnel Drivers (6-Factor)",
    "TRACK · Plan vs Actual (Why off?)",
    "DECIDE · Budget Actions (Optimizer)",
    "ENGINE VALUE · Impact vs Legacy",
    "GOVERN · Periods",
    "GOVERN · Data Quality",
    "ADVANCED · Model & Incrementality",
    "Metrics Dictionary",
]
page = st.sidebar.radio("페이지 이동", options=PAGES, index=0, key="page_nav")

# ---------- Tabs (Business Flow) ----------


# ===========================================
# ===========================================
if page == 'GOVERN · Periods':
    st.markdown('## Summary & Period Settings')
    st.markdown("<div class='smallcap'>운영 관점에서 기간 정의를 3개로 분리하고(Actual/History/Forecast), 모든 페이지에 동일하게 적용함</div>", unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    cA, cB, cC = st.columns(3)
    with cA:
        st.metric('리포트(Actuals View)', f"{report_start} ~ {report_end}")
    with cB:
        st.metric('최적화 과거기간(History)', f"{hist_start} ~ {hist_end}")
    with cC:
        st.metric('성과기간(Forecast/Impact)', f"{fc_start} ~ {fc_end}")

    st.markdown('### 이번 리포트 윈도우 KPI (선택 채널 기준)')
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric('Spend', format_won(tot_spend))
    with k2: st.metric('Premium', format_won(tot_prem))
    with k3: st.metric('ROI', f"{roi:,.2f}")
    with k4: st.metric('CPL', format_won(cpl))

    st.markdown('### Forecast/Impact 기간 월 개수')
    months = _month_range(fc_start, fc_end) if is_monthly else []
    if is_monthly:
        st.write(f"월 단위 데이터 기준: {len(months)}개월 ({', '.join(months[:12])}{'...' if len(months)>12 else ''})")
    else:
        st.write(f"일 단위 데이터 기준: {(pd.to_datetime(fc_end) - pd.to_datetime(fc_start)).days + 1}일")

    st.info('왼쪽 사이드바에서 기간을 바꾸면, 모든 페이지가 동일한 기간 정의로 갱신됨.')

if page == 'MONITOR · Executive Monitor (MTD)':
    # Executive Summary (Decide) - Consulting spec (Calendar Month)
    render_executive_summary(
        panel_ch,
        opt_simulation=st.session_state.get("opt_simulation"),
        default_target_premium=float(PREMIUM_TARGET or 0.0),
        analysis_start=report_start,
        analysis_end=report_end,
        selected_channels=selected_channels,
    )

# TAB 2: Drivers & Decomposition
# =========================================================
if page == 'DIAGNOSE · Funnel Drivers (6-Factor)':
    st.markdown("## Drivers & Decomposition")
    st.markdown("<div class='smallcap'>무엇이 성과를 움직였는가 (시간/채널 분해)</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # --- 6-factor funnel snapshot (report window) ---
    st.markdown("### Funnel Snapshot (Premium = Spend × 1/CPL × Attempt × Connect × Close × APS)")
    agg = df.groupby("channel", as_index=False).agg(
        spend=("spend", "sum"),
        leads=("leads", "sum"),
        attempts=("tm_attempts", "sum") if "tm_attempts" in df.columns else ("leads", "sum"),
        connected=("tm_connected", "sum") if "tm_connected" in df.columns else ("leads", "sum"),
        contracts=("contracts", "sum"),
        premium=("premium", "sum"),
    )
    agg["cpl"] = agg.apply(lambda r: safe_div(r["spend"], r["leads"]), axis=1)
    agg["attempt"] = agg.apply(lambda r: safe_div(r["attempts"], r["leads"]), axis=1)
    agg["connect"] = agg.apply(lambda r: safe_div(r["connected"], r["attempts"]), axis=1)
    agg["close"] = agg.apply(lambda r: safe_div(r["contracts"], r["connected"]), axis=1)
    agg["aps"] = agg.apply(lambda r: safe_div(r["premium"], r["contracts"]), axis=1)
    agg["roi"] = agg.apply(lambda r: safe_div(r["premium"], r["spend"]), axis=1)
    agg = agg.sort_values("premium", ascending=False)
    show = agg.copy()
    for c in ["spend", "premium", "cpl", "aps"]:
        show[c] = show[c].apply(format_won)
    for c in ["attempt", "connect", "close"]:
        show[c] = show[c].apply(lambda x: format_ratio(x, 2))
    show["roi"] = show["roi"].apply(lambda x: f"{x:,.2f}")
    st.dataframe(
        show[["channel", "spend", "premium", "roi", "cpl", "attempt", "connect", "close", "aps"]].rename(
            columns={
                "channel": "채널",
                "spend": "Spend",
                "premium": "Premium",
                "roi": "ROI",
                "cpl": "CPL",
                "attempt": "Attempts/Leads",
                "connect": "Connected/Attempts",
                "close": "Contracts/Connected",
                "aps": "APS",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=300,
    )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Daily totals
    daily = df.groupby("date", as_index=False).agg(
        spend=("spend","sum"),
        premium=("premium","sum"),
        leads=("leads","sum"),
        contracts=("contracts","sum"),
    ).sort_values("date")

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("<div class='card'><div class='smallcap'>일별 광고비</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.plot(daily["date"], daily["spend"])
        ax.set_xlabel("")
        ax.set_ylabel("Spend")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'><div class='smallcap'>일별 보험료(Proxy Revenue)</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.plot(daily["date"], daily["premium"])
        ax.set_xlabel("")
        ax.set_ylabel("Premium")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Channel mix share
    st.markdown("### Channel Mix (Spend Share)")
    mix = df.groupby("channel", as_index=False).agg(spend=("spend","sum"), premium=("premium","sum"))
    mix["spend_share"] = mix["spend"] / max(1.0, mix["spend"].sum())
    mix = mix.sort_values("spend", ascending=False)

    left, right = st.columns([1,1])
    with left:
        st.markdown("<div class='card'><div class='smallcap'>채널별 비중</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.bar(mix["channel"], mix["spend_share"])
        ax.set_ylabel("Spend Share")
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='smallcap'>채널별 요약</div>", unsafe_allow_html=True)
        tmp = mix.copy()
        tmp["광고비"] = tmp["spend"].apply(money)
        tmp["보험료"] = tmp["premium"].apply(money)
        tmp["광고비 비중"] = tmp["spend_share"].apply(lambda x: pct(x,0))
        st.dataframe(tmp[["channel","광고비","광고비 비중","보험료"]].rename(columns={"channel":"채널"}), use_container_width=True, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Counterfactual: channel OFF impact (if available)
    st.markdown("### Incrementality (Channel OFF Counterfactual)")
    if cf_ch is None:
        st.info("out/counterfactuals_channel.csv가 없어 채널 OFF 시나리오(증분효과) 분석을 표시할 수 없음. scripts/run_all.sh 실행 권장.")
    else:
        cf = cf_ch.copy()
        cf["date"] = pd.to_datetime(cf["date"]).dt.date
        cf = cf[(cf["date"] >= report_start) & (cf["date"] <= report_end)]

        # pandas groupby([]) raises: "No group keys passed".
        # Here we want the overall mean across the filtered window.
        # Spend column differs by pipeline mode (Bayesian vs fallback).
        spend_col = "spend_cf" if "spend_cf" in cf.columns else ("spend_mean" if "spend_mean" in cf.columns else ("spend" if "spend" in cf.columns else None))
        if spend_col is None:
            st.warning("counterfactuals_channel.csv에 spend 컬럼이 없어 예산 영향표 일부가 생략됨.")

        base_row = cf[cf["scenario"]=="BASE"].agg(
            premium=("premium_mean","mean"),
            leads=("leads_mean","mean"),
            spend=(spend_col,"mean") if spend_col else ("spend_cf","mean")
        )
        base_prem = float(base_row.get("premium", np.nan))
        base_leads = float(base_row.get("leads", np.nan))

        impacts = []
        for sc in sorted([s for s in cf["scenario"].unique() if s.startswith("OFF_")]):
            ch = sc.replace("OFF_","")
            m_row = cf[cf["scenario"]==sc].agg(
                premium=("premium_mean","mean"),
                leads=("leads_mean","mean"),
            )
            prem = float(m_row.get("premium", np.nan))
            leads = float(m_row.get("leads", np.nan))
            if (not np.isfinite(prem)) and (not np.isfinite(leads)):
                continue
            impacts.append({
                "channel": ch,
                "ΔPremium (BASE - OFF)": base_prem - prem,
                "ΔLeads (BASE - OFF)": base_leads - leads,
            })
        # If there are no OFF_ scenarios (or they don't contain usable metrics),
        # impacts can be empty -> DataFrame has no columns -> sort/bar will crash.
        imp = pd.DataFrame(
            impacts,
            columns=["channel", "ΔPremium (BASE - OFF)", "ΔLeads (BASE - OFF)"],
        )

        if imp.empty:
            st.warning(
                "OFF_* counterfactual 시나리오가 없거나(혹은 premium_mean/leads_mean 값이 비어 있음) 채널 증분효과(OFF 손실)를 계산할 수 없음.\n"
                "→ `out/counterfactuals_channel.csv`의 scenario 컬럼에 BASE와 OFF_*가 있는지 확인 필요."
            )
        else:
            imp = imp.sort_values("ΔPremium (BASE - OFF)", ascending=False)

            c1, c2 = st.columns([1,1])
            with c1:
                st.markdown("<div class='card'><div class='smallcap'>채널 OFF 시 보험료 손실(일평균)</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.bar(imp["channel"], imp["ΔPremium (BASE - OFF)"])
                ax.set_ylabel("Premium Loss / day")
                ax.tick_params(axis='x', rotation=30)
                st.pyplot(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='card'><div class='smallcap'>표 (일평균 손실)</div>", unsafe_allow_html=True)
                show = imp.copy()
                show["보험료 손실"] = show["ΔPremium (BASE - OFF)"].apply(money)
                show["리드 손실"] = show["ΔLeads (BASE - OFF)"].apply(lambda x: f"{x:,.1f}")
                st.dataframe(show[["channel","보험료 손실","리드 손실"]].rename(columns={"channel":"채널"}), use_container_width=True, height=300)
                st.markdown("<div class='note'>* 증분효과 해석: OFF 시나리오에서 감소한 만큼이 채널의 ‘기여(Incremental)’로 간주됨.</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### Contribution Waterfall (채널 기여도 워터폴)")
    st.markdown("<div class='note'>* 목적: ‘어떤 채널이 총 성과에 얼마나 기여했는지’를 한 장으로 설명. Counterfactual이 있으면 ΔPremium 기반, 없으면 Premium 분해(Proxy) 기반으로 표시.</div>", unsafe_allow_html=True)

    contrib = None
    mode = "proxy"
    if cf_ch is not None:
        try:
            cf = cf_ch.copy()
            cf["date"] = pd.to_datetime(cf["date"]).dt.date
            cf = cf[(cf["date"] >= report_start) & (cf["date"] <= report_end)]
            base_row = cf[cf["scenario"]=="BASE"].agg(premium=("premium_mean","mean"))
            base_prem = float(base_row.get("premium", np.nan))
            impacts = []
            for sc in sorted([s for s in cf["scenario"].unique() if s.startswith("OFF_")]):
                ch = sc.replace("OFF_","")
                m_row = cf[cf["scenario"]==sc].agg(premium=("premium_mean","mean"))
                prem = float(m_row.get("premium", np.nan))
                if not np.isfinite(prem):
                    continue
                impacts.append({"channel": ch, "contribution": base_prem - prem})
            contrib = pd.DataFrame(impacts).sort_values("contribution", ascending=False)
            mode = "incremental"
        except Exception:
            contrib = None

    if contrib is None or len(contrib)==0:
        tmp = df.groupby("channel", as_index=False).agg(premium=("premium","sum")).sort_values("premium", ascending=False)
        tmp = tmp.rename(columns={"premium":"contribution"})
        contrib = tmp

    top_n = 8
    c = contrib.copy()
    c["contribution"] = pd.to_numeric(c["contribution"], errors="coerce").fillna(0.0)
    c = c.sort_values("contribution", ascending=False)
    if len(c) > top_n:
        top = c.head(top_n)
        others = pd.DataFrame([{ "channel": "Others", "contribution": float(c.tail(len(c)-top_n)["contribution"].sum()) }])
        cplot = pd.concat([top, others], ignore_index=True)
    else:
        cplot = c

    # Build a simple waterfall: cumulative sum from 0
    vals = cplot["contribution"].tolist()
    labels = cplot["channel"].tolist()
    cum = [0.0]
    for v in vals:
        cum.append(cum[-1] + float(v))
    starts = cum[:-1]

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("<div class='card'><div class='smallcap'>워터폴</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.bar(labels, vals, bottom=starts)
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("ΔPremium" if mode=="incremental" else "Premium")
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='smallcap'>표</div>", unsafe_allow_html=True)
        t = cplot.copy()
        t["기여도"] = t["contribution"].apply(money)
        st.dataframe(t[["channel","기여도"]].rename(columns={"channel":"채널"}), use_container_width=True, height=320)
        st.markdown(
            "<div class='note'>"
            + ("* Counterfactual 기반: BASE 대비 OFF 손실(일평균) = 증분 기여도" if mode=="incremental" else "* Proxy 기반: 윈도우 내 Premium 합을 기여도로 표시 (정밀도 낮음)")
            + "</div></div>",
            unsafe_allow_html=True
        )

# =========================================================
# TAB 3: Model & Confidence
# =========================================================
if page == 'ADVANCED · Model & Incrementality':
    st.markdown("## Model & Confidence")
    st.markdown("<div class='smallcap'>추정 결과의 불확실성과 신뢰도를 함께 제공</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    if post_ch is None:
        st.info("out/posterior_summary_channel.csv가 없어 모델 파라미터/불확실성 표를 표시할 수 없음. scripts/run_all.sh 실행 권장.")
    else:
        st.markdown("### Posterior Summary (Channel Parameters)")
        p = post_ch.copy()

        # ------------------------------------------------------------------
        # Two possible schemas:
        # 1) Bayesian output: param/mean/sd/hdi... (from PyMC posterior_summary)
        # 2) Fallback output: channel/roi_mean/rr_mean/pps_mean/... (from run_all deterministic mode)
        # ------------------------------------------------------------------

        if "param" in p.columns:
            # focus: channel coefficients a_ch[*]
            p_ch = p[p["param"].astype(str).str.contains(r"a_ch\[")]
            if len(p_ch) == 0:
                p_ch = p

            def _extract_channel(s: str) -> str:
                # a_ch[google] -> google
                s = str(s)
                if "[" in s and "]" in s:
                    return s.split("[", 1)[1].split("]", 1)[0]
                return s

            p_ch["채널"] = p_ch["param"].apply(_extract_channel)
            p_ch = p_ch.sort_values("mean", ascending=False)

            cols = [c for c in ["채널","mean","sd","hdi_3%","hdi_97%","ess_bulk","r_hat"] if c in p_ch.columns]
            show = p_ch[cols].copy()
            show = show.rename(columns={
                "mean":"평균(Mean)",
                "sd":"표준편차(SD)",
                "hdi_3%":"HDI 3%",
                "hdi_97%":"HDI 97%",
                "ess_bulk":"ESS",
                "r_hat":"R-hat",
            })
        else:
            # Fallback schema: show empirical summaries instead of posterior params
            p_ch = p.copy()
            # normalize possible column names
            if "channel" in p_ch.columns and "채널" not in p_ch.columns:
                p_ch["채널"] = p_ch["channel"].astype(str)

            rename_map = {
                "roi_mean": "ROI(프리미엄/광고비)",
                "rr_mean": "전환율(계약/리드)",
                "pps_mean": "객단가(프리미엄/계약)",
                "spend": "광고비",
                "leads": "리드",
                "contracts": "계약",
                "premium": "프리미엄",
            }
            cols = ["채널"] + [c for c in rename_map.keys() if c in p_ch.columns]
            show = p_ch[cols].copy()
            show = show.rename(columns=rename_map)
        st.dataframe(show, use_container_width=True, height=360)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### How to Read (해석 가이드)")
        st.markdown(
            "- **신뢰구간(HDI)** 폭이 넓을수록 불확실성이 큼 → 데이터/실험/세분화 변수 보강 필요\n"
            "- **ESS**가 낮으면 샘플링이 불안정할 수 있음 → 학습 기간 확대/정규화/모델 단순화 고려\n"
            "- **R-hat**가 1.01 이상이면 수렴 이슈 가능성 → 샘플 수 증가/모델 재설계 권장\n"
            "- 다음 단계 의사결정은 ‘점추정’이 아니라 **구간(밴드)** 기반 시나리오로 수행 권장"
        )

# =========================================================
# TAB 4: Budget Optimizer
# =========================================================
# =========================================================
# TAB 4: Budget Optimizer (Act)
# =========================================================
if page == 'DECIDE · Budget Actions (Optimizer)':
    render_optimizer_engine(
        df_hist,
        tot_spend=float(hist_tot_spend),
        history_start=hist_start,
        history_end=hist_end,
        forecast_start=fc_start,
        forecast_end=fc_end,
        bud=bud,
        premium_target=float(PREMIUM_TARGET or 0.0),
    )


# =========================================================
# ENGINE VALUE: Engine vs Legacy ops baseline
# =========================================================
if page == 'ENGINE VALUE · Impact vs Legacy':
    st.markdown("## Engine Value · Impact vs Legacy")
    st.markdown(
        "<div class='smallcap'>엔진이 없던 운영 방식(과거 믹스 유지) 대비, 동일 예산에서 예상 성과가 얼마나 개선되는지 보여줌</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    runs = list_runs("out/runs")
    default_run = st.session_state.get("last_plan_run_id")
    if default_run and default_run in runs:
        run_id = default_run
    else:
        run_id = runs[0] if runs else ""

    if not run_id:
        st.info("저장된 플랜(run)이 없음. 먼저 'DECIDE · Budget Actions'에서 플랜 저장 필요.")
        st.stop()

    try:
        plan_df = load_plan_by_channel(run_id)
    except Exception as e:
        st.error(f"플랜 로드 실패: {e}")
        st.stop()

    # History rates (legacy baseline)
    h = df_hist.copy()
    if h.empty:
        st.warning("History 기간 데이터가 없어 Legacy 비교를 계산할 수 없음.")
        st.stop()

    h_by = h.groupby("channel", as_index=False).agg(
        spend=("spend", "sum"),
        leads=("leads", "sum"),
        attempts=("tm_attempts", "sum") if "tm_attempts" in h.columns else ("leads", "sum"),
        connected=("tm_connected", "sum") if "tm_connected" in h.columns else ("leads", "sum"),
        contracts=("contracts", "sum"),
        premium=("premium", "sum"),
    )
    h_by["cpl"] = h_by.apply(lambda r: safe_div(r["spend"], r["leads"]), axis=1)
    h_by["attempt_rate"] = h_by.apply(lambda r: safe_div(r["attempts"], r["leads"]), axis=1)
    h_by["connect_rate"] = h_by.apply(lambda r: safe_div(r["connected"], r["attempts"]), axis=1)
    h_by["close_rate"] = h_by.apply(lambda r: safe_div(r["contracts"], r["connected"]), axis=1)
    h_by["aps"] = h_by.apply(lambda r: safe_div(r["premium"], r["contracts"]), axis=1)

    total_budget = float(plan_df["recommended_spend"].sum())
    h_total_spend = float(h_by["spend"].sum())
    if h_total_spend <= 0:
        st.warning("History spend 합이 0이라 Legacy 믹스 계산 불가.")
        st.stop()

    h_by["legacy_share"] = h_by["spend"] / h_total_spend
    h_by["legacy_spend"] = h_by["legacy_share"] * total_budget

    # Legacy prediction using historical funnel rates
    def _legacy_predict(row):
        spend = float(row["legacy_spend"])
        cpl = float(row["cpl"])
        leads = spend / cpl if cpl > 0 else 0.0
        attempts = leads * float(row["attempt_rate"])
        connected = attempts * float(row["connect_rate"])
        contracts = connected * float(row["close_rate"])
        premium = contracts * float(row["aps"])
        return pd.Series({"legacy_leads": leads, "legacy_contracts": contracts, "legacy_premium": premium})

    h_pred = h_by.join(h_by.apply(_legacy_predict, axis=1))

    # Engine plan (recommended) prediction already provided
    engine = plan_df.groupby("channel", as_index=False).agg(
        engine_spend=("recommended_spend", "sum"),
        engine_leads=("pred_leads", "sum"),
        engine_contracts=("pred_contracts", "sum"),
        engine_premium=("pred_premium", "sum"),
    )

    cmp = pd.merge(engine, h_pred[["channel", "legacy_spend", "legacy_leads", "legacy_contracts", "legacy_premium"]], on="channel", how="outer").fillna(0.0)
    cmp["uplift_premium"] = cmp["engine_premium"] - cmp["legacy_premium"]
    cmp["roi_engine"] = cmp.apply(lambda r: safe_div(r["engine_premium"], r["engine_spend"]), axis=1)
    cmp["roi_legacy"] = cmp.apply(lambda r: safe_div(r["legacy_premium"], r["legacy_spend"]), axis=1)

    tot_engine_p = float(cmp["engine_premium"].sum())
    tot_legacy_p = float(cmp["legacy_premium"].sum())
    tot_engine_s = float(cmp["engine_spend"].sum())
    tot_legacy_s = float(cmp["legacy_spend"].sum())
    uplift = tot_engine_p - tot_legacy_p

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Budget (same)", format_won(tot_engine_s))
    a2.metric("Engine Pred Premium", format_won(tot_engine_p))
    a3.metric("Legacy Pred Premium", format_won(tot_legacy_p))
    a4.metric("Uplift vs Legacy", format_won(uplift))

    st.markdown("### Channel view")
    show = cmp.copy()
    show = show.sort_values("uplift_premium", ascending=False)
    show["engine_spend"] = show["engine_spend"].apply(format_won)
    show["legacy_spend"] = show["legacy_spend"].apply(format_won)
    show["engine_premium"] = show["engine_premium"].apply(format_won)
    show["legacy_premium"] = show["legacy_premium"].apply(format_won)
    show["uplift_premium"] = show["uplift_premium"].apply(format_won)
    show["roi_engine"] = show["roi_engine"].apply(lambda x: f"{x:,.2f}")
    show["roi_legacy"] = show["roi_legacy"].apply(lambda x: f"{x:,.2f}")
    st.dataframe(
        show[["channel", "legacy_spend", "engine_spend", "legacy_premium", "engine_premium", "uplift_premium", "roi_legacy", "roi_engine"]].rename(
            columns={
                "channel": "채널",
                "legacy_spend": "Legacy 예산",
                "engine_spend": "Engine 예산",
                "legacy_premium": "Legacy 예측 보험료",
                "engine_premium": "Engine 예측 보험료",
                "uplift_premium": "Uplift",
                "roi_legacy": "Legacy ROI",
                "roi_engine": "Engine ROI",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### Actual (Impact period) availability")
    mask_imp = (panel_ch["date"] >= fc_start) & (panel_ch["date"] <= fc_end) & (panel_ch["channel"].isin(selected_channels))
    df_imp = panel_ch.loc[mask_imp].copy()
    if df_imp.empty:
        st.info("Impact 기간 Actual 데이터가 아직 없음(미래 기간이거나 미집계). Plan vs Actual은 TRACK 페이지에서 확인.")
    else:
        prem_imp = float(df_imp.get("premium", pd.Series([0]*len(df_imp))).sum())
        spend_imp = float(df_imp.get("spend", pd.Series([0]*len(df_imp))).sum())
        st.metric("Impact Actual Premium", format_won(prem_imp))
        st.metric("Impact Actual ROI", f"{safe_div(prem_imp, spend_imp):,.2f}")

if page == 'GOVERN · Data Quality':
    st.markdown("## Monitoring & Data Quality")
    st.markdown("<div class='smallcap'>지표 신뢰성 · 경보 · 계측(Measurement) 개선</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    left, right = st.columns([1,1])
    with left:
        st.markdown("<div class='card'><div class='smallcap'>Data Quality Gate</div><div style='margin-top:10px'></div>", unsafe_allow_html=True)
        if dq is None:
            st.info("out/data_quality_report.json가 없어 데이터 품질 결과를 표시할 수 없음.")
        else:
            summ = dq.get("summary", {})
            p = int(summ.get("PASS", 0))
            w = int(summ.get("WARN", 0))
            f = int(summ.get("FAIL", 0))

            c1,c2,c3 = st.columns(3)
            with c1:
                badge(f"PASS {p}", "ok")
            with c2:
                badge(f"WARN {w}", "mid" if w>0 else "ok")
            with c3:
                badge(f"FAIL {f}", "warn" if f>0 else "ok")

            fails = [r for r in dq.get("results", []) if r.get("level") in ("FAIL","WARN")]
            if fails:
                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                st.markdown("**Attention Needed**")
                df_fail = pd.DataFrame(fails)
                st.dataframe(df_fail, use_container_width=True, height=220)
            else:
                st.markdown("<div class='note'>현재 윈도우 기준 품질 경보 없음.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><div class='smallcap'>Metric Lineage (계산 로직/입력 근거)</div><div style='margin-top:10px'></div>", unsafe_allow_html=True)
        if lineage is None:
            st.info("out/metric_lineage.csv가 없어 지표 계보(입력→산식→산출)를 표시할 수 없음.")
        else:
            st.dataframe(lineage, use_container_width=True, height=320)
            st.markdown("<div class='note'>* 보고 지표는 lineage로 ‘어떤 입력파일/함수/산식’에서 생성됐는지 추적 가능해야 함.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 운영 팁 (Ops)")
    st.markdown(
        "- 매일: 데이터 품질 Gate 통과 여부 확인 → FAIL 발생 시 원천 수집/조인 로직 우선 수정\n"
        "- 매주: 증분효과(OFF 시나리오)와 권장 예산안 비교 → ‘차이가 커진 채널’을 우선 점검\n"
        "- 매월: 모델 불확실성(HDI/ESS/R-hat) 점검 → 세분화 변수/실험 설계로 개선"
    )

# ===========================================
if page == 'Metrics Dictionary':
    render_metrics_dictionary()

# TAB 7: Plan vs Actual (Tracking)
# =========================================================
if page == 'TRACK · Plan vs Actual (Why off?)':
    st.markdown("## Plan vs Actual Tracking")
    st.markdown("<div class='smallcap'>최적화 플랜(권장 예산/예측)과 실제 성과(Actuals)를 월 단위로 정합성 검증하며 비교</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='note'>리포트(Actuals View): {report_start}~{report_end} · 성과기간(Impact): {fc_start}~{fc_end} · 최적화 과거기간: {hist_start}~{hist_end}</div>",
        unsafe_allow_html=True,
    )


    # -------- Plan selection --------
    runs = list_runs("out/runs")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        default_run = st.session_state.get("last_plan_run_id")
        if default_run and default_run in runs:
            run_id = st.selectbox("플랜 Run 선택", options=runs, index=runs.index(default_run))
        else:
            run_id = st.selectbox("플랜 Run 선택", options=runs) if runs else ""

    with c2:
        mode = st.selectbox("비교 범위", options=["성과기간(Impact) 전체", "단일월"], index=0, key="track_mode")

    with c3:
        st.markdown(
            "<div class='note'>플랜 저장(4번 페이지) 후 out/runs 아래에 run 폴더가 생성됨. Actuals는 아래 업로드.</div>",
            unsafe_allow_html=True,
        )

    # Tracking window
    if mode == "단일월":
        default_cmp_month = _month_str(report_end)
        month = st.text_input("비교 월(YYYY-MM)", value=default_cmp_month, key="track_month")
        month = (month or "").strip()[:7]
        # single month window
        try:
            y, m = int(month[:4]), int(month[5:7])
            track_start = datetime(y, m, 1).date()
            # end = last day of month
            if m == 12:
                track_end = datetime(y+1, 1, 1).date() - timedelta(days=1)
            else:
                track_end = datetime(y, m+1, 1).date() - timedelta(days=1)
        except Exception:
            st.error("비교 월 형식이 올바르지 않음. 예: 2026-03")
            st.stop()
    else:
        track_start = fc_start
        track_end = fc_end

    n_months = months_inclusive(track_start, track_end)
    st.markdown(
        f"<div class='note'>Tracking 기간: <b>{track_start} ~ {track_end}</b> (월수: {n_months}) — Plan은 월단위 플랜을 월수만큼 스케일링하여 동일 기간으로 비교함</div>",
        unsafe_allow_html=True,
    )
    if not runs:
        st.info("저장된 플랜 Run 이 없음. 4) Budget Optimizer에서 '플랜 저장'을 먼저 실행.")
    else:
        try:
            plan_df = load_plan_by_channel(run_id)
        except Exception as e:
            st.error(f"플랜 로딩 실패: {e}")
            plan_df = None

        # -------- Actuals upload --------
        st.markdown("### Actuals 업로드")
        st.markdown(
            "<div class='note'>필수 컬럼: month(YYYY-MM) 또는 date(YYYY-MM-DD), channel, spend, leads, contracts, premium</div>",
            unsafe_allow_html=True,
        )
        up = st.file_uploader("actuals.csv 업로드", type=["csv"], key="actuals_uploader")

        # fallback: sample data bundled
        sample_path = os.path.join("data", "actuals_sample.csv")
        use_sample = st.checkbox("샘플 actuals 사용", value=(up is None and os.path.exists(sample_path)))

        actuals_df = None
        if up is not None:
            try:
                actuals_df = pd.read_csv(up)
            except Exception as e:
                st.error(f"업로드 파일 읽기 실패: {e}")
        elif use_sample and os.path.exists(sample_path):
            actuals_df = pd.read_csv(sample_path)

        if actuals_df is None:
            st.info("Actuals CSV를 업로드하거나 샘플을 선택.")
        else:
            try:
                actuals_df = validate_actuals_df(actuals_df)
            except Exception as e:
                st.error(f"Actuals 스키마 오류: {e}")
                actuals_df = None

        if plan_df is not None and actuals_df is not None:
            try:
                cmp_df, totals = compare_plan_vs_actual_period(plan_df, actuals_df, start=track_start, end=track_end, scale_by_forecast_months=True)
                # KPI cards
                k1, k2, k3, k4 = st.columns(4)
                k1.markdown(f"<div class='card'><div class='smallcap'>Plan Spend</div><div class='kpi'>{format_won(totals['recommended_spend'])}</div></div>", unsafe_allow_html=True)
                k2.markdown(f"<div class='card'><div class='smallcap'>Actual Spend</div><div class='kpi'>{format_won(totals['actual_spend'])}</div></div>", unsafe_allow_html=True)
                k3.markdown(f"<div class='card'><div class='smallcap'>Plan Premium</div><div class='kpi'>{format_won(totals['pred_premium'])}</div></div>", unsafe_allow_html=True)
                k4.markdown(f"<div class='card'><div class='smallcap'>Actual Premium</div><div class='kpi'>{format_won(totals['actual_premium'])}</div></div>", unsafe_allow_html=True)

                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                st.markdown("### Channel-level Variance")
                show = cmp_df.copy()
                for col in [
                    "recommended_spend",
                    "actual_spend",
                    "var_spend",
                    "pred_leads",
                    "actual_leads",
                    "var_leads",
                    "pred_contracts",
                    "actual_contracts",
                    "var_contracts",
                    "pred_premium",
                    "actual_premium",
                    "var_premium",
                ]:
                    if col in show.columns:
                        show[col] = show[col].apply(money)

                st.dataframe(
                    show[[
                        "period",
                        "channel",
                        "recommended_spend",
                        "actual_spend",
                        "var_spend",
                        "pred_premium",
                        "actual_premium",
                        "var_premium",
                        "roi_plan",
                        "roi_actual",
                    ]].rename(
                        columns={
                            "period": "기간",
                            "channel": "채널",
                            "recommended_spend": "Plan 예산",
                            "actual_spend": "Actual 예산",
                            "var_spend": "예산 차이",
                            "pred_premium": "Plan 보험료",
                            "actual_premium": "Actual 보험료",
                            "var_premium": "보험료 차이",
                            "roi_plan": "Plan ROI",
                            "roi_actual": "Actual ROI",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=380,
                )

                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                st.markdown("### Why did it miss? (6-Factor driver decomposition)")
                st.markdown(
                    "<div class='smallcap'>Premium 차이를 6개 요인(Spend, CPL, Attempt, Connect, Close, APS)로 Shapley 분해하고, 채널별 Top 원인을 랭킹</div>",
                    unsafe_allow_html=True,
                )

                exp = explain_gap_table(cmp_df)
                if exp is None or exp.empty:
                    st.info("분해를 계산할 데이터가 충분하지 않음.")
                else:
                    exp_show = exp.copy()
                    for c in [
                        "premium_gap",
                        "contrib_spend",
                        "contrib_cpl",
                        "contrib_attempt",
                        "contrib_connect",
                        "contrib_close",
                        "contrib_aps",
                    ]:
                        if c in exp_show.columns:
                            exp_show[c] = exp_show[c].apply(money)

                    for c in [
                        "roi_plan",
                        "roi_actual",
                        "attempt_plan",
                        "attempt_actual",
                        "connect_plan",
                        "connect_actual",
                        "close_plan",
                        "close_actual",
                    ]:
                        if c in exp_show.columns:
                            exp_show[c] = exp_show[c].apply(lambda x: format_ratio(x, 2) if x is not None else "-")

                    for c in ["cpl_plan", "cpl_actual", "aps_plan", "aps_actual"]:
                        if c in exp_show.columns:
                            exp_show[c] = exp_show[c].apply(money)

                    st.dataframe(
                        exp_show[
                            [
                                "channel",
                                "premium_gap",
                                "top_reason",
                                "contrib_spend",
                                "contrib_cpl",
                                "contrib_attempt",
                                "contrib_connect",
                                "contrib_close",
                                "contrib_aps",
                                "roi_plan",
                                "roi_actual",
                            ]
                        ].rename(
                            columns={
                                "channel": "채널",
                                "premium_gap": "보험료 오차(Actual-Plan)",
                                "top_reason": "Top 원인",
                                "contrib_spend": "기여: Spend",
                                "contrib_cpl": "기여: CPL",
                                "contrib_attempt": "기여: Attempt",
                                "contrib_connect": "기여: Connect",
                                "contrib_close": "기여: Close",
                                "contrib_aps": "기여: APS",
                                "roi_plan": "Plan ROI",
                                "roi_actual": "Actual ROI",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                        height=360,
                    )

                    agg = exp[["contrib_spend", "contrib_cpl", "contrib_attempt", "contrib_connect", "contrib_close", "contrib_aps"]].sum(numeric_only=True)
                    rank = sorted([(k, float(v)) for k, v in agg.items()], key=lambda kv: abs(kv[1]), reverse=True)
                    if rank:
                        pretty = " · ".join([f"{name.replace('contrib_', '').upper()}: {money(val)}" for name, val in rank])
                        st.markdown(f"<div class='note'>전체 Top 원인(절대값 기준): {pretty}</div>", unsafe_allow_html=True)


                # Simple plots: premium plan vs actual by channel
                st.markdown("### Visual")
                dplot = cmp_df.sort_values("actual_premium", ascending=False).head(12)
                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(dplot["channel"], dplot["pred_premium"], marker="o", label="Plan")
                ax.plot(dplot["channel"], dplot["actual_premium"], marker="o", label="Actual")
                ax.set_xlabel("Channel")
                ax.set_ylabel("Premium")
                ax.tick_params(axis='x', rotation=35)
                ax.legend()
                st.pyplot(fig, use_container_width=True)

                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                st.markdown("### Data Quality Checks")
                # Quick integrity checks to prevent common mistakes
                issues = []
                if abs(float(cmp_df["recommended_spend"].sum()) - float(totals["recommended_spend"])) > 1e-6:
                    issues.append("Plan spend total mismatch.")
                if abs(float(cmp_df["actual_spend"].sum()) - float(totals["actual_spend"])) > 1e-6:
                    issues.append("Actual spend total mismatch.")
                if (cmp_df[["recommended_spend", "actual_spend", "pred_premium", "actual_premium"]] < 0).any().any():
                    issues.append("Negative values detected.")
                if issues:
                    st.error("; ".join(issues))
                else:
                    st.success("정합성 체크 통과 (기본).")

            except Exception as e:
                st.error(f"비교 계산 실패: {e}")