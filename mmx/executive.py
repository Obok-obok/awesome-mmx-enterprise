
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .utils import add_total_row_breakdown, masked_number_input, money, month_range_calendar, parse_number, pct, safe_div
from .formatting import format_won


def _fit_log_curve(one: pd.DataFrame, ycol: str):
    """Simple saturation curve y = a * log(1 + spend/gamma) with fallback linear."""
    one = one[["spend", ycol]].copy()
    one["spend"] = pd.to_numeric(one["spend"], errors="coerce").fillna(0.0)
    one[ycol] = pd.to_numeric(one[ycol], errors="coerce").fillna(0.0)

    s = one["spend"].values
    y = one[ycol].values

    if (np.nanmax(s) <= 0) or (np.nanmax(y) <= 0) or (len(one) < 8):
        eff = float(np.sum(y) / max(1.0, np.sum(s)))
        def pred(x):
            return eff * np.asarray(x, dtype=float)
        return {"kind": "linear", "eff": eff, "pred": pred}

    q = np.quantile(s[s > 0], [0.2, 0.4, 0.6, 0.8]) if np.any(s > 0) else np.array([1, 2, 3, 4], dtype=float)
    gammas = np.unique(np.clip(q, 1.0, None))
    if len(gammas) < 3:
        gammas = np.unique(np.clip(np.array([
            np.mean(s[s > 0]) if np.any(s > 0) else 1.0,
            np.max(s),
            np.median(s[s > 0]) if np.any(s > 0) else 1.0,
        ]), 1.0, None))

    best = None
    for g in gammas:
        x = np.log1p(s / float(g))
        denom = float(np.dot(x, x)) + 1e-9
        a = float(np.dot(x, y) / denom)
        predy = a * x
        mse = float(np.mean((predy - y) ** 2))
        if (best is None) or (mse < best["mse"]):
            best = {"a": a, "g": float(g), "mse": mse}

    a = best["a"]; g = best["g"]
    def pred(x):
        x = np.asarray(x, dtype=float)
        return a * np.log1p(x / g)

    return {"kind": "log", "a": a, "g": g, "pred": pred}


def _greedy_allocate_to_close_gap(
    channels: list[str],
    cur_spend: dict[str, float],
    rr_map: dict[str, float],
    pps_map: dict[str, float],
    lead_models: dict[str, dict],
    gap_premium: float,
    *,
    step_budget: float,
    max_extra_budget: float,
) -> tuple[pd.DataFrame, dict]:
    """Allocate extra budget across channels to close premium gap using marginal gains."""

    def pred_leads(ch: str, spend: float) -> float:
        m = lead_models.get(ch)
        if m is None:
            return 0.0
        return float(m["pred"]([spend])[0])

    def pred_premium(ch: str, spend: float) -> float:
        leads = pred_leads(ch, spend)
        contracts = leads * float(rr_map.get(ch, 0.0))
        return contracts * float(pps_map.get(ch, 0.0))

    alloc = {ch: 0.0 for ch in channels}
    cur = {ch: float(cur_spend.get(ch, 0.0)) for ch in channels}
    gained = 0.0
    spent = 0.0

    # safety cap to avoid infinite loops
    max_iter = int(max_extra_budget / max(step_budget, 1.0)) + 5
    for _ in range(max_iter):
        if spent + step_budget > max_extra_budget + 1e-9:
            break
        if gained >= gap_premium - 1e-9:
            break

        best_ch = None
        best_gain = -1e18
        for ch in channels:
            s0 = cur[ch]
            s1 = s0 + step_budget
            g = pred_premium(ch, s1) - pred_premium(ch, s0)
            if g > best_gain:
                best_gain = g
                best_ch = ch

        if best_ch is None or best_gain <= 0:
            break

        alloc[best_ch] += step_budget
        cur[best_ch] += step_budget
        spent += step_budget
        gained += best_gain

    rows = []
    for ch in channels:
        s0 = float(cur_spend.get(ch, 0.0))
        m0 = pred_premium(ch, s0)
        m1 = pred_premium(ch, s0 + alloc[ch])
        rows.append({
            "channel": ch,
            "current_spend": s0,
            "extra_spend": alloc[ch],
            "expected_premium_gain": max(0.0, m1 - m0),
            "marginal_premium_per_step": max(0.0, pred_premium(ch, s0 + step_budget) - pred_premium(ch, s0)),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("expected_premium_gain", ascending=False)
    summary = {"extra_budget": spent, "expected_gain": gained, "coverage": safe_div(gained, gap_premium)}
    return out, summary



def _inject_exec_css() -> None:
    st.markdown(
        """
<style>
/* Executive tiles */
.exec-grid {margin-top: 0.25rem;}
.exec-note {color:#6b7280; font-size:12px;}
.exec-card {
  padding:18px 16px;
  border-radius:16px;
  background:#F6F7FB;
  border:1px solid rgba(16,24,40,0.06);
  height: 118px;
  display:flex;
  flex-direction:column;
  justify-content:center;
  gap:4px;
}
.exec-kpi {
  font-weight: 750;
  font-size: clamp(16px, 1.6vw, 22px);
  line-height: 1.1;
  letter-spacing: -0.02em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.exec-sub {
  font-size: 12px;
  color: rgba(17,24,39,0.62);
}
.exec-title {
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.exec-section {
  margin-top: 10px;
}
.hr {
  border-top: 1px solid rgba(17,24,39,0.08);
  margin: 14px 0 14px 0;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_executive_summary(
    df_full: pd.DataFrame,
    *,
    opt_simulation: dict | None = None,
    default_target_premium: float = 23_000_000.0,
    analysis_start: "datetime | None" = None,
    analysis_end: "datetime | None" = None,
    selected_channels: list[str] | None = None,
) -> None:
    """
    Executive Summary (Decide) - Consulting spec.
    - Calendar month 기준 (매월 1일 ~ 월말)
    - Target vs MTD
    - Funnel (Budget -> Leads -> Sales -> RR -> Premium/Sale -> Total Premium)
    - Media breakdown + 합계
    - (있으면) Optimizer 시뮬레이션 효과를 숫자로 비교표 출력
    """
    _inject_exec_css()

    if df_full is None or df_full.empty:
        st.warning("패널 데이터가 없어 Executive Summary를 계산할 수 없음. out/panel_daily_channel.csv 확인 필요.")
        return

    if "date" not in df_full.columns:
        st.warning("date 컬럼이 없어 월(MTD) 기준을 계산할 수 없음. out/panel_daily_channel.csv 스키마 확인 필요.")
        return

    df = df_full.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Channel filter (match global selector)
    if selected_channels:
        df = df[df.get("channel").astype(str).isin([str(x) for x in selected_channels])]

    # Period mode: Calendar Month (default) vs Selected Range (global filters)
    cal_start, cal_end = month_range_calendar()
    sel_start = pd.to_datetime(analysis_start) if analysis_start is not None else None
    sel_end = pd.to_datetime(analysis_end) if analysis_end is not None else None

    mode_opts = ["Calendar Month", "Selected Range"]
    if sel_start is None or sel_end is None:
        mode_opts = ["Calendar Month"]
    mode = st.selectbox("Executive 기준 기간", options=mode_opts, index=0)

    if mode == "Selected Range" and sel_start is not None and sel_end is not None:
        start, end = sel_start, sel_end
        period_label = f"{start.date().isoformat()} ~ {end.date().isoformat()} (Selected Range)"
    else:
        start, end = cal_start, cal_end
        period_label = f"{start.date().isoformat()} ~ {end.date().isoformat()} (Calendar Month)"

    df_m = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    # --- base metrics (MTD) : compute BEFORE any inputs that depend on them ---
    mtd_premium = float(pd.to_numeric(df_m.get("premium", 0.0), errors="coerce").fillna(0.0).sum())
    mtd_spend = float(pd.to_numeric(df_m.get("spend", 0.0), errors="coerce").fillna(0.0).sum())
    total_leads = float(pd.to_numeric(df_m.get("leads", 0.0), errors="coerce").fillna(0.0).sum())
    total_sales = float(pd.to_numeric(df_m.get("contracts", 0.0), errors="coerce").fillna(0.0).sum())
    total_attempts = float(pd.to_numeric(df_m.get("tm_attempts", 0.0), errors="coerce").fillna(0.0).sum())
    total_connected = float(pd.to_numeric(df_m.get("tm_connected", 0.0), errors="coerce").fillna(0.0).sum())
    mtd_budget_used = mtd_spend

    st.markdown(f"<div class='exec-title'>Target vs MTD (이번달 목표 대비 현황)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='exec-note'>조회 기준: {period_label}</div>", unsafe_allow_html=True)

    _c1, _c2 = st.columns([1.0, 1.0])
    with _c1:
        _mt = masked_number_input(
            "이번달 목표 보험료",
            key="monthly_target_premium",
            default=float(st.session_state.get("monthly_target_premium", default_target_premium)),
        )
    with _c2:
        # Budget is a shared state managed by the Optimizer tab (single source of truth).
        # We only display it here to avoid widget-state drift across tabs.
        _shared_budget = float(st.session_state.get("monthly_total_budget", mtd_spend))
        _shared_budget = float(max(0.0, _shared_budget if np.isfinite(_shared_budget) else float(mtd_spend)))
        st.markdown(
            f"<div class='exec-card'><div class='exec-kpi'>{format_won(_shared_budget)}</div><div class='exec-sub'>이번달 총 예산(Optimizer 입력값)</div></div>",
            unsafe_allow_html=True,
        )
        st.caption("예산 변경은 Budget Optimizer 탭에서 입력하면 Executive Summary에 자동 반영됨")

    monthly_target = float(max(0.0, _mt if np.isfinite(_mt) else float(default_target_premium)))
    monthly_budget = _shared_budget
    total_budget = monthly_budget
    st.caption(f"입력값(해석): 목표 {format_won(monthly_target)} · 총 예산 {format_won(monthly_budget)}")
    st.caption(f"입력값(해석): 목표 {format_won(monthly_target)} · 총 예산 {format_won(monthly_budget)}")
    rr = safe_div(total_sales, total_leads)
    premium_per_sale = safe_div(mtd_premium, total_sales)

    achievement = safe_div(mtd_premium, monthly_target)
    gap = float(monthly_target - mtd_premium)

    # remaining days within calendar month (as-of date -> month end)
    # - 기본: 분석기간 종료일(Selected Range) 기준으로 월말까지 남은 일수 계산
    # - 분석기간 종료일이 미래이면 오늘 기준으로 계산
    today = pd.Timestamp.today(tz=None).normalize()
    as_of = (pd.Timestamp(analysis_end).normalize() if analysis_end is not None else today)
    if as_of > today:
        as_of = today
    remaining_days = int(max(0, (end.normalize() - as_of).days))

    daily_required = safe_div(gap, remaining_days) if remaining_days > 0 else np.nan
    daily_required_disp = format_won(daily_required) if remaining_days > 0 else "-"

    # --- KPI tiles ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"<div class='exec-card'><div class='exec-kpi'>{format_won(monthly_target)}</div><div class='exec-sub'>이번달 목표 보험료</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='exec-card'><div class='exec-kpi'>{format_won(mtd_premium)}</div><div class='exec-sub'>현재 MTD 보험료</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='exec-card'><div class='exec-kpi'>{pct(achievement)}</div><div class='exec-sub'>목표 달성률</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='exec-card'><div class='exec-kpi'>{format_won(gap)}</div><div class='exec-sub'>잔여 필요 보험료</div></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='exec-card'><div class='exec-kpi'>{daily_required_disp}</div><div class='exec-sub'>일평균 필요 보험료</div></div>", unsafe_allow_html=True)
    if remaining_days <= 0:
        c5.caption("분석 기준일이 월말(또는 이후)이라 일평균 필요 보험료 계산 불가")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # --- Funnel tiles ---
    st.markdown("<div class='exec-section'><b>Funnel</b> (예산 → Leads → Sales → RR → 건당 보험료 → 총보험료)</div>", unsafe_allow_html=True)
    f1, f2, f3, f4, f5, f6 = st.columns(6)
    f1.markdown(f"<div class='exec-card'><div class='exec-kpi'>{format_won(total_budget)}</div><div class='exec-sub'>예산</div></div>", unsafe_allow_html=True)
    f2.markdown(f"<div class='exec-card'><div class='exec-kpi'>{money(total_leads)}</div><div class='exec-sub'>Lead</div></div>", unsafe_allow_html=True)
    f3.markdown(f"<div class='exec-card'><div class='exec-kpi'>{money(total_sales)}</div><div class='exec-sub'>Sales(Contracts)</div></div>", unsafe_allow_html=True)
    f4.markdown(f"<div class='exec-card'><div class='exec-kpi'>{pct(rr)}</div><div class='exec-sub'>전환율(RR)</div></div>", unsafe_allow_html=True)
    f5.markdown(f"<div class='exec-card'><div class='exec-kpi'>{format_won(premium_per_sale)}</div><div class='exec-sub'>건당 보험료</div></div>", unsafe_allow_html=True)
    f6.markdown(f"<div class='exec-card'><div class='exec-kpi'>{format_won(mtd_premium)}</div><div class='exec-sub'>총보험료(MTD)</div></div>", unsafe_allow_html=True)

    # --- Efficiency metrics & SEM-style factorization ---
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("<div class='exec-section'><b>효율 지표</b> (Efficiency)</div>", unsafe_allow_html=True)

    cpl = safe_div(mtd_budget_used, total_leads)          # 예산 -> 리드
    cvr = safe_div(total_sales, total_leads)           # 리드 -> 계약
    apc = safe_div(mtd_premium, total_sales)           # 계약 -> 보험료 (Avg premium per contract)
    cpa = safe_div(mtd_budget_used, total_sales)          # 예산 -> 계약
    roi = safe_div(mtd_premium, mtd_budget_used)          # 보험료/예산

    e1, e2, e3, e4, e5 = st.columns(5)
    e1.markdown(f"<div class='exec-card'><div class='exec-kpi'>{format_won(cpl)}</div><div class='exec-sub'>CPL (원/Lead)</div></div>", unsafe_allow_html=True)
    e2.markdown(f"<div class='exec-card'><div class='exec-kpi'>{pct(cvr)}</div><div class='exec-sub'>Lead→Contract 전환율</div></div>", unsafe_allow_html=True)
    e3.markdown(f"<div class='exec-card'><div class='exec-kpi'>{format_won(apc)}</div><div class='exec-sub'>계약당 보험료</div></div>", unsafe_allow_html=True)
    e4.markdown(f"<div class='exec-card'><div class='exec-kpi'>{format_won(cpa)}</div><div class='exec-sub'>CPA (원/Contract)</div></div>", unsafe_allow_html=True)
    e5.markdown(f"<div class='exec-card'><div class='exec-kpi'>{roi:.2f}</div><div class='exec-sub'>ROI (Premium/Spend)</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='exec-section'><b>구조방정식(SEM) 관점 Factor</b> (ROI = Premium/Spend 5-Factor 분해)</div>", unsafe_allow_html=True)

    # ROI factorization:
    # Premium/Spend = (Leads/Spend) * (Attempts/Leads) * (Connected/Attempts) * (Contracts/Connected) * (Premium/Contract)
    leads_per_spend = safe_div(total_leads, mtd_budget_used)
    attempts_per_lead = safe_div(total_attempts, total_leads)
    connected_per_attempt = safe_div(total_connected, total_attempts)
    contracts_per_connected = safe_div(total_sales, total_connected)
    premium_per_contract = safe_div(mtd_premium, total_sales)

    roi_implied = leads_per_spend * attempts_per_lead * connected_per_attempt * contracts_per_connected * premium_per_contract
    roi_gap = roi_implied - roi if np.isfinite(roi_implied) and np.isfinite(roi) else np.nan

    sem_df = pd.DataFrame([
        {"Factor": "Leads / Spend", "값": f"{leads_per_spend:.6f}" if np.isfinite(leads_per_spend) else "-", "해석": "지출 1원당 리드 획득량 (상단 퍼널 효율)"},
        {"Factor": "Attempts / Leads", "값": f"{attempts_per_lead:.3f}" if np.isfinite(attempts_per_lead) else "-", "해석": "리드 1건당 통화 시도 횟수 (처리 강도/운영 레버)"},
        {"Factor": "Connected / Attempts", "값": pct(connected_per_attempt) if np.isfinite(connected_per_attempt) else "-", "해석": "통화 시도 대비 연결율 (콜센터 포화/리스트 품질)"},
        {"Factor": "Contracts / Connected", "값": pct(contracts_per_connected) if np.isfinite(contracts_per_connected) else "-", "해석": "연결된 리드의 계약 전환율 (상담 품질/상품 경쟁력)"},
        {"Factor": "Premium / Contract", "값": format_won(premium_per_contract) if np.isfinite(premium_per_contract) else "-", "해석": "계약 1건당 평균 보험료 (APC)"},
        {"Factor": "ROI (Premium/Spend)", "값": f"{roi:.4f}" if np.isfinite(roi) else "-", "해석": "최종 목표지표"},
        {"Factor": "5-Factor 곱(재구성 ROI)", "값": f"{roi_implied:.4f}" if np.isfinite(roi_implied) else "-", "해석": "5개 factor의 곱으로 복원된 ROI"},
        {"Factor": "차이(재구성-실제)", "값": f"{roi_gap:+.6f}" if np.isfinite(roi_gap) else "-", "해석": "0에 가까울수록 집계/정의가 일치"},
    ])
    st.dataframe(sem_df, use_container_width=True, hide_index=True)
    st.caption("검증: 5-Factor 곱(재구성 ROI)이 실제 ROI(Premium/Spend)와 일치해야 함. 차이가 크면 집계 단위/분모 0 처리/기간 필터를 점검.")

    # --- Channel SEM decomposition (bottleneck finder) ---
    st.markdown("<div class='exec-section'><b>채널별 SEM 분해</b> (ROI 5-Factor) · 병목 채널 식별</div>", unsafe_allow_html=True)
    if "channel" in df_m.columns:
        ch = df_m.groupby("channel", as_index=False).agg({
            "spend": "sum",
            "leads": "sum",
            "tm_attempts": "sum",
            "tm_connected": "sum",
            "contracts": "sum",
            "premium": "sum",
        })
        ch.rename(columns={
            "spend": "mtd_spend",
            "leads": "mtd_leads",
            "tm_attempts": "mtd_attempts",
            "tm_connected": "mtd_connected",
            "contracts": "mtd_contracts",
            "premium": "mtd_premium",
        }, inplace=True)

        # Legacy KPIs (keep for continuity)
        ch["CPL"] = ch.apply(lambda r: safe_div(r["mtd_spend"], r["mtd_leads"]), axis=1)
        ch["CVR"] = ch.apply(lambda r: safe_div(r["mtd_contracts"], r["mtd_leads"]), axis=1)
        ch["APC"] = ch.apply(lambda r: safe_div(r["mtd_premium"], r["mtd_contracts"]), axis=1)
        ch["ROI"] = ch.apply(lambda r: safe_div(r["mtd_premium"], r["mtd_spend"]), axis=1)

        # ✅ ROI 5-Factor decomposition
        ch["Leads/Spend"] = ch.apply(lambda r: safe_div(r["mtd_leads"], r["mtd_spend"]), axis=1)
        ch["Attempts/Leads"] = ch.apply(lambda r: safe_div(r["mtd_attempts"], r["mtd_leads"]), axis=1)
        ch["Connected/Attempts"] = ch.apply(lambda r: safe_div(r["mtd_connected"], r["mtd_attempts"]), axis=1)
        ch["Contracts/Connected"] = ch.apply(lambda r: safe_div(r["mtd_contracts"], r["mtd_connected"]), axis=1)
        ch["Premium/Contract"] = ch.apply(lambda r: safe_div(r["mtd_premium"], r["mtd_contracts"]), axis=1)

        ch["ROI_implied"] = (
            ch["Leads/Spend"]
            * ch["Attempts/Leads"]
            * ch["Connected/Attempts"]
            * ch["Contracts/Connected"]
            * ch["Premium/Contract"]
        )
        ch["ROI_gap"] = ch["ROI_implied"] - ch["ROI"]

        # bottleneck: choose the weakest link among the 5 factors (higher is better for ROI math)
        def _rank(series: pd.Series) -> pd.Series:
            s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            return s.rank(pct=True, ascending=True)

        r1 = _rank(ch["Leads/Spend"])
        r2 = _rank(ch["Attempts/Leads"])
        r3 = _rank(ch["Connected/Attempts"])
        r4 = _rank(ch["Contracts/Connected"])
        r5 = _rank(ch["Premium/Contract"])

        bottleneck = []
        for a, b, c, d, e in zip(r1, r2, r3, r4, r5):
            trio = {
                "Leads/Spend": a,
                "Attempts/Leads": b,
                "Connected/Attempts": c,
                "Contracts/Connected": d,
                "Premium/Contract": e,
            }
            # pick the smallest percentile (worst)
            bn = min(trio, key=lambda k: (trio[k] if pd.notna(trio[k]) else -1.0))
            bottleneck.append(bn)
        ch["병목"] = bottleneck

        # pretty format
        show = ch[[
            "channel",
            "mtd_spend",
            "mtd_leads",
            "mtd_attempts",
            "mtd_connected",
            "mtd_contracts",
            "mtd_premium",
            "Leads/Spend",
            "Attempts/Leads",
            "Connected/Attempts",
            "Contracts/Connected",
            "Premium/Contract",
            "ROI",
            "ROI_implied",
            "ROI_gap",
            "병목",
        ]].copy()

        show["mtd_spend"] = show["mtd_spend"].apply(format_won)
        show["mtd_premium"] = show["mtd_premium"].apply(format_won)
        show["mtd_leads"] = show["mtd_leads"].apply(money)
        show["mtd_attempts"] = show["mtd_attempts"].apply(money)
        show["mtd_connected"] = show["mtd_connected"].apply(money)
        show["mtd_contracts"] = show["mtd_contracts"].apply(money)

        show["Leads/Spend"] = show["Leads/Spend"].apply(lambda x: f"{x:.6f}" if np.isfinite(x) else "-")
        show["Attempts/Leads"] = show["Attempts/Leads"].apply(lambda x: f"{x:.2f}" if np.isfinite(x) else "-")
        show["Connected/Attempts"] = show["Connected/Attempts"].apply(lambda x: pct(x) if np.isfinite(x) else "-")
        show["Contracts/Connected"] = show["Contracts/Connected"].apply(lambda x: pct(x) if np.isfinite(x) else "-")
        show["Premium/Contract"] = show["Premium/Contract"].apply(lambda x: format_won(x) if np.isfinite(x) else "-")
        show["ROI"] = show["ROI"].apply(lambda x: f"{x:.4f}" if np.isfinite(x) else "-")
        show["ROI_implied"] = show["ROI_implied"].apply(lambda x: f"{x:.4f}" if np.isfinite(x) else "-")
        show["ROI_gap"] = show["ROI_gap"].apply(lambda x: f"{x:+.6f}" if np.isfinite(x) else "-")

        show.rename(columns={
            "channel": "채널",
            "mtd_spend": "MTD 예산",
            "mtd_leads": "MTD 리드",
            "mtd_attempts": "MTD 통화시도",
            "mtd_connected": "MTD 연결",
            "mtd_contracts": "MTD 계약",
            "mtd_premium": "MTD 보험료",
            "Leads/Spend": "Leads/Spend",
            "Attempts/Leads": "Attempts/Leads",
            "Connected/Attempts": "Connected/Attempts",
            "Contracts/Connected": "Contracts/Connected",
            "Premium/Contract": "Premium/Contract",
            "ROI": "ROI",
            "ROI_implied": "5-Factor 곱(ROI)",
            "ROI_gap": "차이(곱-실제)",
        }, inplace=True)

        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("병목 해석: 각 채널에서 5-Factor(Leads/Spend, Attempts/Leads, Connected/Attempts, Contracts/Connected, Premium/Contract) 중 상대적으로 가장 약한 단계(퍼센타일 기준) 표시")
    else:
        st.info("채널별 분해를 위해 out/panel_daily_channel.csv에 channel 컬럼 필요")



    # --- Required spend calculator (to hit target) ---
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("<div class='exec-section'><b>목표 달성 역산</b> (Required Spend Calculator)</div>", unsafe_allow_html=True)

    if gap <= 0:
        st.success("현재 MTD 기준으로 이번달 목표 보험료를 이미 달성했음.")
    else:
        cur_premium_per_spend = safe_div(mtd_premium, mtd_budget_used)
        req_budget_cur = safe_div(gap, cur_premium_per_spend) if cur_premium_per_spend > 0 else np.nan
        req_daily_spend_cur = safe_div(req_budget_cur, remaining_days) if remaining_days > 0 else np.nan

        # If optimizer simulation exists, show implied premium-per-spend under optimized allocation
        opt_budget_need = np.nan
        opt_daily_spend_need = np.nan
        if opt_simulation and isinstance(opt_simulation, dict) and opt_simulation.get("totals"):
            t = opt_simulation.get("totals", {})
            opt_budget = float(t.get("budget", np.nan))
            opt_premium = float(t.get("premium", np.nan))
            opt_premium_per_spend = safe_div(opt_premium, opt_budget) if np.isfinite(opt_budget) and opt_budget > 0 else 0.0
            if opt_premium_per_spend > 0:
                opt_budget_need = safe_div(gap, opt_premium_per_spend)
                opt_daily_spend_need = safe_div(opt_budget_need, remaining_days) if remaining_days > 0 else np.nan

        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(
            f"<div class='exec-card'><div class='exec-kpi'>{format_won(gap)}</div><div class='exec-sub'>남은 필요 보험료(Goal Gap)</div></div>",
            unsafe_allow_html=True,
        )
        r2.markdown(
            f"<div class='exec-card'><div class='exec-kpi'>{money(req_budget_cur) if np.isfinite(req_budget_cur) else '-'}" \
            f"</div><div class='exec-sub'>추가 필요 예산(현재 효율 기준)</div></div>",
            unsafe_allow_html=True,
        )
        _r3_val = money(req_daily_spend_cur) if (remaining_days > 0 and np.isfinite(req_daily_spend_cur)) else "-"
        r3.markdown(
            f"<div class='exec-card'><div class='exec-kpi'>{_r3_val}</div><div class='exec-sub'>일평균 추가 예산(현재 효율)</div></div>",
            unsafe_allow_html=True,
        )
        if remaining_days <= 0:
            r3.caption("분석 기준일이 월말(또는 이후)이라 일평균 추가 예산 계산 불가")
        r4.markdown(
            f"<div class='exec-card'><div class='exec-kpi'>{money(opt_budget_need) if np.isfinite(opt_budget_need) else '-'}" \
            f"</div><div class='exec-sub'>추가 필요 예산(Optimizer 효율 가정)</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='exec-note'>* 역산 가정: (현재) Premium/Spend 효율 고정. (Optimizer) 시뮬레이션에서 추정된 Premium/Spend 효율 고정.</div>",
            unsafe_allow_html=True,
        )


        # --- Channel-level goal closure (decomposed) ---
        st.markdown("<div class='exec-section'><b>채널별 목표 달성 역산</b></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='exec-note'>목표 미달(Gap)을 줄이기 위해, 채널별 한계효과(추가 예산 1 step당 예상 보험료 증가)를 기준으로 추가 예산을 배분함. "
            "모델은 최근 데이터로 채널별 Spend→Leads 포화(로그) 곡선을 근사하고, 채널별 RR/건당보험료(MTD)를 적용해 Premium 증가를 추정함.</div>",
            unsafe_allow_html=True,
        )

        # window for modeling (recent 8 weeks, fallback month)
        df_model = df_full.copy()
        if "date" in df_model.columns:
            df_model["date"] = pd.to_datetime(df_model["date"], errors="coerce")
            end = df_model["date"].max()
            start = end - pd.Timedelta(days=56)
            df_model = df_model[df_model["date"].between(start, end)]
        if df_model.empty:
            df_model = df_m.copy()

        # channel-level RR and premium-per-sale from MTD
        rr_by = df_m.groupby("channel").apply(lambda x: safe_div(x["contracts"].sum(), x["leads"].sum())).to_dict()
        pps_by = df_m.groupby("channel").apply(lambda x: safe_div(x["premium"].sum(), x["contracts"].sum())).to_dict()

        # Fill missing with global
        g_rr2 = safe_div(total_sales, total_leads)
        g_pps2 = safe_div(mtd_premium, total_sales)
        def _safe_float(x):
            """Coerce mixed types to float; return NaN on failure."""
            try:
                if x is None:
                    return np.nan
                if isinstance(x, (int, float, np.number)):
                    return float(x)
                return float(pd.to_numeric(x, errors="coerce"))
            except Exception:
                return np.nan

        # sanitize per-channel RR / premium-per-sale (avoid np.isfinite on non-numeric types)
        rr_by_clean = {}
        for k, v in rr_by.items():
            vv = _safe_float(v)
            rr_by_clean[k] = vv if np.isfinite(vv) and vv > 0 else g_rr2
        rr_by = rr_by_clean
        pps_by_clean = {}
        for k, v in pps_by.items():
            vv = _safe_float(v)
            pps_by_clean[k] = vv if np.isfinite(vv) and vv > 0 else g_pps2
        pps_by = pps_by_clean
        # Build lead models per channel
        lead_models = {}
        if "leads" in df_model.columns:
            for ch, one in df_model.groupby("channel"):
                lead_models[ch] = _fit_log_curve(one, "leads")
        else:
            st.warning("leads 컬럼이 없어 채널별 역산(포화모델)을 수행할 수 없음. out/panel_daily_channel.csv 확인 필요.")
            lead_models = {}

        cur_spend_by = df_m.groupby("channel")["spend"].sum().to_dict()
        channels = sorted(list(set(df_m["channel"].unique())))

        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        with c1:
            _cap_default = float(req_budget_cur) if np.isfinite(req_budget_cur) and req_budget_cur > 0 else float(total_budget * 0.2)
            extra_budget_cap = masked_number_input(
                "추가 예산 상한(시나리오)",
                key="extra_budget_cap",
                default=_cap_default,
                help="목표 갭을 메우기 위해 추가로 투입 가능한 예산(가정).",
            )
            extra_budget_cap = float(max(0.0, extra_budget_cap if np.isfinite(extra_budget_cap) else _cap_default))
        with c2:
            _step2_default = max(1000.0, float(extra_budget_cap) * 0.02)
            _step2 = st.text_input(
                "배분 단위(step)",
                value=money(_step2_default),
                help="추가 예산을 배분할 최소 단위. 값이 작을수록 정밀하지만 느려질 수 있음.",
            )
            step_budget2 = float(max(100.0, parse_number(_step2, default=_step2_default)))
            st.caption(f"입력값(해석): {money(step_budget2)}")
        with c3:
            include_channels = st.multiselect(
                "추가 집행 후보 채널(선택)",
                options=channels,
                default=channels,
                help="특정 채널을 제외하고 싶으면 여기서 빼면 됨.",
            )

        if len(include_channels) == 0:
            st.info("추가 집행 후보 채널을 1개 이상 선택 필요.")
        elif not lead_models:
            st.info("채널별 Spend→Leads 모델을 만들 수 없어 역산을 수행하지 않음.")
        else:
            alloc_df, alloc_sum = _greedy_allocate_to_close_gap(
                include_channels,
                cur_spend_by,
                rr_by,
                pps_by,
                lead_models,
                gap,
                step_budget=float(step_budget2),
                max_extra_budget=float(extra_budget_cap),
            )

            # Format and show insights
            coverage = float(alloc_sum.get("coverage", 0.0))
            st.markdown(
                f"<div class='exec-card'><div class='exec-kpi'>{pct(min(coverage, 1.0))}</div>"
                f"<div class='exec-sub'>목표 갭 커버리지(추정) = 추가예산 배분으로 메우는 비율</div></div>",
                unsafe_allow_html=True,
            )

            # add total row for display
            alloc_disp = alloc_df.copy()
            try:
                tot = {
                    "channel": "합계",
                    "current_spend": float(pd.to_numeric(alloc_disp["current_spend"], errors="coerce").fillna(0.0).sum()),
                    "extra_spend": float(pd.to_numeric(alloc_disp["extra_spend"], errors="coerce").fillna(0.0).sum()),
                    "expected_premium_gain": float(pd.to_numeric(alloc_disp["expected_premium_gain"], errors="coerce").fillna(0.0).sum()),
                    "marginal_premium_per_step": float(pd.to_numeric(alloc_disp["marginal_premium_per_step"], errors="coerce").fillna(0.0).mean()),
                }
                alloc_disp = pd.concat([alloc_disp, pd.DataFrame([tot])], ignore_index=True)
            except Exception:
                pass

            show = alloc_disp.copy()
            show["marginal_premium_per_step"] = show["marginal_premium_per_step"].apply(money)
            show["expected_premium_gain"] = show["expected_premium_gain"].apply(money)
            show["extra_spend"] = show["extra_spend"].apply(money)
            show["current_spend"] = show["current_spend"].apply(money)

            st.dataframe(
                show.rename(
                    columns={
                        "channel": "채널",
                        "current_spend": "현재 예산(MTD)",
                        "extra_spend": "추가 예산(배분)",
                        "expected_premium_gain": "예상 보험료 증가",
                        "marginal_premium_per_step": f"한계 보험료 증가/step",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            top3 = alloc_df.head(3)["channel"].tolist()
            st.markdown(
                f"<div class='exec-note'><b>추가 집행 우선순위(추정):</b> {', '.join(top3)} "
                f"(한계효과 및 포화 고려)</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # --- Breakdown ---
    st.markdown("<div class='exec-section'><b>매체별 Breakdown</b></div>", unsafe_allow_html=True)
    need_cols = {"channel", "spend", "leads", "contracts", "premium"}
    if not need_cols.issubset(set(df_m.columns)):
        st.info("매체별 Breakdown 표를 만들기 위한 컬럼이 부족함 (필요: channel, spend, leads, contracts, premium).")
    else:
        breakdown = df_m.groupby("channel", as_index=False).agg(
            예산=("spend", "sum"),
            Lead=("leads", "sum"),
            Sales=("contracts", "sum"),
            총보험료=("premium", "sum"),
        )
        breakdown["RR"] = breakdown.apply(lambda r: safe_div(r["Sales"], r["Lead"]), axis=1)
        breakdown["건당 보험료"] = breakdown.apply(lambda r: safe_div(r["총보험료"], r["Sales"]), axis=1)

        # Additional metrics for consulting-style diagnosis
        breakdown["ROI"] = breakdown.apply(lambda r: safe_div(r["총보험료"], r["예산"]), axis=1)  # premium/spend
        breakdown["CPL"] = breakdown.apply(lambda r: safe_div(r["예산"], r["Lead"]), axis=1)      # spend/lead
        breakdown["CPS"] = breakdown.apply(lambda r: safe_div(r["예산"], r["Sales"]), axis=1)     # spend/sale

        breakdown = breakdown.replace([np.inf, -np.inf], 0).fillna(0)
        breakdown = add_total_row_breakdown(breakdown, total_rr=rr, total_premium_per_sale=premium_per_sale)

        # Total row: also compute ROI/CPL/CPS for the appended total
        # (safe: may not exist if add_total_row_breakdown signature changes)
        try:
            idx_total = breakdown["channel"].astype(str).eq("합계")
            if idx_total.any():
                b = breakdown.loc[idx_total].iloc[0]
                roi_t = safe_div(float(b["총보험료"]), float(b["예산"]))
                cpl_t = safe_div(float(b["예산"]), float(b["Lead"]))
                cps_t = safe_div(float(b["예산"]), float(b["Sales"]))
                breakdown.loc[idx_total, "ROI"] = roi_t
                breakdown.loc[idx_total, "CPL"] = cpl_t
                breakdown.loc[idx_total, "CPS"] = cps_t
        except Exception:
            pass

        # Auto commentary (top/bottom channels by ROI, excluding total)
        try:
            diag = breakdown[breakdown["channel"].astype(str) != "합계"].copy()
            diag["ROI"] = pd.to_numeric(diag["ROI"], errors="coerce").fillna(0.0)
            diag["RR"] = pd.to_numeric(diag["RR"], errors="coerce").fillna(0.0)
            top_roi = diag.sort_values("ROI", ascending=False).head(1)
            low_roi = diag.sort_values("ROI", ascending=True).head(1)
            top_rr = diag.sort_values("RR", ascending=False).head(1)
            low_rr = diag.sort_values("RR", ascending=True).head(1)

            def _chrow(d: pd.DataFrame):
                if d.empty:
                    return ("-", 0.0)
                return (str(d.iloc[0]["channel"]), float(d.iloc[0]["ROI"]))

            tr_ch, tr_v = _chrow(top_roi)
            lr_ch, lr_v = _chrow(low_roi)
            rr_top_ch = str(top_rr.iloc[0]["channel"]) if not top_rr.empty else "-"
            rr_top_v = float(top_rr.iloc[0]["RR"]) if not top_rr.empty else 0.0
            rr_low_ch = str(low_rr.iloc[0]["channel"]) if not low_rr.empty else "-"
            rr_low_v = float(low_rr.iloc[0]["RR"]) if not low_rr.empty else 0.0

            st.markdown(
                "<div class='exec-note'><b>Auto Insight</b><br>"
                f"- ROI 상위 채널: <b>{tr_ch}</b> ({tr_v:.2f}x) / 하위 채널: <b>{lr_ch}</b> ({lr_v:.2f}x)<br>"
                f"- RR 상위 채널: <b>{rr_top_ch}</b> ({rr_top_v*100:.1f}%) / 하위 채널: <b>{rr_low_ch}</b> ({rr_low_v*100:.1f}%)"
                "</div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        # format
        for col in ["예산", "총보험료", "건당 보험료", "CPL", "CPS"]:
            breakdown[col] = breakdown[col].apply(money)
        for col in ["Lead", "Sales"]:
            breakdown[col] = breakdown[col].apply(lambda x: f"{int(round(float(x))):,}")
        breakdown["RR"] = breakdown["RR"].apply(pct)
        breakdown["ROI"] = breakdown["ROI"].apply(lambda x: f"{float(x):.2f}x" if np.isfinite(float(x)) else "-")

        st.dataframe(
            breakdown[["channel","예산","Lead","Sales","RR","CPL","CPS","건당 보험료","총보험료","ROI"]],
            use_container_width=True,
            height=340,
            hide_index=True,
        )

    # --- Simulation effect ---
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("<div class='exec-section'><b>예산 배분 시뮬레이션 효과</b> (Optimizer 적용 결과)</div>", unsafe_allow_html=True)


    # --- Month Planner (A/B/C) projection & recommendation ---
    mp = None
    if isinstance(opt_simulation, dict):
        mp = opt_simulation.get("month_planner")
    if isinstance(mp, dict):
        st.markdown("<div class='exec-section'><b>월말 Projection (A/B/C) · 목표 달성 가능성</b></div>", unsafe_allow_html=True)

        target_premium = float(mp.get("target_premium", monthly_target))
        sc = mp.get("scenarios", {}) if isinstance(mp.get("scenarios"), dict) else {}

        def _scenario_card(name: str, month_end: dict) -> str:
            p = float(month_end.get("premium", 0.0))
            l = float(month_end.get("leads", 0.0))
            c = float(month_end.get("contracts", 0.0))
            att = safe_div(p, target_premium) if target_premium > 0 else np.nan
            badge = "달성 가능" if p >= target_premium else "달성 어려움"
            return f"""<div class='card'>
                <div class='smallcap'>{name}</div>
                <div class='kpi'>{money(p)}</div>
                <div class='kpi_sub'>Premium(월말 예측) · 달성률 {pct(att)}</div>
                <div class='smallcap'>Leads {l:,.0f} · Contracts {c:,.0f} · {badge}</div>
            </div>"""

        cA, cB, cC = st.columns(3)
        A = sc.get("A_engine", {}); B = sc.get("B_keep_mix", {}); C = sc.get("C_manual", {})
        cA.markdown(_scenario_card("A) 엔진 권장", (A.get("month_end") or {})), unsafe_allow_html=True)
        cB.markdown(_scenario_card("B) 현재 Mix 유지", (B.get("month_end") or {})), unsafe_allow_html=True)
        cC.markdown(_scenario_card("C) Manual", (C.get("month_end") or {})), unsafe_allow_html=True)

        # Benefit: Engine vs Human
        ben = mp.get("benefit", {}) if isinstance(mp.get("benefit"), dict) else {}
        b1, b2, b3 = st.columns(3)
        b1.markdown(f"<div class='card'><div class='smallcap'>Benefit</div><div class='kpi'>{money(float(ben.get('engine_vs_keep_mix', 0.0)))}</div><div class='kpi_sub'>엔진 vs Mix 유지</div></div>", unsafe_allow_html=True)
        b2.markdown(f"<div class='card'><div class='smallcap'>Benefit</div><div class='kpi'>{money(float(ben.get('engine_vs_manual', 0.0)))}</div><div class='kpi_sub'>엔진 vs Manual</div></div>", unsafe_allow_html=True)
        b3.markdown(f"<div class='card'><div class='smallcap'>Benefit</div><div class='kpi'>{money(float(ben.get('engine_vs_best_human', 0.0)))}</div><div class='kpi_sub'>엔진 vs Best Human</div></div>", unsafe_allow_html=True)

        # Final recommendation
        st.markdown("<div class='exec-section'><b>최종 제안</b></div>", unsafe_allow_html=True)
        # choose best scenario by premium, tie-break: meets target
        choices = []
        for key, label in [("A_engine","A) 엔진 권장"),("B_keep_mix","B) 현재 Mix 유지"),("C_manual","C) Manual")]:
            s = sc.get(key, {})
            me = s.get("month_end", {}) if isinstance(s, dict) else {}
            p = float(me.get("premium", 0.0))
            meet = int(p >= target_premium)
            choices.append((meet, p, label, key))
        choices.sort(reverse=True)
        best = choices[0]
        st.markdown(
            f"""<div class='card'>
                <div class='smallcap'>추천 시나리오</div>
                <div class='kpi'>{best[2]}</div>
                <div class='kpi_sub'>우선순위: 목표 달성 가능성 → Premium 최대화</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption("해석: 목표 달성 가능(달성률)과 월말 Premium(예측)을 동시에 고려한 추천임. Manual이 엔진보다 좋게 나오면, 해당 배분을 그대로 엔진 제약/탐험 규칙에 반영하는 것이 바람직함.")
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    # NOTE: Budget Optimizer 탭을 실행하기 전에는 session_state에 opt_simulation이 없을 수 있음.
    # 기존 코드에서는 opt_simulation이 None일 때 .get() 호출로 AttributeError가 발생했음.
    if not isinstance(opt_simulation, dict) or len(opt_simulation) == 0:
        st.info("Optimizer 시뮬레이션이 아직 실행되지 않았음. (Budget Optimizer 탭에서 실행하면 여기에 자동 반영됨)")
        # month_planner (Budget Optimizer 탭의 This Month 플래너)는 opt_simulation 없이도 표시 가능
        opt_simulation = {"objective": "", "totals": {}}

    # opt_simulation expected keys (new schema from optimizer.py)
    obj = str(opt_simulation.get("objective", ""))
    t = opt_simulation.get("totals", {}) if isinstance(opt_simulation.get("totals", {}), dict) else {}
    opt_budget = float(t.get("budget", np.nan))
    exp_leads = float(t.get("leads", np.nan))
    exp_contracts = float(t.get("contracts", np.nan))
    exp_premium = float(t.get("premium", np.nan))

    exp_rr = safe_div(exp_contracts, exp_leads) if np.isfinite(exp_leads) else np.nan
    exp_pps = safe_div(exp_premium, exp_contracts) if np.isfinite(exp_contracts) else np.nan

    cur = {
        "예산": total_budget,
        "Lead": total_leads,
        "Sales": total_sales,
        "RR": rr,
        "건당 보험료": premium_per_sale,
        "총보험료": mtd_premium,
    }
    opt = {
        "예산": opt_budget if np.isfinite(opt_budget) else total_budget,
        "Lead": exp_leads,
        "Sales": exp_contracts,
        "RR": exp_rr if np.isfinite(exp_rr) else rr,
        "건당 보험료": exp_pps if np.isfinite(exp_pps) else premium_per_sale,
        "총보험료": exp_premium,
    }

    rep = pd.DataFrame({"항목": list(cur.keys()), "현재": list(cur.values()), "시뮬레이션(최적화)": [opt[k] for k in cur.keys()]})
    rep["변화"] = rep["시뮬레이션(최적화)"] - rep["현재"]

    # format per row (safe)
    def fmt_cell(item, v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        if item == "RR":
            return pct(float(v))
        return money(float(v))

    rep_disp = rep.copy()
    rep_disp["현재"] = rep_disp.apply(lambda r: fmt_cell(r["항목"], r["현재"]), axis=1)
    rep_disp["시뮬레이션(최적화)"] = rep_disp.apply(lambda r: fmt_cell(r["항목"], r["시뮬레이션(최적화)"]), axis=1)
    rep_disp["변화"] = rep_disp.apply(lambda r: fmt_cell(r["항목"], r["변화"]), axis=1)

    st.dataframe(rep_disp, use_container_width=True, height=260, hide_index=True)

    # headline insight
    if np.isfinite(exp_premium):
        delta = exp_premium - mtd_premium
        st.markdown(
            f"<div class='exec-note'>요약: Optimizer({obj}) 적용 시 총보험료가 <b>{money(exp_premium)}</b>로 추정되며, "
            f"현재 대비 <b>{money(delta)}</b> 변화(채널별 RR/건당보험료를 MTD 기준으로 반영).</div>",
            unsafe_allow_html=True,
        )