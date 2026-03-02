
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from .utils import money, safe_div, parse_number, masked_number_input
from .formatting import format_won
from .tracking import save_plan_run
from .optimizer_core import ChannelConstraint, optimize_min_spend_for_target


def render_optimizer_engine(
    df_window: pd.DataFrame,
    *,
    tot_spend: float,
    bud: pd.DataFrame | None,
    history_start,
    history_end,
    forecast_start,
    forecast_end,
    premium_target: float = 0.0,
) -> None:
    """
    Budget Optimizer (Act)
    - Objective: Premium / Leads / Contracts
    - Fixed total budget + step allocation
    - Channel lock + min/max ratios
    - Simple saturation curve (log) heuristic
    - Stores simulation summary to st.session_state["opt_simulation"]
    """
    st.markdown("## Budget Optimizer")
    st.markdown("<div class='smallcap'>채널 Lock + 총예산 고정 + 목표 선택 기반 최적화 엔진</div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    # Period context
    st.markdown(
        f"<div class='note'>최적화 과거기간: {history_start} ~ {history_end} · 성과기간(Impact): {forecast_start} ~ {forecast_end}</div>",
        unsafe_allow_html=True,
    )


    if df_window is None or df_window.empty:
        st.info("윈도우 데이터가 없어 최적화를 수행할 수 없음.")
        return

    # ------------------------------------------------------------
    # (NEW) This Month Budget Planner
    # - Monthly total budget (available this month)
    # - MTD funnel (Spend → Leads → Contracts → Premium)
    # - Remaining budget allocation + end-of-month forecast
    # ------------------------------------------------------------
    # 기준월: df_window 내 최신 날짜가 속한 달
    if "date" in df_window.columns:
        _max_dt = pd.to_datetime(df_window["date"], errors="coerce").dropna().max()
    else:
        _max_dt = pd.Timestamp.today().normalize()
    as_of = _max_dt.normalize()
    month_start = as_of.replace(day=1)
    month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
    days_elapsed = int((as_of - month_start).days) + 1
    days_in_month = int((month_end - month_start).days) + 1
    days_remaining = int(max(0, (month_end - as_of).days))

    df_window2 = df_window.copy()
    if "date" in df_window2.columns:
        df_window2["date"] = pd.to_datetime(df_window2["date"], errors="coerce")
    df_mtd = df_window2
    if "date" in df_window2.columns:
        df_mtd = df_window2[(df_window2["date"] >= month_start) & (df_window2["date"] <= as_of)].copy()

    mtd_spend = float(pd.to_numeric(df_mtd.get("spend", 0.0), errors="coerce").fillna(0.0).sum())
    mtd_leads = float(pd.to_numeric(df_mtd.get("leads", 0.0), errors="coerce").fillna(0.0).sum())
    mtd_contracts = float(pd.to_numeric(df_mtd.get("contracts", 0.0), errors="coerce").fillna(0.0).sum())
    mtd_premium = float(pd.to_numeric(df_mtd.get("premium", 0.0), errors="coerce").fillna(0.0).sum())

    st.markdown("### This Month: Budget & Funnel (이번달 예산/퍼널)")
    st.markdown(
        f"<div class='note'>기준월: {month_start.date().isoformat()} ~ {month_end.date().isoformat()} · 기준일: {as_of.date().isoformat()}</div>",
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    with b1:
        monthly_budget = masked_number_input(
            "이번달 총 예산(Available)",
            key="opt_monthly_total_budget",
            default=float(max(mtd_spend, tot_spend)),
        )
        monthly_budget = float(max(0.0, monthly_budget if np.isfinite(monthly_budget) else float(max(mtd_spend, tot_spend))))
        # Sync with Executive Summary (shared state)
        st.session_state["monthly_total_budget"] = monthly_budget
        st.caption(f"입력값(해석): {format_won(monthly_budget)}")
    with b2:
        st.markdown(f"<div class='card'><div class='kpi'>{format_won(mtd_spend)}</div><div class='kpi_sub'>MTD 사용 예산</div></div>", unsafe_allow_html=True)
    with b3:
        remaining_budget = float(max(0.0, monthly_budget - mtd_spend))
        st.markdown(f"<div class='card'><div class='kpi'>{format_won(remaining_budget)}</div><div class='kpi_sub'>남은 예산</div></div>", unsafe_allow_html=True)
    with b4:
        burn_rate = safe_div(mtd_spend, days_elapsed)
        st.markdown(f"<div class='card'><div class='kpi'>{format_won(burn_rate)}</div><div class='kpi_sub'>일평균 소진</div></div>", unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    f1.markdown(f"<div class='card'><div class='smallcap'>Spend</div><div class='kpi'>{format_won(mtd_spend)}</div><div class='kpi_sub'>MTD</div></div>", unsafe_allow_html=True)
    f2.markdown(f"<div class='card'><div class='smallcap'>Leads</div><div class='kpi'>{money(mtd_leads)}</div><div class='kpi_sub'>CPL {format_won(safe_div(mtd_spend, mtd_leads))}</div></div>", unsafe_allow_html=True)
    f3.markdown(f"<div class='card'><div class='smallcap'>Contracts</div><div class='kpi'>{money(mtd_contracts)}</div><div class='kpi_sub'>RR {safe_div(mtd_contracts, mtd_leads)*100:.1f}%</div></div>", unsafe_allow_html=True)
    f4.markdown(f"<div class='card'><div class='smallcap'>Premium</div><div class='kpi'>{format_won(mtd_premium)}</div><div class='kpi_sub'>ROI {safe_div(mtd_premium, mtd_spend):.2f}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # objective & budgets
    oc1, oc2, oc3 = st.columns([1,1,2])
    with oc1:
        st.markdown("**고정 Premium Target**")
        st.caption("조직 기준으로 고정(설정 파일 기반) · 화면에서 수정 불가")
        st.markdown(f"<div class='card'><div class='smallcap'>Premium Target</div><div class='kpi'>{format_won(premium_target)}</div><div class='kpi_sub'>성과기간(Impact) 기준</div></div>", unsafe_allow_html=True)
        objective = "ROI(=Premium 최대화)"
    with oc2:
        # Use text_input to show thousand separators reliably
        total_budget = masked_number_input("총예산(고정)", key="optimizer_total_budget", default=float(tot_spend))
        total_budget = float(max(0.0, total_budget if np.isfinite(total_budget) else float(tot_spend)))
    with oc3:
        _step_default = max(1000.0, float(total_budget) * 0.005)
        step_budget = masked_number_input("최적화 단위(증분)", key="optimizer_step_budget", default=_step_default)
        step_budget = float(max(100.0, step_budget if np.isfinite(step_budget) else _step_default))
        #fault)))
        st.caption(f"입력값(해석): {format_won(step_budget)}")

    # Objective column (for display only). Internally we always predict:
    #   spend -> leads (saturation), then channel-level RR & premium/sale to derive contracts & premium.
    if objective.startswith("ROI"):
        ycol = "premium"
    elif objective.startswith("Leads"):
        ycol = "leads"
    else:
        ycol = "contracts"

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # constraints
    cur_by_ch = df_window.groupby("channel", as_index=False).agg(
        current_spend=("spend","sum"),
    )
    gc1, gc2, gc3 = st.columns([1,1,2])
    with gc1:
        default_min = st.number_input("기본 최소비율 (현재 대비)", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
    with gc2:
        default_max = st.number_input("기본 최대비율 (현재 대비)", min_value=0.0, max_value=10.0, value=1.6, step=0.05)
    with gc3:
        st.markdown("<div class='note'>예: 0.7~1.6이면 현재 예산 대비 -30%~+60% 범위 내에서만 조정. Lock 채널은 현재 고정.</div>", unsafe_allow_html=True)

    guard = cur_by_ch[["channel","current_spend"]].copy()
    guard["lock"] = False
    guard["min_ratio"] = float(default_min)
    guard["max_ratio"] = float(default_max)
    guard = guard.rename(columns={"channel":"채널","current_spend":"현재"})
    edited = st.data_editor(
        guard,
        use_container_width=True,
        hide_index=True,
        column_config={
            "lock": st.column_config.CheckboxColumn("Lock", help="선택 시 해당 채널은 현재 예산 고정"),
            "min_ratio": st.column_config.NumberColumn("Min", min_value=0.0, max_value=2.0, step=0.05),
            "max_ratio": st.column_config.NumberColumn("Max", min_value=0.0, max_value=10.0, step=0.05),
        },
    )

    # ------------------------------------------------------------
    # Baseline conversion params (channel-specific)
    # ------------------------------------------------------------
    by_ch = df_window.groupby("channel", as_index=False).agg(
        spend=("spend", "sum"),
        leads=("leads", "sum") if "leads" in df_window.columns else ("spend", "sum"),
        contracts=("contracts", "sum") if "contracts" in df_window.columns else ("spend", "sum"),
        premium=("premium", "sum") if "premium" in df_window.columns else ("spend", "sum"),
    )
    g_rr = safe_div(float(by_ch["contracts"].sum()), float(by_ch["leads"].sum()))
    g_pps = safe_div(float(by_ch["premium"].sum()), float(by_ch["contracts"].sum()))

    rr_map: dict[str, float] = {}
    pps_map: dict[str, float] = {}
    for _, r in by_ch.iterrows():
        ch = str(r["channel"])
        rr = safe_div(float(r["contracts"]), float(r["leads"]))
        pps = safe_div(float(r["premium"]), float(r["contracts"]))
        rr_map[ch] = rr if rr > 0 else g_rr
        pps_map[ch] = pps if pps > 0 else g_pps

    # ------------------------------------------------------------
    # Fit saturation (log) for spend -> leads (primary), linear fallback
    # ------------------------------------------------------------
    def fit_log(one: pd.DataFrame, ycol: str):
        one = one[["spend", ycol]].copy()
        one["spend"] = pd.to_numeric(one["spend"], errors="coerce").fillna(0.0)
        one[ycol] = pd.to_numeric(one[ycol], errors="coerce").fillna(0.0)

        s = one["spend"].values
        y = one[ycol].values

        if (np.nanmax(s) <= 0) or (np.nanmax(y) <= 0) or (len(one) < 8):
            eff = float(np.sum(y) / max(1.0, np.sum(s)))
            def pred(x):
                return eff * np.asarray(x, dtype=float)
            return {"kind":"linear", "eff":eff, "pred":pred}

        # gamma grid by spend quantiles
        q = np.quantile(s[s>0], [0.2,0.4,0.6,0.8]) if np.any(s>0) else np.array([1,2,3,4], dtype=float)
        gammas = np.unique(np.clip(q, 1.0, None))
        if len(gammas) < 3:
            gammas = np.unique(np.clip(np.array([np.mean(s[s>0]) if np.any(s>0) else 1.0, np.max(s), np.median(s[s>0]) if np.any(s>0) else 1.0]), 1.0, None))

        best = None
        for g in gammas:
            x = np.log1p(s / float(g))
            denom = float(np.dot(x, x)) + 1e-9
            a = float(np.dot(x, y) / denom)
            predy = a * x
            mse = float(np.mean((predy - y)**2))
            if (best is None) or (mse < best["mse"]):
                best = {"a":a, "g":float(g), "mse":mse}

        a = best["a"]; g = best["g"]
        def pred(x):
            x = np.asarray(x, dtype=float)
            return a * np.log1p(x / g)
        return {"kind":"log", "a":a, "g":g, "pred":pred}

    # Build per-channel spend->leads predictors
    lead_models: dict[str, dict] = {}
    if "leads" not in df_window.columns:
        st.warning("leads 컬럼이 없어 최적화 엔진(채널별 전환 체인)을 사용할 수 없음.")
        return

    for ch, one in df_window.groupby("channel"):
        lead_models[ch] = fit_log(one, "leads")

    def pred_leads(ch: str, spend: float) -> float:
        m = lead_models.get(ch)
        if m is None:
            return 0.0
        return float(m["pred"]([spend])[0])

    def pred_contracts(ch: str, spend: float) -> float:
        return pred_leads(ch, spend) * float(rr_map.get(ch, g_rr))

    def pred_premium(ch: str, spend: float) -> float:
        return pred_contracts(ch, spend) * float(pps_map.get(ch, g_pps))

    def pred_objective(ch: str, spend: float) -> float:
        if ycol == "leads":
            return pred_leads(ch, spend)
        if ycol == "contracts":
            return pred_contracts(ch, spend)
        # premium
        return pred_premium(ch, spend)

    # ------------------------------------------------------------
    # (NEW) Remaining budget allocation (this month)
    # ------------------------------------------------------------
    
    if remaining_budget > 0 and days_remaining > 0:
        st.markdown("### Executive Target + 남은 예산 배분 + 월말 목표 달성 가능성")
        st.markdown(
            "<div class='note'>가정: 남은 예산은 남은 일수에 균등 집행(일 단위). 채널별 Spend→Leads는 최근 데이터에서 포화(로그)로 근사. RR/건당 보험료는 MTD 기준으로 적용.</div>",
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------
        # (NEW) Executive Target 연결: 목표 Premium 입력 → 목표 달성 가능성(예산 제약 포함)
        # --------------------------------------------------------
        col_t1, col_t2, col_t3 = st.columns([1.2, 1.0, 1.0])
        with col_t1:
            target_premium = masked_number_input(
                "이번달 목표 보험료(월말 Premium 목표)",
                key="mtd_target_premium",
                default=float(max(mtd_premium, 0.0)) * 1.20,
                help="이번달 월말(말일)까지 달성하고 싶은 Premium 목표(원).",
            )
        with col_t2:
            target_metric = st.selectbox(
                "배분 최적화 기준(목표 지표)",
                options=["Premium", "Leads", "Contracts"],
                index=0,
                help="남은 예산을 어떤 목표 지표 기준으로 배분할지 선택.",
                key="mtd_target_metric",
            )
        with col_t3:
            scenario_view = st.radio(
                "비교 시나리오",
                options=["3안 비교(A/B/C)", "상세(채널별)"],
                horizontal=True,
                index=0,
                key="mtd_scenario_view",
            )

        # MTD 기준 일평균 spend (채널별)
        mtd_by_ch = df_mtd.groupby("channel", as_index=False).agg(spend=("spend", "sum"))
        cur_daily = {str(r["channel"]): float(r["spend"]) / max(1, days_elapsed) for _, r in mtd_by_ch.iterrows()}

        # 채널 목록은 모델이 생성된 채널 기준
        chs = sorted(list(lead_models.keys()))
        for ch in chs:
            cur_daily.setdefault(ch, 0.0)

        # 목표 지표별 gain 함수
        def gain_fn(ch: str, s1: float, s0: float) -> float:
            if target_metric == "Leads":
                return float(pred_leads(ch, s1) - pred_leads(ch, s0))
            if target_metric == "Contracts":
                return float(pred_contracts(ch, s1) - pred_contracts(ch, s0))
            # default Premium
            return float(pred_premium(ch, s1) - pred_premium(ch, s0))

        # step 설정(총액/일액)
        step_total = float(max(1000.0, remaining_budget * 0.01))
        step_total = float(min(step_total, max(1000.0, remaining_budget / 20.0)))
        step_total = float(max(1000.0, step_total))
        step_daily = step_total / float(days_remaining)

        # --------------------------------------------------------
        # 시나리오 생성
        # A) 엔진 권장(한계효과 기반)
        # B) 현재 Mix 유지
        # C) 업무자 수동 배분(Manual)
        # --------------------------------------------------------
        def build_add_total_keep_mix() -> dict[str, float]:
            add_total = {ch: 0.0 for ch in chs}
            tot_cur = float(sum(cur_daily.values()))
            if tot_cur <= 0:
                for ch in chs:
                    add_total[ch] = remaining_budget / float(len(chs))
            else:
                for ch in chs:
                    add_total[ch] = remaining_budget * (cur_daily[ch] / tot_cur)
            return add_total

        def build_add_total_recommended() -> dict[str, float]:
            add_total = {ch: 0.0 for ch in chs}
            rem = float(remaining_budget)
            it = 0
            max_it = 5000
            while rem >= step_total - 1e-9 and it < max_it:
                best_ch = None
                best_gain = -1e18
                for ch in chs:
                    s0 = float(cur_daily[ch] + (add_total[ch] / float(days_remaining)))
                    s1 = float(s0 + step_daily)
                    gain_day = gain_fn(ch, s1, s0)
                    gain = float(gain_day) * float(days_remaining)  # month-total gain
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_ch = ch
                if best_ch is None:
                    break
                add_total[best_ch] += step_total
                rem -= step_total
                it += 1
            if rem > 1e-6:
                for ch in chs:
                    add_total[ch] += rem / float(len(chs))
            return add_total

        def simulate_plan(add_total: dict[str, float]) -> dict:
            rows_plan = []
            inc_leads = 0.0
            inc_contracts = 0.0
            inc_premium = 0.0
            for ch in chs:
                add_d = float(add_total.get(ch, 0.0) / float(days_remaining))
                s0 = float(cur_daily[ch])
                s1 = float(s0 + add_d)
                d_leads = (pred_leads(ch, s1) - pred_leads(ch, s0)) * float(days_remaining)
                d_contracts = (pred_contracts(ch, s1) - pred_contracts(ch, s0)) * float(days_remaining)
                d_premium = (pred_premium(ch, s1) - pred_premium(ch, s0)) * float(days_remaining)
                inc_leads += float(d_leads)
                inc_contracts += float(d_contracts)
                inc_premium += float(d_premium)
                rows_plan.append({
                    "채널": ch,
                    "추가 배정(남은예산)": float(add_total.get(ch, 0.0)),
                    "추가 Leads(예측)": float(d_leads),
                    "추가 Contracts(예측)": float(d_contracts),
                    "추가 Premium(예측)": float(d_premium),
                })
            plan = pd.DataFrame(rows_plan).sort_values("추가 배정(남은예산)", ascending=False)
            return {
                "plan": plan,
                "inc_leads": float(inc_leads),
                "inc_contracts": float(inc_contracts),
                "inc_premium": float(inc_premium),
                "month_end": {
                    "spend": float(mtd_spend + remaining_budget),
                    "leads": float(mtd_leads + inc_leads),
                    "contracts": float(mtd_contracts + inc_contracts),
                    "premium": float(mtd_premium + inc_premium),
                }
            }

        add_A = build_add_total_recommended()
        add_B = build_add_total_keep_mix()

        # Manual 입력(채널별 추가 배정)
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("#### Scenario C) 업무자 수동 배분(Manual)")
        manual_seed = pd.DataFrame({"채널": chs, "추가 배정(남은예산)": [float(add_B.get(ch, 0.0)) for ch in chs]})
        manual_edit = st.data_editor(
            manual_seed,
            use_container_width=True,
            height=260,
            hide_index=True,
            column_config={
                "채널": st.column_config.TextColumn(disabled=True),
                "추가 배정(남은예산)": st.column_config.NumberColumn(format="%,d", help="남은 예산(원) 중 채널별로 추가 배정할 금액"),
            },
            key="mtd_manual_alloc_editor",
        )
        manual_sum = float(pd.to_numeric(manual_edit["추가 배정(남은예산)"], errors="coerce").fillna(0.0).sum())
        c_fix1, c_fix2 = st.columns([1.2, 1.0])
        with c_fix1:
            st.caption(f"Manual 합계: {money(manual_sum)} / 남은 예산: {money(remaining_budget)}")
        with c_fix2:
            auto_norm = st.toggle("합계 자동 정규화(남은 예산에 맞춤)", value=True, key="mtd_manual_autonorm")

        add_C = {ch: 0.0 for ch in chs}
        if manual_sum <= 0:
            add_C = add_B.copy()
        else:
            scale = float(remaining_budget / manual_sum) if auto_norm else 1.0
            for _, r in manual_edit.iterrows():
                ch = str(r["채널"])
                v = float(pd.to_numeric(r["추가 배정(남은예산)"], errors="coerce") or 0.0)
                add_C[ch] = max(0.0, v * scale)

        # 시뮬레이션(A/B/C)
        sim_A = simulate_plan(add_A)
        sim_B = simulate_plan(add_B)
        sim_C = simulate_plan(add_C)

        # store month planner simulation into session_state for Executive Summary
        month_planner_sim = {
            "objective": str(target_metric),
            "target_premium": float(target_premium),
            "as_of": str(as_of.date().isoformat()),
            "month_start": str(month_start.date().isoformat()),
            "month_end": str(month_end.date().isoformat()),
            "budget": {
                "monthly_budget": float(monthly_budget),
                "mtd_spend": float(mtd_spend),
                "remaining_budget": float(remaining_budget),
                "days_elapsed": int(days_elapsed),
                "days_remaining": int(days_remaining),
            },
            "mtd": {
                "leads": float(mtd_leads),
                "contracts": float(mtd_contracts),
                "premium": float(mtd_premium),
            },
            "scenarios": {
                "A_engine": {"name": "A) 엔진 권장", **sim_A},
                "B_keep_mix": {"name": "B) 현재 Mix 유지", **sim_B},
                "C_manual": {"name": "C) Manual", **sim_C},
            },
        }
        # benefit: engine vs keep-mix and vs manual (premium 기준)
        pA = float(sim_A["month_end"]["premium"]); pB = float(sim_B["month_end"]["premium"]); pC = float(sim_C["month_end"]["premium"])
        month_planner_sim["benefit"] = {
            "engine_vs_keep_mix": float(pA - pB),
            "engine_vs_manual": float(pA - pC),
            "best_human": float(max(pB, pC)),
            "engine_vs_best_human": float(pA - max(pB, pC)),
        }
        existing = st.session_state.get("opt_simulation") if isinstance(st.session_state.get("opt_simulation"), dict) else {}
        existing = dict(existing) if existing else {}
        existing["month_planner"] = month_planner_sim
        st.session_state["opt_simulation"] = existing


        # --------------------------------------------------------
        # A/B/C 비교 화면 (목표 달성 가능성 포함)
        # --------------------------------------------------------
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("#### A/B/C 성과 비교 (엔진 권장 vs Mix 유지 vs Manual)")

        def feasibility_badge(p_end: float) -> str:
            ok = bool(p_end >= float(target_premium))
            label = "달성 가능" if ok else "달성 어려움"
            bg = "#1f7a3a" if ok else "#a15c00"
            return f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;color:white;background:{bg};'>{label}</span>"

        sA, sB, sC = st.columns(3)
        for col, name, sim in [
            (sA, "A) 엔진 권장", sim_A),
            (sB, "B) 현재 Mix 유지", sim_B),
            (sC, "C) Manual", sim_C),
        ]:
            p_end = float(sim["month_end"]["premium"])
            gap = float(target_premium - p_end)
            col.markdown(
                f"""<div class='card'>
                    <div class='kpi'>{money(p_end)}</div>
                    <div class='kpi_sub'>월말 Premium(예측) · {feasibility_badge(p_end)}</div>
                    <div class='smallcap'>목표 대비 Gap: {money(gap)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            col.markdown(
                f"""<div class='card'>
                    <div class='kpi'>{sim['month_end']['leads']:,.0f}</div>
                    <div class='kpi_sub'>월말 Leads(예측)</div>
                    <div class='smallcap'>Contracts: {sim['month_end']['contracts']:,.0f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # 공통 KPI(예산)
        e1, e2, e3 = st.columns(3)
        e1.markdown(f"<div class='card'><div class='kpi'>{money(monthly_budget)}</div><div class='kpi_sub'>이번달 총예산(가정)</div></div>", unsafe_allow_html=True)
        e2.markdown(f"<div class='card'><div class='kpi'>{format_won(mtd_spend)}</div><div class='kpi_sub'>MTD 사용예산</div></div>", unsafe_allow_html=True)
        e3.markdown(f"<div class='card'><div class='kpi'>{format_won(remaining_budget)}</div><div class='kpi_sub'>남은 예산</div></div>", unsafe_allow_html=True)

        # --------------------------------------------------------
        # 상세(채널별) 테이블: 선택 시나리오별로 표시
        # --------------------------------------------------------
        if scenario_view.startswith("상세"):
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            tabA, tabB, tabC = st.tabs(["A) 엔진 권장", "B) Mix 유지", "C) Manual"])
            for tab, sim in [(tabA, sim_A), (tabB, sim_B), (tabC, sim_C)]:
                with tab:
                    plan = sim["plan"].copy()
                    show = plan.copy()
                    show["추가 배정(남은예산)"] = show["추가 배정(남은예산)"].apply(money)
                    show["추가 Leads(예측)"] = show["추가 Leads(예측)"].apply(lambda x: f"{x:,.1f}")
                    show["추가 Contracts(예측)"] = show["추가 Contracts(예측)"].apply(lambda x: f"{x:,.1f}")
                    show["추가 Premium(예측)"] = show["추가 Premium(예측)"].apply(money)
                    st.dataframe(show, use_container_width=True, height=320, hide_index=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    elif days_remaining == 0:
        st.info("월말(또는 이후)이라 남은 예산 배분/예측을 계산할 수 없음.")
    else:
        st.info("남은 예산이 0이거나, MTD 예산이 입력 예산을 초과함.")

    # apply constraints (per-channel)
    cur = {r["채널"]: float(r["현재"]) for _, r in edited.iterrows()}
    cons = {
        str(r["채널"]): ChannelConstraint(
            lock=bool(r.get("lock", False)),
            min_ratio=float(r.get("min_ratio", default_min)),
            max_ratio=float(r.get("max_ratio", default_max)),
        )
        for _, r in edited.iterrows()
    }

    step = float(step_budget)

    # ------------------------------
    # Objective (primary): Min spend to reach fixed Premium target
    # If target is 0 or not set, fallback to full-budget maximization (legacy behavior)
    # ------------------------------
    use_target_mode = bool(premium_target and float(premium_target) > 0)

    if use_target_mode:
        res = optimize_min_spend_for_target(
            channels=cur.keys(),
            current_spend=cur,
            constraints=cons,
            pred_leads=pred_leads,
            pred_contracts=pred_contracts,
            pred_premium=pred_premium,
            premium_target=float(premium_target),
            step=step,
            budget_cap=float(total_budget),
        )
        alloc = res.recommended_spend
        remaining = float(res.unspent_budget)
        objective = "MIN_SPEND_FOR_TARGET_PREMIUM"
        ycol = "premium"
    else:
        # Fallback: maximize selected objective with full fixed budget
        lock = {k: cons[k].lock for k in cur.keys()}
        lo = {k: float(cur[k]) * float(cons[k].min_ratio) for k in cur.keys()}
        hi = {k: float(cur[k]) * float(cons[k].max_ratio) for k in cur.keys()}

        alloc = {}
        for ch in cur.keys():
            if lock.get(ch, False):
                alloc[ch] = cur[ch]
            else:
                alloc[ch] = max(lo[ch], min(cur[ch], hi[ch]))

        remaining = float(total_budget - sum(alloc.values()))
        if remaining < -1e-6:
            st.error("제약 조건(특히 Lock/Min)이 강해서 총예산(고정)보다 예산 합이 큼. Min/Lock을 완화 필요.")
            return

        max_iter = 5000
        it = 0
        while remaining >= step - 1e-9 and it < max_iter:
            best_ch = None
            best_gain = -1e18
            for ch in alloc.keys():
                if lock.get(ch, False):
                    continue
                s_now = alloc[ch]
                if s_now + step > hi[ch] + 1e-9:
                    continue
                gain = pred_objective(ch, s_now + step) - pred_objective(ch, s_now)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_ch = ch
            if best_ch is None:
                break
            alloc[best_ch] += step
            remaining -= step
            it += 1

        # distribute leftover by headroom
        if remaining > 1e-6:
            head = {ch: max(0.0, hi[ch] - alloc[ch]) for ch in alloc.keys()}
            tot_head = float(sum(head.values()))
            if tot_head > 1e-9:
                for ch in alloc.keys():
                    add = remaining * (head[ch] / tot_head)
                    alloc[ch] = min(hi[ch], alloc[ch] + add)

    # output
    rows = []
    tot_pred = 0.0
    tot_leads = 0.0
    tot_contracts = 0.0
    tot_premium = 0.0
    for ch in sorted(alloc.keys()):
        s = float(alloc[ch])
        v = pred_objective(ch, s)
        l = pred_leads(ch, s)
        c = pred_contracts(ch, s)
        p = pred_premium(ch, s)
        tot_pred += v
        tot_leads += l
        tot_contracts += c
        tot_premium += p
        rows.append({
            "채널": ch,
            "현재": cur.get(ch, 0.0),
            "권장(최적화)": s,
            "Δ": s - cur.get(ch, 0.0),
            "예측 Leads": l,
            "예측 Contracts": c,
            "예측 Premium": p,
            f"예측 {ycol}": v,
            "Lock": bool(cons.get(ch, ChannelConstraint()).lock),
            "Min": float(cur.get(ch, 0.0)) * float(cons.get(ch, ChannelConstraint()).min_ratio),
            "Max": float(cur.get(ch, 0.0)) * float(cons.get(ch, ChannelConstraint()).max_ratio),
        })
    out_df = pd.DataFrame(rows)

    # store simulation for executive summary
    _existing_month_planner = None
    _existing = st.session_state.get("opt_simulation")
    if isinstance(_existing, dict):
        _existing_month_planner = _existing.get("month_planner")
    st.session_state["opt_simulation"] = {
        "objective": objective,
        "ycol": ycol,
        "tot_pred": float(tot_pred),
        "total_budget": float(total_budget),
        "unspent_budget": float(max(0.0, remaining)),
        "totals": {
            "budget": float(out_df["권장(최적화)"].sum()),
            "leads": float(tot_leads),
            "contracts": float(tot_contracts),
            "premium": float(tot_premium),
        },
        "premium_target": float(premium_target or 0.0),
        "by_channel": out_df[["채널","권장(최적화)","Δ","예측 Leads","예측 Contracts","예측 Premium"]].to_dict(orient="records"),
        "assumptions": {
            "rr_by_channel": rr_map,
            "premium_per_sale_by_channel": pps_map,
            "rr_global": float(g_rr),
            "premium_per_sale_global": float(g_pps),
        },
    }
    if _existing_month_planner is not None:
        try:
            st.session_state["opt_simulation"]["month_planner"] = _existing_month_planner
        except Exception:
            pass

    # ------------------------------------------------------------
    # (NEW) Persist plan for Plan vs Actual tracking
    # ------------------------------------------------------------
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### Save Plan Run (플랜 저장)")
    st.markdown(
        "<div class='note'>최적화 결과(권장 예산 + 예측치)를 파일로 저장해두면, 이후 실제 성과(Actuals)와 비교(Tracking) 가능.</div>",
        unsafe_allow_html=True,
    )
    s1, s2, s3 = st.columns([1, 1, 2])
    with s1:
        month_key = "optimizer_plan_month"
        default_month = month_start.strftime("%Y-%m")
        plan_month = st.text_input("저장 월(YYYY-MM)", value=default_month, key=month_key)
        plan_month = (plan_month or default_month).strip()[:7]
    with s2:
        root_dir = st.text_input("저장 경로", value="out/runs", key="optimizer_plan_root")
        root_dir = (root_dir or "out/runs").strip()
    with s3:
        if st.button("✅ 플랜 저장", use_container_width=True):
            try:
                paths = save_plan_run(
                    opt_simulation=st.session_state.get("opt_simulation", {}),
                    month=plan_month,
                    history_start=str(history_start),
                    history_end=str(history_end),
                    forecast_start=str(forecast_start),
                    forecast_end=str(forecast_end),
                    root_dir=root_dir,
                )
                st.success(f"저장 완료: {paths.run_dir}")
                st.session_state["last_plan_run_id"] = paths.run_id
            except Exception as e:
                st.error(f"저장 실패: {e}")


    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"<div class='card'><div class='smallcap'>총예산(상한)</div><div class='kpi'>{format_won(float(total_budget))}</div></div>", unsafe_allow_html=True)
    sub2 = "목표 달성 최소집행" if (premium_target and float(premium_target)>0) else "제약 반영"
    k2.markdown(
        f"<div class='card'><div class='smallcap'>권장 예산(합)</div><div class='kpi'>{format_won(float(out_df['권장(최적화)'].sum()))}</div><div class='kpi_sub'>{sub2}</div></div>",
        unsafe_allow_html=True,
    )
    label = "예측 Premium" if ycol=="premium" else f"예측 {ycol}"
    k3.markdown(f"<div class='card'><div class='smallcap'>{label}</div><div class='kpi'>{format_won(float(tot_pred))}</div><div class='kpi_sub'>Impact 기간</div></div>", unsafe_allow_html=True)
    if premium_target and float(premium_target) > 0:
        gap = float(tot_premium) - float(premium_target)
        sub = "Target 달성" if gap >= 0 else "Target 미달"
        k4.markdown(f"<div class='card'><div class='smallcap'>Target 대비</div><div class='kpi'>{format_won(gap)}</div><div class='kpi_sub'>{sub}</div></div>", unsafe_allow_html=True)
        if remaining > 1e-6:
            st.info(f"✅ Target을 만족하는 최소 집행으로 계산됨. 남는 예산(미사용): {format_won(float(remaining))}")
    else:
        k4.markdown("<div class='card'><div class='smallcap'>Target 대비</div><div class='kpi'>-</div><div class='kpi_sub'>미설정</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # display totals row (tables only)
    disp_df = out_df.copy()
    tot_row = {
        "채널": "합계",
        "현재": float(disp_df["현재"].sum()),
        "권장(최적화)": float(disp_df["권장(최적화)"].sum()),
        "Δ": float(disp_df["Δ"].sum()),
        "예측 Leads": float(disp_df["예측 Leads"].sum()),
        "예측 Contracts": float(disp_df["예측 Contracts"].sum()),
        "예측 Premium": float(disp_df["예측 Premium"].sum()),
        f"예측 {ycol}": float(disp_df.get(f"예측 {ycol}", pd.Series([0.0])).sum()),
        "Lock": False,
        "Min": np.nan,
        "Max": np.nan,
    }
    disp_df = pd.concat([disp_df, pd.DataFrame([tot_row])], ignore_index=True)

    show = disp_df.copy()
    for col in ["현재","권장(최적화)","Δ","Min","Max", f"예측 {ycol}","예측 Leads","예측 Contracts","예측 Premium"]:
        show[col] = show[col].apply(money)
    st.dataframe(
        show[["채널","현재","권장(최적화)","Δ","예측 Leads","예측 Contracts","예측 Premium", f"예측 {ycol}","Lock","Min","Max"]],
        use_container_width=True,
        height=360,
        hide_index=True,
    )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Chart: side-by-side (현재 vs 권장) + Δ label
    # - Optimizer 결과가 없으면 미출력
    # - 데이터가 0이면 안내문
    # ------------------------------------------------------------
    st.markdown("### Budget by Channel (현재 vs 권장)")
    if out_df is None or out_df.empty:
        st.info("최적화 결과가 없어 차트를 표시할 수 없음.")
    else:
        dplot = out_df[["채널", "현재", "권장(최적화)"]].copy()
        dplot["현재"] = pd.to_numeric(dplot["현재"], errors="coerce").fillna(0.0)
        dplot["권장(최적화)"] = pd.to_numeric(dplot["권장(최적화)"], errors="coerce").fillna(0.0)
        dplot["Δ"] = dplot["권장(최적화)"] - dplot["현재"]

        # 정렬: 권장 예산 기준(작은→큰)
        dplot = dplot.sort_values("권장(최적화)")

        if float(dplot["권장(최적화)"].sum()) <= 0 and float(dplot["현재"].sum()) <= 0:
            st.info("예산 데이터가 0으로만 구성되어 차트를 표시할 수 없음.")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))

            # ---- Unit scaling (원 vs 백만원) ----
            max_raw = float(max(dplot["현재"].max(), dplot["권장(최적화)"].max(), 1.0))
            if max_raw >= 1e8:
                scale = 1e6
                unit = "백만원"
            else:
                scale = 1.0
                unit = "원"

            dplot_s = dplot.copy()
            dplot_s["현재_s"] = dplot_s["현재"] / scale
            dplot_s["권장_s"] = dplot_s["권장(최적화)"] / scale
            dplot_s["Δ_s"] = dplot_s["Δ"] / scale

            y = np.arange(len(dplot_s))
            h = 0.38

            ax.barh(y - h/2, dplot_s["현재_s"], height=h, label="Current")
            ax.barh(y + h/2, dplot_s["권장_s"], height=h, label="Recommended")

            ax.set_yticks(y)
            ax.set_yticklabels(dplot_s["채널"].tolist())
            ax.set_xlabel(f"Budget ({unit})")
            ax.set_ylabel("Channel")
            ax.legend(loc="lower right")

            # x축: 과학적 표기 제거 + 콤마
            ax.ticklabel_format(style="plain", axis="x")
            try:
                from matplotlib.ticker import StrMethodFormatter
                ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            except Exception:
                pass

            # Δ 라벨 (각 채널 우측, 원 단위 콤마)
            maxv_s = float(max(dplot_s["현재_s"].max(), dplot_s["권장_s"].max(), 1.0))
            ax.set_xlim(0, maxv_s * 1.35)
            for i, row in enumerate(dplot_s.itertuples(index=False, name=None)):
                # row: (채널, 현재, 권장(최적화), Δ, 현재_s, 권장_s, Δ_s)
                cur_s = float(row[4])
                rec_s = float(row[5])
                delta_raw = float(row[3])
                anchor = float(max(cur_s, rec_s))
                ax.text(anchor + maxv_s*0.03, i, f"Δ {money(delta_raw)}", va="center", fontsize=9)

            st.pyplot(fig, clear_figure=True)

    st.markdown("<div class='note'>* 본 최적화는 ‘윈도우 내 일별 데이터’로 간단한 포화(체감) 곡선을 근사한 휴리스틱임.</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Base Output (원본 산출물)
    # ------------------------------------------------------------
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### Base Output (원본 산출물)")
    if bud is None:
        st.info("out/budget_recommendation.csv가 없어 원본 권장 예산안을 표시할 수 없음.")
    else:
        b = bud.copy()
        b["현재"] = pd.to_numeric(b["current_spend"], errors="coerce").fillna(0.0).apply(money)
        b["권장(원본)"] = pd.to_numeric(b["recommended_spend"], errors="coerce").fillna(0.0).apply(money)
        b["Δ"] = pd.to_numeric(b["delta"], errors="coerce").fillna(0.0).apply(money)
        bdisp = b[["channel","현재","권장(원본)","Δ"]].rename(columns={"channel":"채널"}).copy()
        # 합계 행
        try:
            cur_sum = pd.to_numeric(b["current_spend"], errors="coerce").fillna(0.0).sum()
            rec_sum = pd.to_numeric(b["recommended_spend"], errors="coerce").fillna(0.0).sum()
            delta_sum = pd.to_numeric(b["delta"], errors="coerce").fillna(0.0).sum()
            bdisp = pd.concat([bdisp, pd.DataFrame([{"채널":"합계","현재":money(cur_sum),"권장(원본)":money(rec_sum),"Δ":money(delta_sum)}])], ignore_index=True)
        except Exception:
            pass
        st.dataframe(bdisp, use_container_width=True, height=300, hide_index=True)