from __future__ import annotations

"""Decision summary section renderer.

This component renders an executive-friendly summary for the latest decision:
- Remaining budget and plan totals (Human vs AI; fallback to Do Nothing)
- Outcome KPIs (RA Premium, uplift, win probability)
- Funnel forecast vs actual (AI plan) in a compact table
- Allocation changes (Top-N) in a small table (index hidden)

Design principles:
- Narrative first (sentence + badges), tables are supporting.
- No raw JSON rendering.
- Hide dataframe index to avoid meaningless 0/1/2 columns.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st

from .ui import fmt_money, fmt_percent, section, style_table, kpi_row
from .tables import add_totals


PlanKey = Literal["ai", "do_nothing", "human"]
RenderMode = Literal["full", "plan_compare_only"]


def _sum_budget(b: dict[str, float]) -> float:
    return float(sum(float(v) for v in b.values()))


def _infer_compare_key(vm_plans: dict[str, object]) -> PlanKey:
    return "human" if "human" in vm_plans else "do_nothing"


def _fmt_count(v: float | int | None) -> str:
    if v is None or pd.isna(v):
        return "-"
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return "-"


def _baseline_premium_per_contract(mart: pd.DataFrame | None) -> float | None:
    if mart is None or mart.empty:
        return None
    try:
        prem = float(pd.to_numeric(mart["premium"], errors="coerce").sum())
        ctr = float(pd.to_numeric(mart["contracts"], errors="coerce").sum())
        if ctr <= 0:
            return None
        return prem / ctr
    except Exception:
        return None


def _build_plan_overall_table(*, vm, base_key: PlanKey, base_label: str, mart: pd.DataFrame | None) -> pd.DataFrame:
    """Overall plan compare table (Base vs AI)."""
    ppc = _baseline_premium_per_contract(mart)

    def _row(label: str, key: PlanKey) -> dict[str, object]:
        p = vm.plans[key]
        total_budget = _sum_budget(p.budget_by_channel)
        exp_premium = float(p.expected_premium)
        exp_contracts = (exp_premium / ppc) if (ppc is not None and ppc > 0 and exp_premium > 0) else None
        return {
            "구분": label,
            "총예산": total_budget,
            "예상 Premium": exp_premium,
            "예상 Contracts": exp_contracts,
            "계약당 Premium": ppc,
        }

    df = pd.DataFrame([_row(base_label, base_key), _row("AI", "ai")])
    if df.empty:
        return df

    out = df.copy()
    out["총예산"] = out["총예산"].apply(lambda x: fmt_money(float(x)) if x is not None else "-")
    out["예상 Premium"] = out["예상 Premium"].apply(lambda x: fmt_money(float(x)) if x is not None else "-")
    out["예상 Contracts"] = out["예상 Contracts"].apply(_fmt_count)
    out["계약당 Premium"] = out["계약당 Premium"].apply(lambda x: fmt_money(float(x)) if x is not None else "-")
    return out


def _build_channel_plan_vertical_table(*, vm, base_key: PlanKey, base_label: str, mart: pd.DataFrame | None) -> pd.DataFrame:
    """Channel x (Base, AI) vertical comparison table.

    Option A: Δ is shown inline on the AI row only.
    """
    ppc = _baseline_premium_per_contract(mart)
    base = vm.plans[base_key]
    ai = vm.plans["ai"]

    base_budget = base.budget_by_channel
    ai_budget = ai.budget_by_channel
    channels = sorted(set(base_budget.keys()) | set(ai_budget.keys()))

    base_total = _sum_budget(base_budget)
    ai_total = _sum_budget(ai_budget)

    base_exp_premium = float(base.expected_premium)
    ai_exp_premium = float(ai.expected_premium)
    base_exp_contracts = (base_exp_premium / ppc) if (ppc is not None and ppc > 0 and base_exp_premium > 0) else None
    ai_exp_contracts = (ai_exp_premium / ppc) if (ppc is not None and ppc > 0 and ai_exp_premium > 0) else None

    rows: list[dict[str, object]] = []
    for ch in channels:
        b = float(base_budget.get(ch, 0.0))
        a = float(ai_budget.get(ch, 0.0))

        bw = (b / base_total) if base_total > 0 else 0.0
        aw = (a / ai_total) if ai_total > 0 else 0.0

        b_prem = (base_exp_premium * bw) if base_exp_premium > 0 else None
        a_prem = (ai_exp_premium * aw) if ai_exp_premium > 0 else None
        b_ctr = (base_exp_contracts * bw) if (base_exp_contracts is not None) else None
        a_ctr = (ai_exp_contracts * aw) if (ai_exp_contracts is not None) else None

        rows.append({
            "채널": ch,
            "구분": base_label,
            "예산": fmt_money(b),
            "예상 Premium": fmt_money(b_prem) if b_prem is not None else "-",
            "예상 Contracts": _fmt_count(b_ctr),
        })

        delta_b = a - b
        delta_p = (a_prem - b_prem) if (a_prem is not None and b_prem is not None) else None
        delta_c = (a_ctr - b_ctr) if (a_ctr is not None and b_ctr is not None) else None

        budget_cell = f"{fmt_money(a)} ({'+' if delta_b >= 0 else ''}{fmt_money(delta_b)})"

        if delta_p is None:
            prem_cell = fmt_money(a_prem) if a_prem is not None else "-"
        else:
            prem_cell = f"{fmt_money(a_prem)} ({'+' if delta_p >= 0 else ''}{fmt_money(delta_p)})"

        if delta_c is None:
            ctr_cell = _fmt_count(a_ctr)
        else:
            sign = "+" if float(delta_c) >= 0 else ""
            ctr_cell = f"{_fmt_count(a_ctr)} ({sign}{_fmt_count(delta_c)})"

        rows.append({
            "채널": ch,
            "구분": "AI",
            "예산": budget_cell,
            "예상 Premium": prem_cell,
            "예상 Contracts": ctr_cell,
        })

    return pd.DataFrame(rows)


def _build_allocation_changes_table(
    *,
    ai_budget: dict[str, float],
    base_budget: dict[str, float],
    base_label: str,
    top_n: int = 5,
) -> pd.DataFrame:
    channels = sorted(set(ai_budget.keys()) | set(base_budget.keys()))
    rows: list[dict[str, object]] = []
    for ch in channels:
        b = float(base_budget.get(ch, 0.0))
        a = float(ai_budget.get(ch, 0.0))
        rows.append(
            {
                "채널": ch,
                f"{base_label} 예산": b,
                "AI 추천 예산": a,
                "Δ 예산(AI-Base)": a - b,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("Δ 예산(AI-Base)", ascending=False, key=lambda s: s.abs())
    df = df.head(int(top_n))
    df = add_totals(df, numeric_cols=[f"{base_label} 예산", "AI 추천 예산", "Δ 예산(AI-Base)"])
    return df


def _render_header(
    *,
    remaining_budget: float,
    human_total: float | None,
    base_label: str,
    base_total: float,
    ai_total: float,
    period_start: str,
    period_end: str,
    rollout_mode: str,
    decision_id: str,
) -> None:
    # Narrative line (no table)
    left = (
        f"남은 예산(이번 배분 가능): {fmt_money(remaining_budget)}  |  "
        f"{base_label} 배분: {fmt_money(base_total)}  |  "
        f"AI 배분: {fmt_money(ai_total)}"
    )
    st.markdown(left)
    cols = st.columns([1, 1, 2])
    with cols[0]:
        st.caption(f"기간: {period_start} ~ {period_end}")
    with cols[1]:
        st.caption(f"Rollout: {rollout_mode}")
    with cols[2]:
        st.caption(f"Decision ID: {decision_id}")


def _render_outcome_kpis(
    *,
    ai_ra: float,
    base_ra: float,
    ai_exp: float,
    base_exp: float,
    ai_q10: float | None,
    base_q10: float | None,
    p_win: float,
    base_label: str,
) -> None:
    delta = ai_ra - base_ra
    uplift = (delta / base_ra) if base_ra > 0 else 0.0
    kpi_row(
        [
            {"label": "AI Risk-Adjusted Premium (예측)", "value": fmt_money(ai_ra), "sub": "추천 플랜"},
            {"label": f"{base_label} Risk-Adjusted Premium (예측)", "value": fmt_money(base_ra), "sub": f"{base_label} 플랜"},
            {"label": "ΔRA (AI-Base)", "value": fmt_money(delta), "sub": f"uplift {fmt_percent(uplift, 1)}"},
            {"label": f"P(AI > {base_label})", "value": fmt_percent(p_win, 1), "sub": "posterior predictive"},
        ]
    )

    # Clarify that RA != Expected Premium by design (risk penalty).
    kpi_row(
        [
            {"label": "AI Expected Premium (E)", "value": fmt_money(ai_exp), "sub": "평균 예측"},
            {"label": f"{base_label} Expected Premium (E)", "value": fmt_money(base_exp), "sub": "평균 예측"},
            {"label": "AI Downside Premium (q10)", "value": fmt_money(ai_q10), "sub": "보수적(10%)"},
            {"label": f"{base_label} Downside Premium (q10)", "value": fmt_money(base_q10), "sub": "보수적(10%)"},
        ]
    )


def _render_funnel_forecast_table(forecast_path: Path) -> None:
    if not forecast_path.exists():
        st.info("퍼널 예측 리포트가 없습니다. (recommend 실행 후 artifacts/recommendations/funnel_forecast 확인)")
        return

    fdf = pd.read_csv(forecast_path)
    if fdf.empty:
        st.info("퍼널 예측 데이터가 비어 있습니다.")
        return

    stage_map = {
        "leads": "리드(Leads)",
        "call_attempt": "통화시도(Call Attempt)",
        "call_connected": "연결(Call Connected)",
        "contracts": "계약(Contracts)",
        "premium": "프리미엄(Premium)",
    }
    fdf["퍼널 단계"] = fdf["stage"].map(stage_map).fillna(fdf["stage"])
    view = fdf[["퍼널 단계", "expected", "actual", "gap"]].copy()
    view.rename(
        columns={
            "expected": "AI 예상(E)",
            "actual": "실제(A)",
            "gap": "Δ(A-E)",
        },
        inplace=True,
    )

    # Formatting: premium is money, others are counts
    money_cols = ["AI 예상(E)", "실제(A)", "Δ(A-E)"] if ("프리미엄" in " ".join(view["퍼널 단계"].astype(str).tolist())) else []
    # We'll apply mixed formatting manually
    styled = view.copy()
    for col in ["AI 예상(E)", "실제(A)", "Δ(A-E)"]:
        def _fmt(v, stage):
            if pd.isna(v):
                return "-"
            if "프리미엄" in stage:
                return fmt_money(float(v))
            return f"{int(round(float(v))):,}"
        styled[col] = [
            _fmt(v, s) for v, s in zip(styled[col].tolist(), styled["퍼널 단계"].tolist())
        ]

    st.dataframe(styled, use_container_width=True, hide_index=True, height=240)


def render_decision_summary(
    *,
    vm,
    artifacts_root: Path,
    mart: pd.DataFrame | None = None,
    base_label_if_human: str = "Human",
    base_label_if_dn: str = "현상유지(Do Nothing)",
    mode: RenderMode = "full",
) -> None:
    """Render plan comparison and latest decision summary.

    Render modes:
    - mode="full": KPI summary only (used in main narrative). Detailed tables live in Drill-down.
    - mode="plan_compare_only": detailed tables only (used in Drill-down tab).
    """
    compare_key = _infer_compare_key(vm.plans)
    base_label = base_label_if_human if compare_key == "human" else base_label_if_dn

    ai_budget = vm.plans["ai"].budget_by_channel
    base_budget = vm.plans[compare_key].budget_by_channel
    remaining_budget = _sum_budget(ai_budget)
    base_total = _sum_budget(base_budget)

    if mode == "full":
        section("최신 의사결정 요약", "전체 → 채널별 순으로 예산/성과를 요약합니다.")

        _render_header(
            remaining_budget=remaining_budget,
            human_total=None,
            base_label=base_label,
            base_total=base_total,
            ai_total=remaining_budget,
            period_start=str(vm.period_start),
            period_end=str(vm.period_end),
            rollout_mode=str(vm.rollout_mode),
            decision_id=str(vm.decision_id),
        )

        st.divider()

        _render_outcome_kpis(
            ai_ra=float(vm.plans["ai"].ra_premium),
            base_ra=float(vm.plans[compare_key].ra_premium),
            ai_exp=float(vm.plans["ai"].expected_premium),
            base_exp=float(vm.plans[compare_key].expected_premium),
            ai_q10=vm.plans["ai"].q10_premium,
            base_q10=vm.plans[compare_key].q10_premium,
            p_win=float(vm.plans["ai"].p_win) if compare_key == "do_nothing" else float(vm.plans["human"].p_win),
            base_label=base_label,
        )

        # NOTE: Detailed plan comparison tables are intentionally rendered only in Drill-down.
        # This keeps the main page compact and avoids duplicated information.
        return

    # mode == "plan_compare_only"
    section("플랜 비교", "AI vs Base(또는 Do Nothing) — 상세 테이블")

    # 1) Overall (전체)
    section("플랜 비교(전체)", "총예산 → 예상 성과")
    overall = _build_plan_overall_table(vm=vm, base_key=compare_key, base_label=base_label, mart=mart)
    if overall.empty:
        st.info("플랜 비교 데이터를 만들 수 없습니다.")
        return
    st.dataframe(style_table(overall), use_container_width=True, hide_index=True)

    st.divider()

    # 2) By channel (채널별) — Option A
    section("플랜 비교(채널별)", "Option A: Δ는 AI 행에만 표시")
    ch = _build_channel_plan_vertical_table(vm=vm, base_key=compare_key, base_label=base_label, mart=mart)
    if ch.empty:
        st.info("채널별 플랜 비교 데이터를 만들 수 없습니다.")
        return
    st.dataframe(style_table(ch), use_container_width=True, hide_index=True, height=420)
