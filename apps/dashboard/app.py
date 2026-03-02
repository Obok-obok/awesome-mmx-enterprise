"""MMx Control Tower — One-page Executive Dashboard.

단일 페이지(Control Tower)로 재구성된 경영진/운영자용 대시보드입니다.

구성:
- Snapshot KPIs (Spend / Premium / ROI / Target attainment / Risk)
- Funnel Health (전환율 & 병목)
- AI Decision (Human vs AI, 없으면 Do Nothing) + 퍼널 예측 변화
- Alerts & Actions (기회/포화/다운사이드 리스크)
- 상세(전체 테이블/아티팩트)는 Expander로 제공

Design reference: card-first executive dashboards (Ads Manager / CEO overview boards).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math

import pandas as pd
import streamlit as st

from components.bootstrap import ensure_project_sys_path, bootstrap
from components.ui import (
    badge,
    badge_html,
    fmt_count,
    fmt_money,
    fmt_percent,
    fmt_ratio,
    kpi_row,
    section,
)
from components.decision_viewmodel import build_decision_viewmodel, latest_decision_path
from components.decision_summary import render_decision_summary
from components.insights import render_insight_card
from components.backtest_view import load_backtest_vm, render_backtest_section



def _to_float(v: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float for dashboard metrics."""
    try:
        if v is None:
            return default
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default

def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def _load_targets(paths: Any) -> pd.DataFrame:
    """Load monthly targets (if uploaded) from curated data directory."""
    curated = Path(paths.data_curated)
    # NOTE:
    # demo_run_all.sh generates sample targets at:
    #   data/curated/targets/monthly_targets.csv
    # Older dashboard versions looked for data/curated/targets_monthly.csv.
    # We support both (and a couple of legacy names) to avoid brittle runtime.
    candidates = [
        curated / "targets_monthly.csv",
        curated / "monthly_targets.csv",
        curated / "targets" / "monthly_targets.csv",
        curated / "targets" / "targets_monthly.csv",
    ]
    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        return pd.DataFrame(columns=["month", "target_premium"])
    df = pd.read_csv(p)
    if df.empty:
        return df

    # Accept a few common business-friendly column names.
    month_col = "month" if "month" in df.columns else ("월" if "월" in df.columns else None)
    target_col = None
    for c in [
        "target_premium",
        "premium_target",
        "premium_goal",
        "target",
        "Premium 목표",
        "프리미엄 목표",
        "목표 Premium",
        "목표_프리미엄",
    ]:
        if c in df.columns:
            target_col = c
            break

    if month_col is None or target_col is None:
        # Return empty but with canonical schema.
        return pd.DataFrame(columns=["month", "target_premium"])

    df = df[[month_col, target_col]].copy()
    df.columns = ["month", "target_premium"]
    # Robust normalization: accept YYYY-MM, YYYY-MM-DD, YYYYMM, YYYY/MM
    s = df["month"].astype(str).str.strip().str.replace("/", "-", regex=False)
    extracted = s.str.extract(r"(\d{4}-\d{2})", expand=False)
    yyyymm = s.str.extract(r"(\d{4})(\d{2})", expand=True)
    fallback = yyyymm[0].fillna("") + "-" + yyyymm[1].fillna("")
    df["month"] = extracted.fillna(fallback).str.slice(0, 7)
    df["target_premium"] = pd.to_numeric(df["target_premium"], errors="coerce").fillna(0.0)
    return df


def _parse_money(v: Any, default: float = 0.0) -> float:
    """Parse numbers that may come as strings with commas/whitespace."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).strip().replace(",", "")
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def _latest_month_from_mart(mart: pd.DataFrame) -> str | None:
    if mart.empty:
        return None
    d = pd.to_datetime(mart["date"]).max()
    return d.strftime("%Y-%m")


def _normalize_month_to_period(value: object) -> pd.Period | None:
    """Normalize various month formats into Period('M')."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Accept: YYYY-MM, YYYY-MM-DD, YYYY/MM, YYYYMM
    s = s.replace("/", "-")
    if len(s) == 6 and s.isdigit():
        s = f"{s[:4]}-{s[4:]}"
    s = s[:7]
    dt = pd.to_datetime(s + "-01", errors="coerce")
    if pd.isna(dt):
        return None
    return dt.to_period("M")


def _compute_target_attainment(*, mart: pd.DataFrame, targets: pd.DataFrame) -> tuple[float | None, float | None, str | None]:
    """Return (attainment_ratio, target_sum, month_range_label) for current coverage.

    Premium KPI is cumulative over the selected mart.
    Target attainment must therefore be computed cumulatively over the same
    month range (avoid MTD vs cumulative mismatches).
    """

    if mart.empty or targets.empty:
        return None, None, None

    mdf = mart.copy()
    mdf["_p"] = pd.to_datetime(mdf["date"], errors="coerce").dt.to_period("M")
    mdf = mdf.dropna(subset=["_p"])
    if mdf.empty:
        return None, None, None

    start_p = mdf["_p"].min()
    end_p = mdf["_p"].max()

    tdf = targets.copy()
    tdf["_p"] = tdf["month"].apply(_normalize_month_to_period)
    tdf = tdf.dropna(subset=["_p"])
    if tdf.empty:
        label = f"{start_p}~{end_p}" if start_p != end_p else str(end_p)
        return None, None, label

    mask = (tdf["_p"] >= start_p) & (tdf["_p"] <= end_p)
    t_sum = float(tdf.loc[mask, "target_premium"].sum())
    label = f"{start_p}~{end_p}" if start_p != end_p else str(end_p)
    if t_sum <= 0:
        return None, None, label

    actual_sum = float(mdf["premium"].sum())
    return _safe_div(actual_sum, t_sum), t_sum, label


def _render_topbar(*, ctx: Any, vm: Any | None, actual_only: bool) -> None:
    """Top action bar (filters summary + export buttons)."""
    c1, c2, c3, c4 = st.columns([2.2, 1.4, 1.2, 1.2])

    with c1:
        st.markdown("### MMx Control Tower")
        st.caption(f"데이터 커버리지: {ctx.coverage}")

    with c2:
        # Policies are displayed in the context bar as well; keep a compact badge set here.
        st.markdown(" ")
        badges = [
            badge(f"Objective: {getattr(ctx.settings, 'objective_mode', 'RISK_ADJUSTED')}", kind="info"),
            badge(f"λ={getattr(ctx.settings, 'policy_lambda', 0.0)}", kind="muted"),
            badge(f"δ={getattr(ctx.settings, 'policy_delta', 0.0)}", kind="muted"),
        ]
        badges = [b for b in badges if isinstance(b, str) and b.strip()]
        if actual_only:
            badges.insert(0, badge("Actual-only", kind="warning"))
        st.markdown(" ".join([b for b in badges if isinstance(b, str)]), unsafe_allow_html=True)

    # Exports
    with c3:
        st.markdown(" ")
        dec_path = latest_decision_path(Path(ctx.paths.artifacts) / "recommendations" / "decisions")
        if dec_path and dec_path.exists():
            st.download_button(
                "결정 아티팩트(JSON) 다운로드",
                data=dec_path.read_bytes(),
                file_name=dec_path.name,
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.button("결정 아티팩트 없음", disabled=True, use_container_width=True)

    with c4:
        st.markdown(" ")
        # Export AI vs baseline budget table if available
        try:
            if vm is None:
                raise RuntimeError("no decision")
            tbl = vm.budget_table  # created by ViewModel
            st.download_button(
                "추천 예산표(CSV) 다운로드",
                data=tbl.to_csv(index=False).encode("utf-8"),
                file_name="budget_recommendation.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception:
            st.button("예산표 없음", disabled=True, use_container_width=True)


def _render_artifacts_health(*, artifacts_root: Path, mart_df: pd.DataFrame, decision_path: Path | None) -> None:
    """Show a small health panel for key artifacts (mart / decision / backtest).

    운영자가 '왜 화면이 비어있는지'를 즉시 이해할 수 있도록,
    존재 여부/경로/마지막 수정시간을 간단히 요약합니다.
    """

    def _mtime(p: Path) -> str:
        try:
            ts = p.stat().st_mtime
            return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "-"

    mart_status = "OK" if (not mart_df.empty) else "MISSING"
    dec_status = "OK" if (decision_path is not None and decision_path.exists()) else "MISSING"
    bt_latest = artifacts_root / "backtests" / "latest"
    bt_status = "OK" if (bt_latest / "metrics_overall.json").exists() else "MISSING"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(badge(f"Mart: {mart_status}", kind="success" if mart_status == "OK" else "warning"), unsafe_allow_html=True)
        st.caption(f"rows={len(mart_df)}")
    with col2:
        st.markdown(badge(f"Decision: {dec_status}", kind="success" if dec_status == "OK" else "warning"), unsafe_allow_html=True)
        st.caption(f"{decision_path if decision_path else (artifacts_root / 'recommendations' / 'decisions')}")
        if decision_path is not None and decision_path.exists():
            st.caption(f"modified: {_mtime(decision_path)}")
    with col3:
        st.markdown(badge(f"Backtest: {bt_status}", kind="success" if bt_status == "OK" else "warning"), unsafe_allow_html=True)
        st.caption(f"{bt_latest}")
        if (bt_latest / "metrics_overall.json").exists():
            st.caption(f"modified: {_mtime(bt_latest / 'metrics_overall.json')}")


def _render_snapshot_kpis(*, mart: pd.DataFrame, mart_full: pd.DataFrame, targets: pd.DataFrame, attainment: float | None, target_value: float | None, target_month: str | None) -> None:
    """Row of executive snapshot KPI cards."""
    spend = float(mart["spend"].sum()) if not mart.empty else 0.0
    premium = float(mart["premium"].sum()) if not mart.empty else 0.0
    roi = _safe_div(premium, spend)

    # Period comparison policy (business rule):
    # - If the selected range is within a single calendar month => MoM (previous month, same day range).
    # - If the selected range spans multiple months => YoY (same date range, previous year).
    # NOTE: Use mart_full for the comparison window to avoid "N/A" when the filtered window
    #       does not contain previous-period rows.
    cur_start = pd.to_datetime(mart["date"].min()) if (not mart.empty and "date" in mart.columns) else None
    cur_end = pd.to_datetime(mart["date"].max()) if (not mart.empty and "date" in mart.columns) else None

    # Normalize date columns for robust filtering.
    if not mart_full.empty and "date" in mart_full.columns:
        mart_full = mart_full.copy()
        mart_full["date"] = pd.to_datetime(mart_full["date"], errors="coerce")
    if not mart.empty and "date" in mart.columns:
        mart = mart.copy()
        mart["date"] = pd.to_datetime(mart["date"], errors="coerce")

    def _sum_between(df: pd.DataFrame, col: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> float | None:
        if df.empty or start_dt is None or end_dt is None or col not in df.columns or "date" not in df.columns:
            return None
        m = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        if m.empty:
            return None
        return float(pd.to_numeric(m[col], errors="coerce").sum())

    def _pct_change(cur: float | None, prev: float | None) -> float | None:
        if cur is None or prev is None or prev == 0:
            return None
        return (cur - prev) / prev

    def _fmt_delta(p: float | None) -> str:
        if p is None or pd.isna(p):
            return "—"
        sign = "+" if p >= 0 else ""
        return f"{sign}{p*100:.1f}%"

    comp_label = None
    comp_prev_start = None
    comp_prev_end = None
    if cur_start is not None and cur_end is not None:
        is_single_month = (cur_start.year == cur_end.year) and (cur_start.month == cur_end.month)
        if is_single_month:
            comp_label = "전월대비"
            comp_prev_start = cur_start - pd.DateOffset(months=1)
            comp_prev_end = cur_end - pd.DateOffset(months=1)
        else:
            comp_label = "전년동기"
            comp_prev_start = cur_start - pd.DateOffset(years=1)
            comp_prev_end = cur_end - pd.DateOffset(years=1)

    spend_prev = _sum_between(mart_full, "spend", comp_prev_start, comp_prev_end) if comp_prev_start is not None else None
    prem_prev = _sum_between(mart_full, "premium", comp_prev_start, comp_prev_end) if comp_prev_start is not None else None
    roi_prev = _safe_div(prem_prev or 0.0, spend_prev or 0.0) if (prem_prev is not None and spend_prev is not None and spend_prev != 0) else None

    spend_comp = f"{comp_label} {_fmt_delta(_pct_change(spend, spend_prev))}" if comp_label else "—"
    prem_comp = f"{comp_label} {_fmt_delta(_pct_change(premium, prem_prev))}" if comp_label else "—"
    roi_comp = f"{comp_label} {_fmt_delta(_pct_change(roi, roi_prev))}" if comp_label else "—"

    # Funnel totals
    leads = int(mart["leads"].sum()) if not mart.empty else 0
    attempts = int(mart["call_attempt"].sum()) if not mart.empty else 0
    connected = int(mart["call_connected"].sum()) if not mart.empty else 0
    contracts = int(mart["contracts"].sum()) if not mart.empty else 0

    premium_per_contract = _safe_div(premium, float(contracts))

    att_label = "목표 달성률"
    att_value = fmt_percent(attainment) if attainment is not None else "-"
    # target_month is a month range label (e.g., 2026-01~2026-03) aligned to the filtered mart coverage.
    att_sub = f"{target_month or '-'} 목표합계 {fmt_money(target_value) if target_value is not None else '-'}"

    # UX: make the comparison window explicit to avoid operator confusion.
    if comp_label and comp_prev_start is not None and comp_prev_end is not None:
        st.caption(
            f"비교 기준: {comp_label} · 비교 구간 {comp_prev_start.date()} ~ {comp_prev_end.date()}"
        )

    kpi_row(
        [
            ("총 집행(Spend)", fmt_money(spend), f"선택 필터 범위 · {spend_comp}"),
            ("총 프리미엄(Premium)", fmt_money(premium), f"선택 필터 범위 · {prem_comp}"),
            ("ROI(Premium/Spend)", f"{roi:.3f}", f"단순 비율 · {roi_comp}"),
            (att_label, att_value, att_sub),
        ]
    )
    kpi_row(
        [
            ("리드(Leads)", fmt_count(leads), "선택 필터 범위"),
            ("연결(Call Connected)", fmt_count(connected), "선택 필터 범위"),
            ("계약(Contracts)", fmt_count(contracts), f"건수 · 계약당 {fmt_money(premium_per_contract)}"),
            ("계약당 Premium(Premium/Contracts)", fmt_money(premium_per_contract), "평균(계약당)"),
        ]
    )
def _render_funnel_health(*, mart_filtered: pd.DataFrame, mart_full: pd.DataFrame) -> None:
    """Funnel health: **rate-first** KPIs + MoM deltas.

    Notes:
        MoM is computed against the immediately preceding period of equal length,
        using the *unfiltered* mart_full coverage. This prevents "전월대비 N/A" regressions
        when the current filter is narrower than the full dataset.
    """
    if mart_filtered.empty:
        st.info("Mart 데이터가 없습니다. (data/mart/daily_channel_fact.csv)")
        return

    from apps.dashboard.viewmodels.funnel_health_vm import build_funnel_health_vm
    from apps.dashboard.components.funnel_health import render_funnel_health

    vm = build_funnel_health_vm(mart_full=mart_full, mart_filtered=mart_filtered)
    render_funnel_health(vm)

    section("Bottleneck & Quality")
    invalid = int(
        (mart_filtered["contracts"] > mart_filtered["call_connected"]).sum()
        + (mart_filtered["call_connected"] > mart_filtered["call_attempt"]).sum()
    )
    if invalid > 0:
        st.markdown(badge(f"데이터 불변조건 위반 {invalid}건", kind="danger"), unsafe_allow_html=True)
    else:
        st.markdown(badge("데이터 불변조건 OK", kind="success"), unsafe_allow_html=True)
    st.caption("※ Funnel Health는 Rate-first(효율)입니다. 규모(amount)는 상세에서 확인하세요.")


def _render_alerts(*, vm: Any) -> None:
    """Alerts & Actions: Opportunity / Saturation / Downside risk."""
    section("Alerts & Actions")
    ch = getattr(vm, "channels", []) or []
    if not ch:
        st.info("인사이트를 생성할 추천/모델 결과가 없습니다.")
        return

    # Heuristics (policy thresholds can be moved to registry later)
    opp = sorted([x for x in ch if _to_float(getattr(x, "marginal_roi", 0.0), 0.0) >= 1.2], key=lambda x: _to_float(getattr(x, "marginal_roi", 0.0), 0.0), reverse=True)
    sat = sorted([x for x in ch if _to_float(getattr(x, "saturation_ratio", 0.0), 0.0) >= 0.8], key=lambda x: _to_float(getattr(x, "saturation_ratio", 0.0), 0.0), reverse=True)
    risk = sorted([x for x in ch if _to_float(getattr(x, "downside_risk", 0.0), 0.0) >= 0.15], key=lambda x: _to_float(getattr(x, "downside_risk", 0.0), 0.0), reverse=True)

    cards = []
    if opp:
        x = opp[0]
        cards.append(
            dict(
                title="High Opportunity",
                conclusion=f"{x.channel} 채널은 한계 ROI가 높아 증액 기회가 큽니다.",
                evidence=[
                    f"mROI: {_to_float(getattr(x, 'marginal_roi', 0.0), 0.0):.2f}",
                    f"saturation_ratio: {_to_float(getattr(x, 'saturation_ratio', 0.0), 0.0):.2f}",
                    f"half-life: {x.half_life:.1f}d",
                ],
                action="Recommendation에서 시나리오를 확인하세요.",
                kind="success",
            )
        )
    if sat:
        x = sat[0]
        cards.append(
            dict(
                title="Saturation Risk",
                conclusion=f"{x.channel} 채널은 포화 구간으로 효율이 떨어질 수 있습니다.",
                evidence=[
                    f"saturation_ratio: {_to_float(getattr(x, 'saturation_ratio', 0.0), 0.0):.2f}",
                    f"mROI: {_to_float(getattr(x, 'marginal_roi', 0.0), 0.0):.2f}",
                    f"EC50: {x.ec50:.1f}",
                ],
                action="증액 대신 타 채널 재배치를 고려하세요.",
                kind="warning",
            )
        )
    if risk:
        x = risk[0]
        cards.append(
            dict(
                title="Downside Risk",
                conclusion=f"{x.channel} 채널은 다운사이드 리스크가 상대적으로 큽니다.",
                evidence=[
                    f"downside_risk: {_to_float(getattr(x, 'downside_risk', 0.0), 0.0):.2f}",
                    f"mROI: {_to_float(getattr(x, 'marginal_roi', 0.0), 0.0):.2f}",
                    f"half-life: {x.half_life:.1f}d",
                ],
                action="Rollout 모드를 보수적으로 유지하거나 λ를 조정하세요.",
                kind="danger",
            )
        )

    if not cards:
        render_insight_card(
            title="No Critical Alerts",
            conclusion="현재 설정된 임계치 기준의 크리티컬 알림은 없습니다.",
            evidence=["mROI/포화/리스크가 임계치를 초과하지 않음"],
            action="Recommendation 패널에서 개선 여지를 탐색하세요.",
            kind="info",
        )
        return

    cols = st.columns(3, vertical_alignment="top")
    for i, card in enumerate(cards[:3]):
        with cols[i]:
            render_insight_card(**card)


def _build_channel_actual_table(*, mart: pd.DataFrame) -> pd.DataFrame:
    """Return channel-level actual funnel table from mart."""
    if mart.empty:
        return pd.DataFrame()
    cols = ["channel", "spend", "leads", "call_attempt", "call_connected", "contracts", "premium"]
    df = mart.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    g = (
        df.groupby("channel", as_index=False)[["spend", "leads", "call_attempt", "call_connected", "contracts", "premium"]]
        .sum()
        .sort_values("spend", ascending=False)
    )
    g["attempt_per_lead"] = g.apply(lambda r: _safe_div(float(r["call_attempt"]), float(r["leads"])), axis=1)
    g["connect_rate"] = g.apply(lambda r: _safe_div(float(r["call_connected"]), float(r["call_attempt"])), axis=1)
    g["crr"] = g.apply(lambda r: _safe_div(float(r["contracts"]), float(r["call_connected"])), axis=1)
    g["premium_per_contract"] = g.apply(lambda r: _safe_div(float(r["premium"]), float(r["contracts"])), axis=1)
    return g


def _build_plan_budget_table(*, vm: Any) -> pd.DataFrame:
    """Channel-level budget table for Do Nothing / Human / AI."""
    plans = getattr(vm, "plans", {}) or {}
    ai = getattr(plans.get("ai"), "budget_by_channel", {}) if plans.get("ai") else {}
    dn = getattr(plans.get("do_nothing"), "budget_by_channel", {}) if plans.get("do_nothing") else {}
    human = getattr(plans.get("human"), "budget_by_channel", {}) if plans.get("human") else {}
    chs = sorted(set(ai) | set(dn) | set(human))
    rows = []
    for ch in chs:
        rows.append(
            {
                "채널": ch,
                "No Doing 예산": _parse_money(dn.get(ch, 0.0)),
                "Human 예산": _parse_money(human.get(ch, 0.0)) if human else None,
                "AI 예산": _parse_money(ai.get(ch, 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Δ(AI-NoDoing)"] = out["AI 예산"] - out["No Doing 예산"]
    if "Human 예산" in out.columns and out["Human 예산"].notna().any():
        out["Δ(AI-Human)"] = out["AI 예산"] - out["Human 예산"].fillna(0.0)
    # Integer display for money columns
    for c in ["No Doing 예산", "Human 예산", "AI 예산", "Δ(AI-NoDoing)", "Δ(AI-Human)"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    return out


def _build_plan_funnel_table(*, mart: pd.DataFrame, vm: Any) -> pd.DataFrame:
    """Best-effort plan funnel outcomes (expected), derived from plan premium and baseline funnel ratios.

    If the decision artifact contains explicit funnel forecasts, those should be preferred in the future.
    For now we derive a consistent funnel from baseline ratios in mart:
      Premium -> Contracts -> Connected -> Attempts -> Leads
    """
    if mart.empty:
        return pd.DataFrame()
    # Baseline ratios from current filtered mart
    spend = float(mart["spend"].sum())
    leads = float(mart["leads"].sum())
    attempts = float(mart["call_attempt"].sum())
    connected = float(mart["call_connected"].sum())
    contracts = float(mart["contracts"].sum())
    premium = float(mart["premium"].sum())

    attempt_per_lead = _safe_div(attempts, leads)
    connect_rate = _safe_div(connected, attempts)
    crr = _safe_div(contracts, connected)
    premium_per_contract = _safe_div(premium, contracts)

    plans = getattr(vm, "plans", {}) or {}
    rows = []
    for key in ["do_nothing", "human", "ai"]:
        if key not in plans:
            continue
        p = plans[key]
        plan_name = {"ai": "AI", "human": "Human", "do_nothing": "No Doing"}.get(key, key)
        budget_map = getattr(p, "budget_by_channel", {}) or {}
        total_budget = float(sum(_parse_money(v) for v in budget_map.values()))
        exp_premium = _parse_money(getattr(p, "expected_premium", 0.0) or 0.0)
        if exp_premium <= 0 or premium_per_contract <= 0 or crr <= 0 or connect_rate <= 0 or attempt_per_lead <= 0:
            # Not enough information; still show budgets and premium if available.
            rows.append(
                {
                    "플랜": plan_name,
                    "총예산": total_budget,
                    "예상 Premium": exp_premium if exp_premium > 0 else None,
                    "예상 Leads": None,
                    "예상 Attempts": None,
                    "예상 Connected": None,
                    "예상 Contracts": None,
                    "계약당 Premium": None,
                }
            )
            continue

        exp_contracts = exp_premium / premium_per_contract
        exp_connected = exp_contracts / crr
        exp_attempts = exp_connected / connect_rate
        exp_leads = exp_attempts / attempt_per_lead

        rows.append(
            {
                "플랜": plan_name,
                "총예산": total_budget,
                "예상 Premium": exp_premium,
                "예상 Leads": exp_leads,
                "예상 Attempts": exp_attempts,
                "예상 Connected": exp_connected,
                "예상 Contracts": exp_contracts,
                "계약당 Premium": premium_per_contract,
            }
        )

    return pd.DataFrame(rows)


def _build_plan_channel_funnel_table(*, vm: Any, plan_funnel: pd.DataFrame) -> pd.DataFrame:
    """Channel-level best-effort outcomes by allocating plan totals proportionally to budget.

    This is a UI aid so stakeholders can inspect "AI/Human/NoDoing" at channel granularity
    without requiring explicit channel-level posterior forecasts.
    """
    if plan_funnel.empty:
        return pd.DataFrame()

    plans = getattr(vm, "plans", {}) or {}
    plan_key_by_name = {"AI": "ai", "Human": "human", "No Doing": "do_nothing"}
    rows: list[dict[str, object]] = []

    for _, r in plan_funnel.iterrows():
        plan_name = str(r.get("플랜"))
        key = plan_key_by_name.get(plan_name)
        if not key or key not in plans:
            continue
        p = plans[key]
        budget_map = getattr(p, "budget_by_channel", {}) or {}
        total_budget = float(sum(_parse_money(v) for v in budget_map.values()))
        if total_budget <= 0:
            continue

        # Pull plan totals
        tot_premium = _parse_money(r.get("예상 Premium"))
        tot_leads = _parse_money(r.get("예상 Leads"))
        tot_attempts = _parse_money(r.get("예상 Attempts"))
        tot_connected = _parse_money(r.get("예상 Connected"))
        tot_contracts = _parse_money(r.get("예상 Contracts"))
        prem_per_contract = _parse_money(r.get("계약당 Premium"))

        for ch, b in budget_map.items():
            b_val = _parse_money(b)
            w = _safe_div(b_val, total_budget)
            rows.append(
                {
                    "플랜": plan_name,
                    "채널": str(ch),
                    "예산": b_val,
                    "예상 Premium": (tot_premium * w) if tot_premium is not None else None,
                    "예상 Leads": (tot_leads * w) if tot_leads is not None else None,
                    "예상 Attempts": (tot_attempts * w) if tot_attempts is not None else None,
                    "예상 Connected": (tot_connected * w) if tot_connected is not None else None,
                    "예상 Contracts": (tot_contracts * w) if tot_contracts is not None else None,
                    "계약당 Premium": prem_per_contract,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Format numeric columns as integers for readability
    for c in ["예산", "예상 Premium", "예상 Leads", "예상 Attempts", "예상 Connected", "예상 Contracts", "계약당 Premium"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    out = out.sort_values(["플랜", "예산"], ascending=[True, False]).reset_index(drop=True)
    return out


def _sanitize_mart(mart: pd.DataFrame) -> pd.DataFrame:
    """Normalize mart schema/dtypes for consistent KPI math.

    The dashboard uses mart as a single source of truth for KPIs and channel tables.
    Any string-typed numbers (e.g., '1,234') must be coerced.
    """
    if mart is None or mart.empty:
        return pd.DataFrame(columns=[
            "date",
            "channel",
            "spend",
            "leads",
            "call_attempt",
            "call_connected",
            "contracts",
            "premium",
        ])
    df = mart.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["spend", "premium"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0.0)
    for col in ["leads", "call_attempt", "call_connected", "contracts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0).astype(int)
    if "channel" in df.columns:
        df["channel"] = df["channel"].astype(str)
    # Drop rows with invalid dates (should not contribute)
    if "date" in df.columns:
        df = df.dropna(subset=["date"])  # type: ignore[arg-type]
    return df


def main() -> None:
    ensure_project_sys_path()

    # Streamlit limitation: set_page_config must be the first Streamlit call.
    try:
        st.set_page_config(page_title="MMx Control Tower", layout="wide")
    except Exception:
        pass

    ctx = bootstrap(page_title="MMx Control Tower")

    nav_tabs = st.tabs(["Control Tower", "Backtest"])

    # -------------------------
    # Backtest tab
    # -------------------------
    with nav_tabs[1]:
        section("Backtest")
        artifacts_root = Path(getattr(ctx.paths, "artifacts", "") or Path(__file__).resolve().parents[2] / "artifacts")
        bt_root = artifacts_root / "backtests"

        # Allow selecting historical runs (latest + run_id folders)
        run_dirs: list[Path] = []
        if bt_root.exists():
            for p in sorted(bt_root.iterdir(), key=lambda x: x.name, reverse=True):
                if not p.is_dir():
                    continue
                if p.name == "latest":
                    continue
                run_dirs.append(p)

        def _label(p: Path) -> str:
            try:
                m = (p / "metrics_overall.json")
                ts = m.stat().st_mtime if m.exists() else p.stat().st_mtime
                dt = pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M")
                return f"{p.name}  ({dt})"
            except Exception:
                return p.name

        options = ["latest"] + [_label(p) for p in run_dirs]
        selected = st.selectbox("Backtest run 선택", options=options, index=0)
        if selected == "latest":
            bt_dir = bt_root / "latest"
        else:
            # map label back to directory
            idx = options.index(selected) - 1
            bt_dir = run_dirs[idx] if 0 <= idx < len(run_dirs) else (bt_root / "latest")

        st.caption(f"조회 경로: {bt_dir}")
        bt_vm = load_backtest_vm(bt_dir)
        render_backtest_section(bt_vm)

    # -------------------------
    # Control Tower tab
    # -------------------------
    with nav_tabs[0]:
        # Normalize mart dtypes once; use as single source of truth everywhere below.
        mart_df = _sanitize_mart(ctx.mart)

        # Decision artifacts are optional.
        artifacts_root = Path(getattr(ctx.paths, "artifacts", "") or Path(__file__).resolve().parents[2] / "artifacts")
        decision_dir = artifacts_root / "recommendations" / "decisions"
        dec_path = latest_decision_path(decision_dir)
        has_decision = bool(dec_path and dec_path.exists())
        actual_only = not has_decision

        if actual_only:
            st.warning(
                "결정(decision) 아티팩트를 찾지 못했습니다. 현재는 Actual-only 모드로 표시됩니다. "
                "(추천/플랜 비교를 보려면 Sample/Production 파이프라인을 실행하세요.)"
            )

        vm = build_decision_viewmodel(mart=mart_df, decision_path=dec_path, human_plan_path=None) if has_decision else None

        # Artifacts health panel (mart/decision/backtest)
        with st.expander("Artifacts Health", expanded=True):
            _render_artifacts_health(artifacts_root=artifacts_root, mart_df=mart_df, decision_path=dec_path)

        targets = _load_targets(ctx.paths)
        attainment, target_value, target_month = _compute_target_attainment(mart=mart_df, targets=targets)

        _render_topbar(ctx=ctx, vm=vm, actual_only=actual_only)

        st.divider()

        section("Performance Snapshot")
        _render_snapshot_kpis(mart=mart_df, mart_full=ctx.mart_full, targets=targets, attainment=attainment, target_value=target_value, target_month=target_month)

        st.divider()

        _render_funnel_health(mart_filtered=mart_df, mart_full=ctx.mart_full)

        st.divider()

        # 2) Drill-down (상세 보기) — always visible (no expander)
        section("Drill-down (상세 보기)")
        tabs = st.tabs(["채널별 현황(실적)", "채널별 퍼널 지표", "플랜 비교(AI/Base/No Doing)"])

        with tabs[0]:
            ch_tbl = _build_channel_actual_table(mart=mart_df)
            if ch_tbl.empty:
                st.info("채널별 집계 데이터가 없습니다.")
            else:
                view = ch_tbl.rename(
                    columns={
                        "channel": "채널",
                        "spend": "Spend",
                        "leads": "Leads",
                        "call_attempt": "Attempts",
                        "call_connected": "Connected",
                        "contracts": "Contracts",
                        "premium": "Premium",
                        "premium_per_contract": "계약당 Premium",
                    }
                )[["채널", "Spend", "Leads", "Attempts", "Connected", "Contracts", "Premium", "계약당 Premium"]].copy()
                for col in ["Spend", "Premium", "계약당 Premium"]:
                    view[col] = pd.to_numeric(view[col], errors="coerce").round(0).astype("Int64")
                for col in ["Leads", "Attempts", "Connected", "Contracts"]:
                    view[col] = pd.to_numeric(view[col], errors="coerce").round(0).astype("Int64")
                st.dataframe(view, use_container_width=True, hide_index=True)

        with tabs[1]:
            ch_tbl = _build_channel_actual_table(mart=mart_df)
            if ch_tbl.empty:
                st.info("채널별 퍼널 지표를 계산할 데이터가 없습니다.")
            else:
                view = pd.DataFrame(
                    {
                        "채널": ch_tbl["channel"].astype(str),
                        "리드당 콜시도": ch_tbl["attempt_per_lead"].apply(lambda x: fmt_ratio(float(x))),
                        "연결율": ch_tbl["connect_rate"].apply(lambda x: fmt_percent(float(x))),
                        "CRR": ch_tbl["crr"].apply(lambda x: fmt_percent(float(x))),
                        "계약당 Premium": ch_tbl["premium_per_contract"].apply(lambda x: fmt_money(float(x))),
                    }
                )
                st.dataframe(view, use_container_width=True, hide_index=True)
                st.caption(
                    "※ 계산식: 리드당 콜시도=Attempts/Leads · 연결율=Connected/Attempts · CRR=Contracts/Connected · 계약당 Premium=Premium/Contracts"
                )

        with tabs[2]:
            if vm is None:
                st.info("플랜 비교는 decision 아티팩트가 필요합니다. Sample/Production 파이프라인을 실행하세요.")
            else:
                render_decision_summary(
                    vm=vm,
                    artifacts_root=Path(ctx.paths.artifacts),
                    mart=mart_df,
                    mode="plan_compare_only",
                )

        st.divider()
        # 3) Latest decision summary (결론/액션)
        if vm is None:
            section("최신 의사결정 요약")
            st.info("현재 Actual-only 모드입니다. decision 아티팩트가 생성되면 AI/Base 비교와 액션 요약이 표시됩니다.")
        else:
            render_decision_summary(
                vm=vm,
                artifacts_root=Path(ctx.paths.artifacts),
                mart=mart_df,
                mode="full",
            )
if __name__ == "__main__":
    main()