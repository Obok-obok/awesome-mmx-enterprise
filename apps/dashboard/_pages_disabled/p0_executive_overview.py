# MMX_SYS_PATH_GUARD: ensure repo root is importable when running via `streamlit run`
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


"""Executive overview.

Design goals:
- Standardized KPI cards
- Targets vs Actual (monthly)
- Do Nothing vs AI comparison is always visible
- Funnel stage forecast vs actual is included in the executive narrative
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# IMPORTANT:
# first Streamlit command executed in that page script.


from components.bootstrap import bootstrap
from components.artifact_gate import ArtifactCheck, render_artifact_gate
from components.plots_business import plot_target_variance_bar
from components.ui import (
    fmt_money,
    fmt_count,
    fmt_ratio,
    fmt_percent,
    kpi_row,
    section,
    style_table,
)
from components.tables import add_totals
from components.decision_viewmodel import (
    latest_decision_path,
    build_decision_viewmodel,
)
from components.tables_business import build_budget_comparison_table
from components.insights import render_top_insights_from_channels


def _load_targets(target_path: Path) -> pd.DataFrame:
    if not target_path.exists():
        return pd.DataFrame(columns=["month", "target_premium"])
    df = pd.read_csv(target_path)
    df["month"] = df["month"].astype(str)
    df["target_premium"] = pd.to_numeric(df["target_premium"], errors="coerce").fillna(0.0)
    return df


def run() -> None:
    ctx = bootstrap("MMx | Executive")

    section("Executive Overview", "목표 → 현상 → 추천 → 검증을 한 눈에 보는 경영진용 화면")
    df = ctx.mart
    if df.empty:
        st.info("선택된 기간/채널에 데이터가 없습니다. (Mart 빌드 + 필터 범위를 확인하세요.)")
        return

    total_spend = float(df["spend"].sum())
    total_premium = float(df["premium"].sum())
    roi = total_premium / (total_spend + 1e-9)

    # Monthly actuals
    m = df.copy()
    m["month"] = m["date"].dt.to_period("M").astype(str)
    actual_by_month = m.groupby("month", as_index=False)["premium"].sum()

    # Targets
    target_path = Path(ctx.paths.data_curated) / "targets/monthly_targets.csv"
    targets = _load_targets(target_path)
    merged = actual_by_month.merge(targets, on="month", how="left").fillna({"target_premium": 0.0})
    merged["attainment"] = merged.apply(
        lambda r: (r["premium"] / r["target_premium"]) if r["target_premium"] > 0 else 0.0,
        axis=1,
    )

    latest_row = merged.tail(1)
    attainment = float(latest_row["attainment"].iloc[0]) if len(latest_row) else 0.0
    target_val = float(latest_row["target_premium"].iloc[0]) if len(latest_row) else 0.0

    kpi_row(
        [
            {"label": "총 광고비(Spend)", "value": fmt_money(total_spend), "sub": "선택 필터 범위"},
            {"label": "총 프리미엄(Premium)", "value": fmt_money(total_premium), "sub": "선택 필터 범위"},
            {"label": "ROI (Premium/Spend)", "value": fmt_ratio(roi, 3), "sub": "단순 비율"},
            {
                "label": "목표 달성률(최근월)",
                "value": fmt_percent(attainment, 1) if target_val > 0 else "—",
                "sub": (f"목표: {fmt_money(target_val)}" if target_val > 0 else "Targets 미등록"),
            },
        ]
    )


    # Funnel actuals (selected filter range)
    st.caption("선택 기간/필터 기준 퍼널 실적 합계입니다. (리드→통화시도→연결→계약)")
    kpi_row(
        [
            {"label": "리드(Leads)", "value": fmt_count(float(df["leads"].sum())), "sub": "선택 필터 범위"},
            {"label": "통화시도(Call Attempt)", "value": fmt_count(float(df["call_attempt"].sum())), "sub": "선택 필터 범위"},
            {"label": "연결(Call Connected)", "value": fmt_count(float(df["call_connected"].sum())), "sub": "선택 필터 범위"},
            {"label": "계약(Contracts)", "value": fmt_count(float(df["contracts"].sum())), "sub": "선택 필터 범위"},
        ]
    )

    st.divider()

    section("목표 대비 갭(월)", "라인 차트 대신, 목표 대비 부족/초과를 한 눈에")
    plot_target_variance_bar(
        merged,
        month_col="month",
        actual_col="premium",
        target_col="target_premium",
        title="월별 Premium: 목표 vs 실적",
        y_label="Premium(원)",
    )

    st.divider()

    decision_dir = Path(ctx.paths.artifacts) / "recommendations/decisions"
    latest = latest_decision_path(decision_dir)
    if latest is None:
        st.info("아직 의사결정(decision)이 없습니다. recommend 실행 후 확인하세요.")
        return

    # Optional human plan artifact (if present)
    from components.decision_viewmodel import latest_human_plan_path
    human_plan = latest_human_plan_path(Path(ctx.paths.artifacts))

    vm = build_decision_viewmodel(mart=ctx.mart, decision_path=latest, human_plan_path=human_plan)

    from components.decision_summary import render_decision_summary
    render_decision_summary(vm=vm, artifacts_root=Path(ctx.paths.artifacts), mart=None, mode="full")

    st.divider()

    section("핵심 인사이트", "추천의 이유(근거 3개 + 액션) 요약")
    render_top_insights_from_channels(vm.channels, max_channels=4)
render_top_insights_from_channels(vm.channels, top_k=3)


run()
