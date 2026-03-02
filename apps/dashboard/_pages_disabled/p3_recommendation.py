# MMX_SYS_PATH_GUARD: ensure repo root is importable when running via `streamlit run`
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


"""Recommendation page (budget allocation).

Redesign goals:
- Compare AI vs Do Nothing (Do Nothing is NEVER 0; fallback from mart)
- Show decision in business terms: ΔSpend, ΔRA, risk
- Replace raw JSON with "Insight cards" (Conclusion → Evidence → Action)
"""

from pathlib import Path

import streamlit as st

from components.bootstrap import bootstrap
from components.ui import section, kpi_row, fmt_money, fmt_percent, style_table
from components.tables import add_totals
from components.artifact_gate import ArtifactCheck, render_artifact_gate
from components.decision_viewmodel import latest_decision_path, build_decision_viewmodel
from components.tables_business import build_channel_decision_table
from components.insights import render_top_insights_from_channels


def run() -> None:
    ctx = bootstrap("MMx | Recommendation")
    section("Budget Recommendation", "총 예산 제약 내에서 Risk-Adjusted Premium을 최대화")

    # Artifact Gate
    checks = [
        ArtifactCheck(
            name="Mart (daily_channel_fact.csv)",
            path=ctx.paths.mart / "daily_channel_fact.csv",
            required=True,
            hint="데이터 마트가 필요합니다. scripts/build_mart.py를 먼저 실행하세요.",
        ),
        ArtifactCheck(
            name="Recommend Decision (dec_*.json)",
            path=Path(ctx.paths.artifacts) / "recommendations/decisions",
            required=True,
            hint="추천/설명 아티팩트를 생성하려면 scripts/recommend.py를 실행하세요.",
        ),
    ]
    ok = render_artifact_gate(
        checks=checks,
        run_buttons={
            "샘플 전체 파이프라인 실행": ["bash", "scripts/demo_run_all.sh"],
            "마트 생성(build_mart)": ["python", "scripts/build_mart.py"],
            "추천 생성(recommend)": ["python", "scripts/recommend.py"],
        },
        header="필수 아티팩트 확인",
    )
    if not ok:
        return

    decision_dir = Path(ctx.paths.artifacts) / "recommendations/decisions"
    latest = latest_decision_path(decision_dir)
    if latest is None:
        st.info("의사결정(decision)이 없습니다. scripts/recommend.py 실행 후 확인하세요.")
        return

    vm = build_decision_viewmodel(mart=ctx.mart, decision_path=latest)

    ai_ra = float(vm.plans["ai"].ra_premium)
    dn_ra = float(vm.plans["do_nothing"].ra_premium)
    p_better = float(vm.plans["ai"].p_win)
    delta_ra = ai_ra - dn_ra

    section("요약", "추천 플랜(AI)과 현상유지(Do Nothing)를 위험조정 기준으로 비교합니다.")
    st.caption(f"예측 대상 기간: {vm.period_start} ~ {vm.period_end} (KST)")
    kpi_row(
        [
            {"label": "AI 위험조정 프리미엄(RA)", "value": fmt_money(ai_ra), "sub": "추천 플랜"},
            {"label": "현상유지 위험조정 프리미엄(RA)", "value": fmt_money(dn_ra), "sub": "Do Nothing"},
            {"label": "ΔRA (AI-DN)", "value": fmt_money(delta_ra), "sub": "클수록 유리"},
            {"label": "P(AI > DN)", "value": fmt_percent(p_better, 1), "sub": "posterior"},
        ]
    )

    st.divider()
    section("채널별 추천", "예산 + 근거(요약)까지 한 번에")

    tdf = build_channel_decision_table(vm.channels)
    tdf = add_totals(
        tdf,
        numeric_cols=[
            "현상유지 예산",
            "AI 추천 예산",
            "Δ 예산(AI-DN)",
        ],
    )
    st.dataframe(
        style_table(
            tdf,
            money_cols=["현상유지 예산", "AI 추천 예산", "Δ 예산(AI-DN)"],
            float_cols=["mROI", "포화도", "half-life", "Downside"],
            digits=3,
        ),
        use_container_width=True,
        height=380,
    )

    st.divider()
    section("핵심 인사이트", "결론 → 근거(3) → 액션")
    render_top_insights_from_channels(vm.channels, top_k=3)


run()
