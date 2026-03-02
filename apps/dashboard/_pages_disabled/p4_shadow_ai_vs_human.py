# MMX_SYS_PATH_GUARD: ensure repo root is importable when running via `streamlit run`
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


"""Shadow Mode evaluation.

Spec (must-have):
1) AI prediction accuracy (forecast vs actual)
2) AI recommendation vs Human plan difference
3) Counterfactual evaluation (posterior predictive): ΔRA, CI, P(AI>Human)
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

# IMPORTANT:
# first Streamlit command executed in that page script.


from components.bootstrap import bootstrap
from components.artifact_gate import ArtifactCheck, render_artifact_gate
from components.ui import section, kpi_row, fmt_money, fmt_percent, style_table
from components.tables import add_totals


def _latest_json(dir_path: Path, pattern: str) -> dict:
    if not dir_path.exists():
        return {}
    files = sorted(dir_path.glob(pattern))
    if not files:
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum(np.abs(actual)) + 1e-9
    return float(np.sum(np.abs(actual - pred)) / denom)


def run() -> None:
    ctx = bootstrap("MMx | Shadow Mode")
    section("Shadow Mode Evaluation", "AI vs Human 플랜을 안전하게 비교(예산 리스크를 통제)")

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
        ArtifactCheck(
            name="Shadow 평가 결과(shadow_*.json)",
            path=Path(ctx.paths.artifacts) / "evaluations/shadow",
            required=False,
            hint="evaluate_shadow.py 실행 시 생성됩니다.",
        ),
    ]
    ok = render_artifact_gate(
        checks=checks,
        run_buttons={
            "샘플 전체 파이프라인 실행": ["bash", "scripts/demo_run_all.sh"],
            "마트 생성(build_mart)": ["python", "scripts/build_mart.py"],
            "추천 생성(recommend)": ["python", "scripts/recommend.py"],
            "Shadow 평가(evaluate_shadow)": ["python", "scripts/evaluate_shadow.py"],
        },
        header="필수 아티팩트 확인",
    )
    if not ok:
        return

    # Inputs
    st.sidebar.markdown("---")
    st.sidebar.subheader("Human 플랜")
    human_json = st.sidebar.text_area(
        "Human 플랜(JSON: {\"channel\": budget, ...})",
        value="",
        height=120,
        help="선택: Human 플랜이 있으면 AI vs Human plan diff를 계산합니다.",
    )
    human_plan: dict[str, float] = {}
    if human_json.strip():
        try:
            obj = json.loads(human_json)
            if isinstance(obj, dict):
                human_plan = {str(k): float(v) for k, v in obj.items()}
        except Exception:
            st.sidebar.error("Human JSON 파싱 실패")

    # Load latest decision & shadow result
    dec_dir = Path(ctx.paths.artifacts) / "recommendations/decisions"
    dec = _latest_json(dec_dir, "dec_*.json")
    sh_dir = Path(ctx.paths.artifacts) / "evaluations/shadow"
    shadow = _latest_json(sh_dir, "shadow_*.json")

    tab1, tab2, tab3 = st.tabs(["① 예측 정확도", "② 플랜 차이", "③ Counterfactual 평가"]) 

    # ① Accuracy
    with tab1:
        section("예측 정확도", "퍼널 단계별 예상(E) vs 실제(A) 비교")
        if not dec:
            st.info("Decision이 없어 forecast artifact를 찾을 수 없습니다.")
        else:
            fpath = Path(ctx.paths.artifacts) / "recommendations/funnel_forecast" / f"{dec.get('decision_id')}.csv"
            if not fpath.exists():
                st.info("funnel_forecast artifact가 없습니다.")
            else:
                fdf = pd.read_csv(fpath)
                fdf = fdf.dropna(subset=["expected", "actual"], how="any")
                if fdf.empty:
                    st.info("예측/실제 값이 충분하지 않습니다.")
                else:
                    wape = _wape(fdf["actual"].to_numpy(float), fdf["expected"].to_numpy(float))
                    kpi_row([
                        {"label": "WAPE(전체)", "value": fmt_percent(wape, 1), "sub": "가중절대오차"},
                        {"label": "예측 기간", "value": str(dec.get("period_start", "-")) + " ~ " + str(dec.get("period_end", "-")), "sub": "decision period"},
                        {"label": "모델 버전", "value": str(dec.get("model_version", "-")), "sub": "production"},
                        {"label": "Reporting Delay", "value": str(dec.get("reporting_delay", "OFF")), "sub": "옵션"},
                    ])
                    disp = fdf[["stage", "expected", "actual", "gap", "ci_low", "ci_high"]].copy()
                    stage_map = {
                        "leads": "리드",
                        "call_attempt": "통화시도",
                        "call_connected": "연결",
                        "contracts": "계약",
                        "premium": "프리미엄",
                    }
                    disp["퍼널 단계"] = disp["stage"].map(stage_map).fillna(disp["stage"])
                    disp = disp.drop(columns=["stage"]).rename(
                        columns={
                            "expected": "예상(E)",
                            "actual": "실제(A)",
                            "gap": "차이(A-E)",
                            "ci_low": "CI Low",
                            "ci_high": "CI High",
                        }
                    )
                    st.dataframe(style_table(disp, money_cols=["예상(E)", "실제(A)"], float_cols=["차이(A-E)", "CI Low", "CI High"], digits=2), use_container_width=True, height=260)

    # ② Plan diff
    with tab2:
        section("플랜 차이", "AI vs Do Nothing vs Human 예산/비중 비교")
        if not dec:
            st.info("Decision이 없습니다.")
        else:
            ai = dec.get("recommended_budget", {})
            dn = dec.get("do_nothing_budget", {})
            all_ch = sorted(set(ai.keys()) | set(dn.keys()) | set(human_plan.keys()))
            rows = []
            ai_total = float(sum(ai.values()))
            dn_total = float(sum(dn.values()))
            hu_total = float(sum(human_plan.values())) if human_plan else 0.0
            for ch in all_ch:
                a = float(ai.get(ch, 0.0))
                d = float(dn.get(ch, 0.0))
                h = float(human_plan.get(ch, 0.0))
                rows.append(
                    {
                        "매체": ch,
                        "AI 예산": a,
                        "DN 예산": d,
                        "Human 예산": h,
                        "AI 비중": a / (ai_total + 1e-9),
                        "DN 비중": d / (dn_total + 1e-9),
                        "Human 비중": h / (hu_total + 1e-9) if hu_total > 0 else 0.0,
                        "Δ(AI-DN)": a - d,
                        "Δ(AI-H)": a - h if human_plan else 0.0,
                    }
                )
            bud = pd.DataFrame(rows)
            bud = add_totals(bud, numeric_cols=["AI 예산", "DN 예산", "Human 예산", "Δ(AI-DN)", "Δ(AI-H)"])
            st.dataframe(
                style_table(
                    bud,
                    money_cols=["AI 예산", "DN 예산", "Human 예산", "Δ(AI-DN)", "Δ(AI-H)"],
                    pct_cols=["AI 비중", "DN 비중", "Human 비중"],
                    digits=2,
                ),
                use_container_width=True,
                height=360,
            )

    # ③ Counterfactual
    with tab3:
        section("Counterfactual 평가", "Posterior predictive로 AI가 더 나았을 확률을 계산")
        if not shadow:
            st.info("Shadow evaluation 결과가 없습니다. scripts/evaluate_shadow.py 실행 후 확인하세요.")
        else:
            kpi_row(
                [
                    {"label": "ΔRA(AI-Human)", "value": fmt_money(float(shadow.get("delta_ra_mean", 0.0))), "sub": "평균"},
                    {"label": "P(AI > Human)", "value": fmt_percent(float(shadow.get("p_ai_better", 0.0)), 1), "sub": "확률"},
                    {"label": "CI(ΔRA)", "value": str(shadow.get("delta_ra_ci", "-")), "sub": "95%"},
                    {"label": "평가 대상", "value": str(shadow.get("decision_id", "-")), "sub": "decision"},
                ]
            )
            st.json(shadow)


run()
