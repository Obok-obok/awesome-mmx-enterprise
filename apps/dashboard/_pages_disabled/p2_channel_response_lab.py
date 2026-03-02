# MMX_SYS_PATH_GUARD: ensure repo root is importable when running via `streamlit run`
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from pathlib import Path
import json

import pandas as pd
import streamlit as st

# IMPORTANT:
# first Streamlit command executed in that page script.


from components.bootstrap import bootstrap
from components.ui import (
    section,
    kpi_row,
    kpi_card,
    fmt_ratio,
    fmt_money,
    badge,
    style_table,
)
from components.plots import plot_curve, CurveMarkers
from components.artifact_gate import ArtifactCheck, render_artifact_gate


def _interp_at(df: pd.DataFrame, x_col: str, y_col: str, x: float) -> float:
    """Linear interpolation of y at x on a curve."""
    if df.empty:
        return float("nan")
    sdf = df.sort_values(x_col)
    xs = sdf[x_col].to_numpy(dtype=float)
    ys = sdf[y_col].to_numpy(dtype=float)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    import numpy as np

    return float(np.interp(x, xs, ys))


def _saturation_level(sat: float) -> tuple[str, str]:
    """Return (label, kind) for a saturation badge."""
    if sat >= 0.90:
        return (f"포화 매우 높음({sat:.2f})", "danger")
    if sat >= 0.85:
        return (f"포화 높음({sat:.2f})", "warn")
    if sat >= 0.60:
        return (f"중간({sat:.2f})", "info")
    return (f"여지 큼({sat:.2f})", "ok")


def _latest_decision(ctx) -> dict:
    dec_dir = ctx.paths.artifacts / "recommendations/decisions"
    if not dec_dir.exists():
        return {}
    files = sorted(dec_dir.glob("dec_*.json"))
    if not files:
        return {}
    return json.loads(files[-1].read_text(encoding="utf-8"))


def run() -> None:
    ctx = bootstrap("MMx | Channel Response")
    section(
        "Channel Response Lab",
        "Adstock+Saturation 기반 반응곡선/한계 ROI/포화지표/반감기 + (AI/Do Nothing/Human/EC50 마커) + 현재 지점 mROI/포화도",
    )

    root = ctx.paths.artifacts / "explainability"

    # Artifact Gate: prevent empty/ambiguous screens.
    checks = [
        ArtifactCheck(name="Mart (daily_channel_fact.csv)", path=ctx.paths.mart / "daily_channel_fact.csv", required=True, hint="데이터 마트가 필요합니다. scripts/build_mart.py를 먼저 실행하세요."),
        ArtifactCheck(name="Recommend Decision (dec_*.json)", path=ctx.paths.artifacts / "recommendations/decisions", required=True, hint="추천/설명 아티팩트를 생성하려면 scripts/recommend.py를 실행하세요."),
        ArtifactCheck(name="Explainability Root", path=root, required=True, hint="recommend 실행 시 explainability 아티팩트가 생성됩니다."),
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

    versions = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not versions:
        st.info("Explainability 버전이 없습니다.")
        return

    # Decision selector for markers
    dec = _latest_decision(ctx)
    has_dec = bool(dec)
    if not has_dec:
        st.warning("추천(Decision) 아티팩트가 없어 마커(현재 추천 예산/Do Nothing)를 표시할 수 없습니다. recommend를 먼저 실행하세요.")

    ver = st.selectbox("Model version", versions, index=len(versions)-1)

    # 초고급형: 시나리오 비교
    st.sidebar.markdown("---")
    st.sidebar.subheader("Scenario 비교")
    compare_mode = st.sidebar.toggle("AI vs Do Nothing vs Human 비교", value=True)
    human_json = st.sidebar.text_area(
        "Human 플랜(JSON: {\"channel\": budget, ...})",
        value="",
        help="선택: Human 예산 계획을 입력하면 곡선에 Human 마커를 추가하고 mROI/포화도를 비교합니다.",
        height=120,
    )
    human_plan: dict[str, float] = {}
    if human_json.strip():
        try:
            import json

            obj = json.loads(human_json)
            if isinstance(obj, dict):
                human_plan = {str(k): float(v) for k, v in obj.items()}
        except Exception:
            st.sidebar.warning("Human 플랜 JSON 파싱 실패. 예: {\"kakao\": 1000000, \"meta\": 2000000}")

    sat_path = root / ver / "saturation_metrics.csv"
    resp_path = root / ver / "response_curve.csv"
    mroi_path = root / ver / "mroi_curve.csv"

    # Fallback: 선택된 모델 버전에 곡선 아티팩트가 없으면, 가장 최근(존재하는) 버전을 자동으로 사용합니다.
    if not (resp_path.exists() and mroi_path.exists()):
        cand = []
        if root.exists():
            for d in sorted([p for p in root.iterdir() if p.is_dir()]):
                if (d / "response_curve.csv").exists() and (d / "mroi_curve.csv").exists():
                    cand.append(d.name)
        if cand:
            fallback_ver = cand[-1]
            if fallback_ver != ver:
                st.warning(
                    f"선택한 모델 버전({ver})에 response/mROI 곡선이 없어, 최근 생성된 버전({fallback_ver})의 아티팩트를 표시합니다."
                )
                ver = fallback_ver
                resp_path = root / ver / "response_curve.csv"
                mroi_path = root / ver / "mroi_curve.csv"

    sat = pd.DataFrame()
    if sat_path.exists():
        sat = pd.read_csv(sat_path)
        sat_disp = sat.rename(columns={
            "channel": ctx.labels.name("channel"),
            "ec50": ctx.labels.name("ec50"),
            "half_life_days": ctx.labels.name("half_life"),
            "saturation_at_mid": ctx.labels.name("saturation_ratio"),
        })
        section("포화/반감기 요약")
        st.dataframe(style_table(sat_disp, float_cols=[ctx.labels.name('ec50'), ctx.labels.name('half_life')], digits=2), use_container_width=True, height=240)

    if resp_path.exists() and mroi_path.exists():
        resp = pd.read_csv(resp_path)
        mroi = pd.read_csv(mroi_path)
        channels = sorted(resp["channel"].unique().tolist())
        ch = st.selectbox("매체", channels)

        # Marker spends
        ai_spend = float(dec.get("recommended_budget", {}).get(ch, 0.0)) if has_dec else None
        dn_spend = None
        baseline_budget = dec.get("baseline_budget", {}) if has_dec else {}
        if baseline_budget and isinstance(baseline_budget, dict):
            dn_spend = float(baseline_budget.get(ch, 0.0))

        human_spend = float(human_plan.get(ch, 0.0)) if (compare_mode and human_plan) else None

        # KPI cards
        if not sat.empty and (sat["channel"] == ch).any():
            row = sat[sat["channel"] == ch].iloc[0]
            ec50_val = float(row["ec50"])
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi_card("EC50(포화지점)", fmt_money(float(row["ec50"])))
            with c2:
                kpi_card("반감기(일)", fmt_ratio(float(row["half_life_days"]), 1))
            with c3:
                kpi_card("포화도(참고)", fmt_ratio(float(row["saturation_at_mid"]), 2), "(총예산 중간값 기준)")
            with c4:
                if ai_spend is not None:
                    kpi_card("현재 추천 예산", fmt_money(ai_spend))
                else:
                    kpi_card("현재 추천 예산", "—")

            # 초고급형 KPI: 현재 지점 mROI/포화도 + 상태 배지
            if ai_spend is not None and ai_spend > 0:
                sat_now = float((ai_spend / 30.0) / ((ai_spend / 30.0) + ec50_val)) if ec50_val > 0 else 0.0
                mroi_now = _interp_at(mroi[mroi["channel"] == ch], "spend", "mroi", ai_spend)
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    kpi_card("현재 지점 mROI", fmt_ratio(mroi_now, 6), "(추천 예산에서의 한계효율)")
                with cc2:
                    kpi_card("현재 지점 포화도", fmt_ratio(sat_now, 2), "(추천 예산 기준)")
                with cc3:
                    lbl, kind = _saturation_level(sat_now)
                    st.markdown("<div class='card'><div class='kpi-label'>포화 상태</div>", unsafe_allow_html=True)
                    badge(lbl, kind=kind)
                    st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # Response curve with markers
        section("반응곡선(Response Curve)", "수확 체감(포화) 포함. 세로선은 의사결정 지점입니다.")
        resp_ch = resp[resp["channel"] == ch].sort_values("spend")
        ec50 = float(row["ec50"]) if (not sat.empty and (sat["channel"] == ch).any()) else None
        plot_curve(
            df=resp_ch,
            x="spend",
            y="response",
            title=f"{ch} 반응곡선",
            x_label="예산(Spend)",
            y_label="반응(Lead proxy)",
            markers=CurveMarkers(current_spend=ai_spend, baseline_spend=dn_spend, human_spend=human_spend, ec50=ec50),
            subtitle_lines=[
                f"AI 추천: {fmt_money(ai_spend) if ai_spend is not None else '—'}",
                f"Do Nothing: {fmt_money(dn_spend) if dn_spend is not None else '—'}",
                f"Human: {fmt_money(human_spend) if human_spend is not None else '—'}" if compare_mode else "",
            ],
        )

        # mROI curve with markers
        section("한계 ROI 곡선(mROI)", "현재 예산 수준에서 1원 추가 집행의 기대 효율(한계효용)입니다.")
        mroi_ch = mroi[mroi["channel"] == ch].sort_values("spend")
        plot_curve(
            df=mroi_ch,
            x="spend",
            y="mroi",
            title=f"{ch} 한계 ROI",
            x_label="예산(Spend)",
            y_label="mROI",
            markers=CurveMarkers(current_spend=ai_spend, baseline_spend=dn_spend, human_spend=human_spend, ec50=ec50),
        )

        # 초고급형: Human vs AI 한계효율 요약
        if compare_mode and human_spend is not None and human_spend > 0 and ai_spend is not None and ai_spend > 0:
            mroi_ai = _interp_at(mroi_ch, "spend", "mroi", ai_spend)
            mroi_h = _interp_at(mroi_ch, "spend", "mroi", human_spend)
            st.caption(
                f"한계효율 비교(해당 채널): AI mROI={fmt_ratio(mroi_ai,6)} vs Human mROI={fmt_ratio(mroi_h,6)} | Δ={fmt_ratio(mroi_ai - mroi_h,6)}"
            )

        # Simple decision narration
        if ai_spend is not None and ec50 is not None:
            if ai_spend / 30.0 > ec50:
                st.info(f"해석: 현재 추천 예산은 EC50(포화지점)을 넘어 포화 구간에 근접합니다. 이 매체는 추가 집행의 효율이 낮아질 수 있습니다.")
            else:
                st.success(f"해석: 현재 추천 예산은 EC50(포화지점) 이전 구간입니다. 추가 집행 시 효율 상승 여지가 있습니다.")

    else:
        st.info("response_curve/mroi_curve 아티팩트가 없습니다. recommend를 먼저 실행하세요.")


run()
