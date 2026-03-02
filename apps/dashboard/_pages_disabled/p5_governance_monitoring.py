# MMX_SYS_PATH_GUARD: ensure repo root is importable when running via `streamlit run`
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


"""Governance & Monitoring.

Design requirements:
- Stable path resolution (never rely on cwd)
- Standard tables and consistent formatting
- Show: promotion log, experiment registry, audit log hints
"""

from pathlib import Path
import json

import pandas as pd
import streamlit as st

# IMPORTANT:
# first Streamlit command executed in that page script.


from components.bootstrap import bootstrap
from components.artifact_gate import ArtifactCheck, render_artifact_gate
from components.ui import section, style_table, kpi_row, badge, fmt_percent, fmt_count
from components.labels import rename_columns_for_display


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sources_df(report: dict) -> pd.DataFrame:
    src = report.get("sources", {}) if isinstance(report, dict) else {}
    rows = []
    for name, v in src.items():
        if not isinstance(v, dict):
            continue
        dp = v.get("date_parse", {}) if isinstance(v.get("date_parse"), dict) else {}
        rows.append(
            {
                "Source": name,
                "Rows": v.get("rows", 0),
                "Rows (after drop unknown)": v.get("rows_after_drop_unknown", None),
                "Unique channels": v.get("unique_channels", 0),
                "Unknown rows": v.get("unknown_rows", 0),
                "Unknown ratio": v.get("unknown_ratio", None),
                "NaT ratio": dp.get("nat_ratio", None),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Sort: most risky sources first
    def _nan_to_neg(x: object) -> float:
        try:
            return float(x)
        except Exception:
            return -1.0
    df["_risk"] = df[["Unknown ratio", "NaT ratio"]].applymap(_nan_to_neg).max(axis=1)
    df = df.sort_values(["_risk", "Rows"], ascending=[False, False]).drop(columns=["_risk"]).reset_index(drop=True)
    return df


def run() -> None:
    ctx = bootstrap("MMx | Governance")
    section("Governance & Monitoring", "모델 승격/실험/감사 로그 및 운영 지표")

    # Data Quality Gate (mart validation report)
    st.divider()
    section("Data Quality Gate", "입력 데이터 정합성(스키마/채널/날짜/퍼널 단조성) 자동 검증 결과")
    rep_path = Path(ctx.paths.logs) / "pipeline/mart_validation_latest.json"
    report = _read_json(rep_path)
    if not report:
        st.info("아직 데이터 정합성 리포트가 없습니다. 먼저 `python scripts/build_mart.py`를 실행하세요.")
    else:
        # KPIs
        mart = report.get("mart", {}) if isinstance(report, dict) else {}
        mono = (mart.get("monotonicity", {}) if isinstance(mart, dict) else {})
        viol_ratio = mono.get("violation_ratio", None)

        # Aggregate worst ratios across sources
        src_df = _sources_df(report)
        worst_unknown = float(src_df["Unknown ratio"].max()) if (not src_df.empty and src_df["Unknown ratio"].notna().any()) else 0.0
        worst_nat = float(src_df["NaT ratio"].max()) if (not src_df.empty and src_df["NaT ratio"].notna().any()) else 0.0

        kpi_row(
            [
                {"label": "Generated at (UTC)", "value": str(report.get("generated_at", "—"))},
                {"label": "Worst NaT ratio", "value": fmt_percent(worst_nat), "sub": f"Threshold: {fmt_percent(float(report.get('max_nat_ratio', 0.0)))}"},
                {"label": "Worst unknown channel", "value": fmt_percent(worst_unknown), "sub": f"Threshold: {fmt_percent(float(report.get('max_unknown_channel_ratio', 0.0)))}"},
                {"label": "Monotonicity violation", "value": fmt_percent(float(viol_ratio) if viol_ratio is not None else 0.0), "sub": f"Threshold: {fmt_percent(float(report.get('max_monotonic_violation_ratio', 0.0)))}"},
            ]
        )

        # Badges for enforcement
        enforce_allow = bool(report.get("enforce_channel_allowlist", False))
        enforce_mono = bool(report.get("enforce_funnel_monotonicity", False))
        c_badge1, c_badge2, c_badge3 = st.columns([1, 1, 3])
        with c_badge1:
            badge("Allowlist ENFORCED" if enforce_allow else "Allowlist REPORT-ONLY", "ok" if enforce_allow else "info")
        with c_badge2:
            badge("Monotonicity ENFORCED" if enforce_mono else "Monotonicity REPORT-ONLY", "ok" if enforce_mono else "info")
        with c_badge3:
            allowed = report.get("allowed_channels", [])
            if allowed:
                st.caption(f"Allowed channels: {', '.join([str(x) for x in allowed])}")
            else:
                st.caption("Allowed channels: (not set)")

        # Per-source table
        if src_df.empty:
            st.caption("No source details.")
        else:
            st.dataframe(
                style_table(
                    src_df,
                    count_cols=["Rows", "Rows (after drop unknown)", "Unique channels", "Unknown rows"],
                    pct_cols=["Unknown ratio", "NaT ratio"],
                ),
                use_container_width=True,
                height=280,
            )

        # Mart-level monotonicity details
        if isinstance(mono, dict) and mono.get("violations"):
            st.caption("Monotonicity violations (examples):")
            st.json(mono.get("violations"))

        with st.expander("View raw validation JSON"):
            st.json(report)

    promo = Path(ctx.paths.logs) / "training/model_promotion_log.csv"
    exp = Path(ctx.paths.logs) / "experiments/experiment_registry.csv"
    audit = Path(ctx.paths.logs) / "api/audit_log.csv"

    c1, c2 = st.columns(2)
    with c1:
        section("Model Promotion Log", "학습 모델 → Production 승격 기록")
        df = _read_csv(promo)
        if df.empty:
            st.info("승격 로그 없음")
        else:
            df_display = rename_columns_for_display(df, ctx.labels)
            st.dataframe(style_table(df_display), use_container_width=True, height=260)

    with c2:
        section("Experiment Registry", "Shadow/AB/rollout 실험 기록")
        df = _read_csv(exp)
        if df.empty:
            st.info("실험 레지스트리 없음")
        else:
            df_display = rename_columns_for_display(df, ctx.labels)
            st.dataframe(style_table(df_display), use_container_width=True, height=260)

    st.divider()
    section("API Audit Log", "입력 데이터/추천 요청/승격 이벤트의 감사 기록")
    df = _read_csv(audit)
    if df.empty:
        st.caption("감사 로그가 없거나 아직 생성되지 않았습니다.")
    else:
        df_display = rename_columns_for_display(df, ctx.labels)
        st.dataframe(style_table(df_display), use_container_width=True, height=260)

    st.divider()
    section("운영 점검", "Data quality / reporting delay 진단")
    st.markdown(
        "- Reporting delay 진단: 주말 누적/다음날 반영 등 체계적 지연이 확인되면 `MMX_REPORTING_DELAY=ON`으로 학습\n"
        "- Data quality 리포트: `apps/api`의 monitoring 엔드포인트 또는 `scripts/data_quality.py` (있다면)로 확인"
    )


run()
