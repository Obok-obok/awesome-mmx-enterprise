# MMX_SYS_PATH_GUARD: ensure repo root is importable when running via `streamlit run`
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


"""Funnel diagnostics.

Standardized requirements:
- Business terminology for all columns
- Rates table + bottleneck callouts
- Numeric formatting and consistent table height
"""

import numpy as np
import pandas as pd
import streamlit as st

# IMPORTANT:
# first Streamlit command executed in that page script.


from components.bootstrap import bootstrap
from components.artifact_gate import ArtifactCheck, render_artifact_gate
from components.ui import section, style_table, fmt_percent, fmt_ratio, kpi_row


def run() -> None:
    ctx = bootstrap("MMx | Funnel Diagnostics")
    df = ctx.mart

    section("Funnel Diagnostics", "퍼널 단계별 전환율과 병목을 점검")
    if df.empty:
        st.info("선택된 기간/채널에 데이터가 없습니다.")
        return

    eps = 1e-9
    x = df.copy()
    x["lead_per_spend"] = x["leads"] / (x["spend"] + eps)
    x["attempt_per_lead"] = x["call_attempt"] / (x["leads"] + eps)
    x["connected_rate"] = x["call_connected"] / (x["call_attempt"] + eps)
    x["contract_rate"] = x["contracts"] / (x["call_connected"] + eps)
    x["premium_per_contract"] = x["premium"] / (x["contracts"] + eps)

    # Aggregate summary by channel
    g = x.groupby("channel", as_index=False).agg(
        spend=("spend", "sum"),
        premium=("premium", "sum"),
        lead_per_spend=("lead_per_spend", "mean"),
        attempt_per_lead=("attempt_per_lead", "mean"),
        connected_rate=("connected_rate", "mean"),
        contract_rate=("contract_rate", "mean"),
        premium_per_contract=("premium_per_contract", "mean"),
    )
    g["roi"] = g["premium"] / (g["spend"] + eps)

    # Bottleneck index: min stage rate among the three controllable stages.
    g["bottleneck_score"] = np.minimum.reduce([g["attempt_per_lead"], g["connected_rate"], g["contract_rate"]])

    kpi_row(
        [
            {"label": "평균 Lead/Spend", "value": fmt_ratio(float(g["lead_per_spend"].mean()), 4), "sub": "리드 효율"},
            {"label": "평균 Attempt/Lead", "value": fmt_percent(float(g["attempt_per_lead"].mean()), 1), "sub": "콜시도 전환"},
            {"label": "평균 Connected/Attempt", "value": fmt_percent(float(g["connected_rate"].mean()), 1), "sub": "연결 전환"},
            {"label": "평균 Contract/Connected", "value": fmt_percent(float(g["contract_rate"].mean()), 1), "sub": "계약 전환"},
        ]
    )

    st.divider()

    section("채널별 요약", "ROI와 병목 지표를 함께 확인")
    disp = g.rename(
        columns={
            "channel": "매체",
            "spend": "광고비(Spend)",
            "premium": "프리미엄(Premium)",
            "roi": "ROI(Premium/Spend)",
            "lead_per_spend": "Lead/Spend",
            "attempt_per_lead": "Attempt/Lead",
            "connected_rate": "Connected/Attempt",
            "contract_rate": "Contract/Connected",
            "premium_per_contract": "Premium/Contract",
            "bottleneck_score": "병목 점수(낮을수록)"
        }
    ).sort_values("ROI(Premium/Spend)", ascending=False)

    st.dataframe(
        style_table(
            disp,
            money_cols=["광고비(Spend)", "프리미엄(Premium)", "Premium/Contract"],
            float_cols=["ROI(Premium/Spend)", "Lead/Spend", "병목 점수(낮을수록)"],
            pct_cols=["Attempt/Lead", "Connected/Attempt", "Contract/Connected"],
            digits=4,
        ),
        use_container_width=True,
        height=360,
    )

    st.divider()
    section("일자별 상세", "데이터 품질/이상치/주말 누적 등을 확인")
    detail = x[[
        "date", "channel", "spend", "leads", "call_attempt", "call_connected", "contracts", "premium",
        "lead_per_spend", "attempt_per_lead", "connected_rate", "contract_rate", "premium_per_contract",
    ]].copy()
    detail = detail.rename(
        columns={
            "date": "일자",
            "channel": "매체",
            "spend": "광고비(Spend)",
            "leads": "리드(Leads)",
            "call_attempt": "통화시도(Attempt)",
            "call_connected": "연결(Connected)",
            "contracts": "계약(Contracts)",
            "premium": "프리미엄(Premium)",
            "lead_per_spend": "Lead/Spend",
            "attempt_per_lead": "Attempt/Lead",
            "connected_rate": "Connected/Attempt",
            "contract_rate": "Contract/Connected",
            "premium_per_contract": "Premium/Contract",
        }
    )
    st.dataframe(
        style_table(
            detail,
            money_cols=["광고비(Spend)", "프리미엄(Premium)", "Premium/Contract"],
            count_cols=["리드(Leads)", "통화시도(Attempt)", "연결(Connected)", "계약(Contracts)"],
            pct_cols=["Attempt/Lead", "Connected/Attempt", "Contract/Connected"],
            float_cols=["Lead/Spend"],
            digits=4,
        ),
        use_container_width=True,
        height=420,
    )


run()
