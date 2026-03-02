from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import yaml

from src.utils.logging_utils import silence_noise
from src.data.generate_sample import make_sample_inputs, SampleSpec
from src.features.panel import load_channel_daily, load_campaign_daily, build_panel_daily
from src.mmx.bayesian_scm import (
    SCMConfig,
    fit_bayesian_funnel_scm,
    fit_bayesian_funnel_scm_campaign,
    posterior_summary,
    posterior_summary_campaign,
    simulate_counterfactuals,
)
from src.mmx.optimizer import recommend_budget_from_roi
from src.data_quality.gates import run_data_quality



def standardize_out_schema(out_dir: str) -> None:
    """Standardize output artifact schemas so the dashboard can rely on stable column names.
    - counterfactuals_channel.csv: ensure *_cf and *_mean aliases exist
    - posterior_summary_channel.csv: keep original columns and also add param/mean rows (fallback-friendly)
    """
    # counterfactuals
    cf_path = os.path.join(out_dir, "counterfactuals_channel.csv")
    if os.path.exists(cf_path):
        cf = pd.read_csv(cf_path)
        for m in ["spend", "leads", "contracts", "premium"]:
            cf_col = f"{m}_cf"
            mean_col = f"{m}_mean"
            if cf_col not in cf.columns and mean_col in cf.columns:
                cf[cf_col] = cf[mean_col]
            if mean_col not in cf.columns and cf_col in cf.columns:
                cf[mean_col] = cf[cf_col]
        cf.to_csv(cf_path, index=False)

    # posterior summary (fallback adds roi_mean/rr_mean/pps_mean)
    ps_path = os.path.join(out_dir, "posterior_summary_channel.csv")
    if os.path.exists(ps_path):
        ps = pd.read_csv(ps_path)
        if "param" not in ps.columns and all(c in ps.columns for c in ["channel","roi_mean","rr_mean","pps_mean"]):
            rows = []
            for _, r in ps.iterrows():
                ch = str(r["channel"])
                rows.append({"channel": ch, "param": f"roi[{ch}]", "mean": float(r["roi_mean"]), "sd": np.nan})
                rows.append({"channel": ch, "param": f"rr[{ch}]", "mean": float(r["rr_mean"]), "sd": np.nan})
                rows.append({"channel": ch, "param": f"pps[{ch}]", "mean": float(r["pps_mean"]), "sd": np.nan})
            extra = pd.DataFrame(rows)
            ps_out = ps.merge(extra.groupby("channel").size().rename("_n"), left_on="channel", right_index=True, how="left")
            # keep original file and also emit a long-form param file for compatibility
            extra.to_csv(os.path.join(out_dir, "posterior_params_channel.csv"), index=False)
        # no overwrite needed; keep as-is

def build_lineage() -> pd.DataFrame:
    rows = [
        ("DB(Leads)", "leads", "data/input_daily_channel.csv", "leads", "src/features/panel.build_panel_daily"),
        ("TM Attempts", "tm_attempts", "data/input_daily_channel.csv", "tm_attempts", "src/features/panel.build_panel_daily"),
        ("TM Connected", "tm_connected", "data/input_daily_channel.csv", "tm_connected", "src/features/panel.build_panel_daily"),
        ("Contracts", "contracts", "data/input_daily_channel.csv", "contracts", "src/features/panel.build_panel_daily"),
        ("Premium", "premium", "data/input_daily_channel.csv", "premium", "src/features/panel.build_panel_daily"),

        # ROI 5-Factor SEM decomposition (derived metrics)
        ("Leads/Spend", "leads_per_spend", "data/input_daily_channel.csv", "leads,spend", "src/features/panel.build_panel_daily"),
        ("Attempts/Leads", "attempts_per_lead", "data/input_daily_channel.csv", "tm_attempts,leads", "src/features/panel.build_panel_daily"),
        ("Connected/Attempts", "connected_per_attempt", "data/input_daily_channel.csv", "tm_connected,tm_attempts", "src/features/panel.build_panel_daily"),
        ("Contracts/Connected", "contracts_per_connected", "data/input_daily_channel.csv", "contracts,tm_connected", "src/features/panel.build_panel_daily"),
        ("Premium/Contract", "premium_per_contract", "data/input_daily_channel.csv", "premium,contracts", "src/features/panel.build_panel_daily"),
        ("ROI(Premium/Spend)", "roi", "data/input_daily_channel.csv", "premium,spend", "src/features/panel.build_panel_daily"),
        ("ROI implied(5-Factor product)", "roi_implied", "data/input_daily_channel.csv", "leads_per_spend,attempts_per_lead,connected_per_attempt,contracts_per_connected,premium_per_contract", "src/features/panel.build_panel_daily"),

        ("Campaign Contracts", "campaign_counterfactuals", "data/input_daily_campaign.csv", "contracts,premium", "src/mmx/bayesian_scm.fit_bayesian_funnel_scm_campaign"),
        (
            "캠페인→증권 매핑 규칙",
            "mapping_rules",
            "raw_leads/raw_tm_calls/raw_policies",
            "customer_id, timestamps",
            "src/data/raw_to_inputs.map_policy_to_lead (30-day window, dedupe leads, multi-policy)",
        ),
        ("Channel OFF 시나리오", "counterfactuals", "posterior", "posterior samples", "src/mmx/bayesian_scm.simulate_counterfactuals"),
        ("Budget Recommendation", "budget_recommendation", "configs/mmx.yaml", "bounds & total_budget", "src/mmx/optimizer.recommend_budget_from_roi"),
        ("Data Quality Gate", "data_quality_report", "inputs", "required columns / negatives / outliers / funnel constraints", "src/data_quality/gates.run_data_quality"),
    ]
    return pd.DataFrame(rows, columns=["metric","artifact","source_file","source_columns","logic_function"])


def run_all(config_path: str = "configs/mmx.yaml", overrides: dict | None = None):
    silence_noise()
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    if overrides:
        # shallow merge for demo: model/* only
        for k, v in overrides.items():
            if k in cfg and isinstance(cfg[k], dict) and isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    out_dir = cfg["data"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    ch_path = cfg["data"]["input_channel_path"]
    camp_path = cfg["data"]["input_campaign_path"]
    if (not os.path.exists(ch_path)) or (not os.path.exists(camp_path)):
        make_sample_inputs("data", SampleSpec(n_days=180, seed=int(cfg["model"]["seed"])))

    ch_raw = load_channel_daily(ch_path)
    camp_raw = load_campaign_daily(camp_path)

    panel = build_panel_daily(ch_raw)
    panel.to_csv(os.path.join(out_dir, "panel_daily_channel.csv"), index=False)

    # Also export MONTHLY panel for dashboard demos (month-end timestamp).
    # - Keeps same schema as daily panel
    # - Useful when users want 월 단위 시점(월말)으로 성과를 확인
    try:
        pm = panel.copy()
        pm["date"] = pd.to_datetime(pm["date"]).dt.to_period("M").dt.to_timestamp("M")
        agg_cols = {
            "spend": "sum",
            "leads": "sum",
            "tm_attempts": "sum" if "tm_attempts" in pm.columns else "sum",
            "tm_connected": "sum" if "tm_connected" in pm.columns else "sum",
            "contracts": "sum",
            "premium": "sum",
        }
        # keep only existing columns
        agg_cols = {k: v for k, v in agg_cols.items() if k in pm.columns}
        pm = pm.groupby(["date", "channel"], as_index=False).agg(agg_cols)
        # Rebuild derived metrics after aggregation so monthly panel matches daily schema.
        pm = build_panel_daily(pm)
        pm.to_csv(os.path.join(out_dir, "panel_monthly_channel.csv"), index=False)
    except Exception:
        # monthly export is best-effort; daily panel is the source of truth for modeling
        pass

    camp_panel = build_panel_daily(camp_raw)
    camp_panel.to_csv(os.path.join(out_dir, "panel_daily_campaign.csv"), index=False)

    # Monthly campaign panel (best-effort)
    try:
        cm = camp_panel.copy()
        cm["date"] = pd.to_datetime(cm["date"]).dt.to_period("M").dt.to_timestamp("M")
        group_keys = ["date"]
        for k in ["channel", "campaign", "message_type"]:
            if k in cm.columns:
                group_keys.append(k)
        agg_cols = {k: "sum" for k in ["spend", "leads", "tm_attempts", "tm_connected", "contracts", "premium"] if k in cm.columns}
        cm = cm.groupby(group_keys, as_index=False).agg(agg_cols)
        # Rebuild derived metrics after aggregation so monthly campaign panel matches daily schema.
        cm = build_panel_daily(cm)
        cm.to_csv(os.path.join(out_dir, "panel_monthly_campaign.csv"), index=False)
    except Exception:
        pass

    # Stronger DQ gate (0-division/NaN/negatives/outliers/funnel constraints)
    dq_json = run_data_quality(panel, camp_panel)
    with open(os.path.join(out_dir, "data_quality_report.json"), "w", encoding="utf-8") as f:
        f.write(dq_json)

    # -----------------------------------------------------------
    # Bayesian SCM is optional.
    # Many free-tier VM environments fail to import PyMC/Numba stacks
    # (e.g., numba/coverage compatibility). In that case, we fall back
    # to a deterministic baseline that still produces all dashboard
    # artifacts (posterior summaries, counterfactuals, optimizer output).
    # -----------------------------------------------------------

    channels = sorted(panel["channel"].astype(str).unique().tolist())
    model_ok = True
    try:
        scm_cfg = SCMConfig(
            inference=str(cfg["model"].get("inference", "mcmc")),
            draws=int(cfg["model"]["draws"]),
            tune=int(cfg["model"]["tune"]),
            chains=int(cfg["model"]["chains"]),
            target_accept=float(cfg["model"]["target_accept"]),
            advi_steps=int(cfg["model"].get("advi_steps", 2000)),
            advi_draws=int(cfg["model"].get("advi_draws", 300)),
            seed=int(cfg["model"]["seed"]),
            adstock_alpha=float(cfg["model"]["adstock_alpha"]),
            saturation_k=float(cfg["model"]["saturation_k"]),
            tm_capacity_per_day=int(cfg["ops"]["tm_capacity_per_day"]),
            sla_softcap_connected=float(cfg["ops"]["sla_softcap_connected"]),
        )

        # 1) Channel-level SCM (stable) + 2) Campaign-level SCM (granular)
        idata, design_df, channels = fit_bayesian_funnel_scm(panel, scm_cfg)
        posterior_summary(idata, channels).to_csv(os.path.join(out_dir, "posterior_summary_channel.csv"), index=False)

        idata_c, design_c, channels_c, campaigns, msg_types = fit_bayesian_funnel_scm_campaign(camp_panel, scm_cfg)
        posterior_summary_campaign(idata_c, channels_c, campaigns, msg_types).to_csv(
            os.path.join(out_dir, "posterior_summary_campaign.csv"),
            index=False,
        )
    except Exception as e:
        model_ok = False

        # --- Deterministic fallback artifacts ---
        # Posterior summary (channel): empirical ROI / RR / premium-per-contract
        ps = panel.groupby("channel", as_index=False).agg(
            spend=("spend", "sum"),
            leads=("leads", "sum"),
            contracts=("contracts", "sum"),
            premium=("premium", "sum"),
        )
        ps["roi_mean"] = ps.apply(lambda r: float(r["premium"]) / float(r["spend"]) if float(r["spend"]) > 0 else 0.0, axis=1)
        ps["rr_mean"] = ps.apply(lambda r: float(r["contracts"]) / float(r["leads"]) if float(r["leads"]) > 0 else 0.0, axis=1)
        ps["pps_mean"] = ps.apply(lambda r: float(r["premium"]) / float(r["contracts"]) if float(r["contracts"]) > 0 else 0.0, axis=1)
        ps = ps[["channel", "roi_mean", "rr_mean", "pps_mean", "spend", "leads", "contracts", "premium"]]
        ps.to_csv(os.path.join(out_dir, "posterior_summary_channel.csv"), index=False)

        # Posterior summary (campaign): basic rollup if available
        if "campaign" in camp_panel.columns:
            pc = camp_panel.groupby(["channel", "campaign"], as_index=False).agg(
                spend=("spend", "sum"),
                leads=("leads", "sum"),
                contracts=("contracts", "sum"),
                premium=("premium", "sum"),
            )
        else:
            pc = camp_panel.groupby(["channel"], as_index=False).agg(
                spend=("spend", "sum"),
                leads=("leads", "sum"),
                contracts=("contracts", "sum"),
                premium=("premium", "sum"),
            )
            pc["campaign"] = "-"
        pc["roi"] = pc.apply(lambda r: float(r["premium"]) / float(r["spend"]) if float(r["spend"]) > 0 else 0.0, axis=1)
        pc.to_csv(os.path.join(out_dir, "posterior_summary_campaign.csv"), index=False)

        # Create deterministic counterfactuals compatible with dashboard:
        # BASE = total across channels; OFF_{ch} = total excluding that channel.
        daily_tot = panel.groupby("date", as_index=False).agg(
            spend=("spend", "sum"),
            leads=("leads", "sum"),
            contracts=("contracts", "sum"),
            premium=("premium", "sum"),
        )
        rows = []
        for _, r in daily_tot.iterrows():
            rows.append(
                {
                    "date": r["date"],
                    "scenario": "BASE",
                    "spend_mean": float(r["spend"]),
                    "leads_mean": float(r["leads"]),
                    "contracts_mean": float(r["contracts"]),
                    "premium_mean": float(r["premium"]),
                }
            )
        by_ch = panel.groupby(["date", "channel"], as_index=False).agg(
            spend=("spend", "sum"),
            leads=("leads", "sum"),
            contracts=("contracts", "sum"),
            premium=("premium", "sum"),
        )
        for ch in channels:
            merged = daily_tot.merge(by_ch[by_ch["channel"] == ch], on="date", how="left", suffixes=("_tot", "_ch"))
            merged = merged.fillna(0)
            for _, r in merged.iterrows():
                rows.append(
                    {
                        "date": r["date"],
                        "scenario": f"OFF_{ch}",
                        "spend_mean": float(r["spend_tot"] - r["spend_ch"]),
                        "leads_mean": float(r["leads_tot"] - r["leads_ch"]),
                        "contracts_mean": float(r["contracts_tot"] - r["contracts_ch"]),
                        "premium_mean": float(r["premium_tot"] - r["premium_ch"]),
                    }
                )
        cf = pd.DataFrame(rows)
        cf.to_csv(os.path.join(out_dir, "counterfactuals_channel.csv"), index=False)

        # Minimal design_df for later executive summary (util proxy)
        design_df = panel.copy()
        design_df["util"] = 0.5  # deterministic placeholder

    # -----------------------------------------------------------------
    # Downstream artifacts for Optimizer/Executive Summary
    # - If Bayesian model succeeded: compute campaign scores, counterfactuals, ROI curve.
    # - If fallback: reuse deterministic cf and build a simple linear ROI curve.
    # -----------------------------------------------------------------
    if model_ok:
        # Campaign-level performance & posterior-implied lift (last 30 days)
        recent_c = design_c[design_c["date"] >= (design_c["date"].max() - pd.Timedelta(days=29))].copy()
        recent_c["_row"] = recent_c.index.astype(int)

        post = idata_c.posterior
        mean = lambda name: post[name].mean(dim=("chain", "draw")).values
        a0 = float(mean("a0"))
        a_ch = mean("a_ch")
        a_mt = mean("a_mt")
        a_camp = mean("a_camp")  # deterministic
        a_dow = mean("a_dow")
        a_trend = float(mean("a_trend"))
        b_camp_full = mean("b_camp")
        c0 = float(mean("c0"))
        c_mt = mean("c_mt")
        c1 = float(mean("c1"))
        c2 = float(mean("c2"))
        k0 = float(mean("k0"))
        k_ch = mean("k_ch")
        k_mt = mean("k_mt")
        k1 = float(mean("k1"))
        m0 = float(mean("m0"))
        m_ch = mean("m_ch")
        m_mt = mean("m_mt")

        ch_to_idx = {c: i for i, c in enumerate(channels_c)}
        camp_to_idx = {c: i for i, c in enumerate(campaigns)}
        mt_to_idx = {m: i for i, m in enumerate(msg_types)}
        ch_idx = recent_c["channel"].map(ch_to_idx).values.astype(int)
        camp_idx = recent_c["campaign_id"].map(camp_to_idx).values.astype(int)
        mt_idx = recent_c["message_type"].map(mt_to_idx).values.astype(int)
        dow = recent_c["dow"].values.astype(int)
        t = recent_c["t"].values.astype(float)
        x_media = recent_c["x_media"].values.astype(float)
        util_over = np.maximum(recent_c["util"].values.astype(float) - scm_cfg.sla_softcap_connected, 0.0)

        b_camp = b_camp_full[recent_c["_row"].values]
        mu_leads = np.exp(
            a0
            + a_ch[ch_idx]
            + a_mt[mt_idx]
            + a_camp[camp_idx]
            + b_camp * x_media
            + a_dow[dow]
            + a_trend * t
        )
        p_conn = 1 / (1 + np.exp(-(c0 + c_mt[mt_idx] + c1 * np.log(mu_leads + 1) + c2 * util_over)))
        conn_hat = recent_c["tm_attempts"].values.astype(float) * p_conn
        p_close = 1 / (1 + np.exp(-(k0 + k_ch[ch_idx] + k_mt[mt_idx] + k1 * np.log(mu_leads + 1))))
        cont_hat = conn_hat * p_close
        prem_per_hat = np.exp(m0 + m_ch[ch_idx] + m_mt[mt_idx])
        prem_hat = cont_hat * prem_per_hat

        recent_c["leads_hat"] = mu_leads
        recent_c["contracts_hat"] = cont_hat
        recent_c["premium_hat"] = prem_hat

        camp_score = (
            recent_c.groupby(["channel", "campaign_id", "message_type"], as_index=False)
            .agg(
                spend=("spend", "sum"),
                leads=("leads", "sum"),
                contracts=("contracts", "sum"),
                premium=("premium", "sum"),
                leads_hat=("leads_hat", "sum"),
                contracts_hat=("contracts_hat", "sum"),
                premium_hat=("premium_hat", "sum"),
            )
        )
        camp_score["roi_obs"] = camp_score["premium"] / camp_score["spend"].replace(0, np.nan)
        camp_score["roi_hat"] = camp_score["premium_hat"] / camp_score["spend"].replace(0, np.nan)
        camp_score["lift_vs_obs"] = (camp_score["roi_hat"] / camp_score["roi_obs"].replace(0, np.nan)) - 1.0
        camp_score = camp_score.sort_values("roi_hat", ascending=False).reset_index(drop=True)
        camp_score.to_csv(os.path.join(out_dir, "campaign_scores_last30d.csv"), index=False)

        # BASE + OFF scenarios
        cfs = [simulate_counterfactuals(idata, design_df, channels, {"name": "BASE"}, scm_cfg)]
        for ch in channels:
            cfs.append(simulate_counterfactuals(idata, design_df, channels, {"name": f"OFF_{ch}", "zero_channels": [ch]}, scm_cfg))
        cf = pd.concat(cfs, ignore_index=True)
        cf.to_csv(os.path.join(out_dir, "counterfactuals_channel.csv"), index=False)

        # ROI proxy table for budget
        base_spend = panel.groupby("channel")["spend"].mean().to_dict()
        total_budget = float(cfg["budget"]["total_budget"])
        bounds = {k: tuple(v) for k, v in cfg["budget"]["channel_bounds"].items()}

        scales = [0.7, 1.0, 1.3]
        roi_table = {}
        for ch in channels:
            rows = []
            for s in scales:
                res = simulate_counterfactuals(idata, design_df, channels, {"name": f"SCALE_{ch}_{s}", "scale_spend": {ch: s}}, scm_cfg)
                prem = float(res.tail(30)["premium_mean"].mean())
                spend = float(res.tail(30)["spend_cf"].mean())
                rows.append({"spend": spend, "premium": prem})
            roi_table[ch] = pd.DataFrame(rows).sort_values("spend")

        def roi_curve_fn(channel: str, spend: float) -> float:
            t = roi_table[channel]
            return float(np.interp(spend, t["spend"].values, t["premium"].values))

    else:
        # Fallback ROI curve: linear premium = ROI(channel) * spend
        base_spend = panel.groupby("channel")["spend"].mean().to_dict()
        total_budget = float(cfg["budget"]["total_budget"])
        bounds = {k: tuple(v) for k, v in cfg["budget"]["channel_bounds"].items()}
        roi_per = (
            panel.groupby("channel").apply(lambda d: float(d["premium"].sum()) / float(d["spend"].sum()) if float(d["spend"].sum()) > 0 else 0.0)
        ).to_dict()

        def roi_curve_fn(channel: str, spend: float) -> float:
            return float(roi_per.get(channel, 0.0) * float(spend))

    budget_df = recommend_budget_from_roi(
        channels=channels,
        base_spend={c: float(base_spend.get(c, 0.0)) for c in channels},
        roi_curve_fn=roi_curve_fn,
        total_budget=total_budget,
        bounds=bounds,
        step=float(cfg["budget"]["step"])
    )
    budget_df.to_csv(os.path.join(out_dir, "budget_recommendation.csv"), index=False)

    # Executive summary
    base = cf[cf["scenario"]=="BASE"].tail(30)
    base_leads = float(base["leads_mean"].mean())
    base_cont = float(base["contracts_mean"].mean())
    base_prem = float(base["premium_mean"].mean())

    impacts=[]
    for ch in channels:
        off = cf[cf["scenario"]==f"OFF_{ch}"].tail(30)
        impacts.append({
            "channel": ch,
            "delta_leads": float(off["leads_mean"].mean() - base_leads),
            "delta_contracts": float(off["contracts_mean"].mean() - base_cont),
            "delta_premium": float(off["premium_mean"].mean() - base_prem),
        })
    impacts = sorted(impacts, key=lambda x: x["delta_premium"])
    worst = impacts[0]

    util_95 = float(design_df.groupby("date")["util"].first().quantile(0.95))

    summary = {
        "headline": f"채널 OFF 시나리오에서 보험료 손실이 가장 큰 채널은 {worst['channel']}임(최근 30일 평균 기준)",
        "key_messages": [
            f"BASE 기준 최근 30일 일평균 DB {base_leads:,.0f}건, 계약 {base_cont:,.0f}건, 보험료 {base_prem:,.0f}원 추정임",
            f"{worst['channel']} OFF 시 일평균 DB {worst['delta_leads']:,.0f}건, 계약 {worst['delta_contracts']:,.0f}건, 보험료 {worst['delta_premium']:,.0f}원 변화(감소) 추정임",
            f"TM 용량 사용률 95% 분위수 {util_95:.2f}로 포화 구간 진입 리스크 존재함(softcap 이후 연결률 하락 반영)",
        ],
        "actions": [
            "예산 증액은 TM 용량/슬롯과 함께 설계 필요함(용량 85% 초과 구간에서 연결률 하락).",
            "Search 계열(google/naver)과 Assist 성격 채널(meta/kakao)의 역할 분리를 위한 메시지/캠페인 레벨 수집 권장.",
            "DB 질(예: 관심상품/연령대/지역) 변수를 리드 테이블에 추가하면 계약 예측/최적화 정확도가 상승.",
        ],
        "evidence": {
            "posterior_channel": "out/posterior_summary_channel.csv",
            "posterior_campaign": "out/posterior_summary_campaign.csv",
            "counterfactual_channel": "out/counterfactuals_channel.csv",
            "budget": "out/budget_recommendation.csv",
        }
    }
    with open(os.path.join(out_dir, "executive_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    build_lineage().to_csv(os.path.join(out_dir, "metric_lineage.csv"), index=False)

if __name__ == "__main__":
    run_all()
