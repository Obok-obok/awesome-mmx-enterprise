from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.data.generate_sample import SampleSpec, make_sample_inputs
from src.mmx.bayesian_scm import SCMConfig, _prep_design, _sanitize_funnel_df
from src.utils.logging_utils import silence_noise
from src.mmx_dynamic.adaptive_scm import AdaptiveSCM, AdaptiveSCMConfig
from src.mmx_dynamic.budget_optimizer import BudgetBounds, recommend_budget_thompson_explore
from src.mmx_dynamic.drift import DriftConfig, ResidualDriftDetector
from src.mmx_dynamic.state_io import load_adaptive_state, save_adaptive_state
from src.ops.ab_holdout import assign_weekly_groups, apply_holdout_multipliers
from src.ops.geo_holdout import GeoHoldoutConfig, build_geo_spend_plan


logger = logging.getLogger(__name__)


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _media_features(panel: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    if "x_media" not in panel.columns:
        panel = panel.copy()
        panel["x_media"] = panel["spend"].astype(float)
    piv = panel.pivot_table(index="date", columns="channel", values="x_media", aggfunc="sum").fillna(0.0)
    for ch in channels:
        if ch not in piv.columns:
            piv[ch] = 0.0
    piv = piv[channels]
    piv.reset_index(inplace=True)
    return piv


def _load_or_bootstrap_model(
    conf: AdaptiveSCMConfig,
    state_path: Path,
    dates: List[pd.Timestamp],
    media: pd.DataFrame,
    facts: pd.DataFrame,
    channels: List[str],
) -> AdaptiveSCM:
    if state_path.exists():
        return load_adaptive_state(AdaptiveSCM, AdaptiveSCMConfig, state_path)

    model = AdaptiveSCM(conf)
    # bootstrap with full history
    for t, d in enumerate(dates):
        x_media = media.loc[media["date"] == d, channels].iloc[0].to_dict()
        day = facts[facts["date"] == d]
        info = model.update_day(
            date=d,
            x_media_by_ch=x_media,
            leads_total=float(day["leads"].sum()),
            tm_attempts_total=float(day["tm_attempts"].sum()),
            connected_by_ch={ch: float(day.loc[day["channel"] == ch, "tm_connected"].sum()) for ch in channels},
            contracts_by_ch={ch: float(day.loc[day["channel"] == ch, "contracts"].sum()) for ch in channels},
            premium_by_ch={ch: float(day.loc[day["channel"] == ch, "premium"].sum()) for ch in channels},
            t_index=t,
        )
        model.state.last_date = str(d.date())
        model.state.t_index = t
    save_adaptive_state(model, state_path)
    return model


def _load_drift_state(path: Path, cfg: DriftConfig) -> ResidualDriftDetector:
    det = ResidualDriftDetector(cfg)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        det.residuals = list(map(float, raw.get("residuals", [])))
        det._persist = int(raw.get("persist", 0))
    return det


def _save_drift_state(path: Path, det: ResidualDriftDetector) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {"residuals": det.residuals[-det.cfg.window :], "persist": det._persist}
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def main(config_path: str = "configs/mmx.yaml") -> None:
    silence_noise()
    cfg = _load_cfg(config_path)
    channels = list(cfg["budget"]["channel_bounds"].keys())

    out_root = Path(cfg["data"]["out_dir"]) / "ops"
    out_dyn = Path(cfg["data"]["out_dir"]) / "dynamic"
    _ensure_dir(out_root)
    _ensure_dir(out_dyn)

    ch_path = cfg["data"]["input_channel_path"]
    if not Path(ch_path).exists():
        make_sample_inputs("data", SampleSpec(n_days=180, seed=int(cfg["model"].get("seed", 42))))

    ch_raw = pd.read_csv(ch_path, parse_dates=["date"])
    ch_raw = _sanitize_funnel_df(ch_raw)
    # Memory guard: keep only recent history for bootstrap/ops loop
    max_hist = int(cfg.get("ops_loop", {}).get("max_history_days", 240))
    if len(ch_raw) > 0:
        dmax = ch_raw["date"].max()
        ch_raw = ch_raw[ch_raw["date"] >= (dmax - pd.Timedelta(days=max_hist - 1))]

    # features (reuse original adstock+saturation)
    scmc = SCMConfig(
        seed=int(cfg["model"].get("seed", 42)),
        adstock_alpha=float(cfg["model"].get("adstock_alpha", 0.5)),
        saturation_k=float(cfg["model"].get("saturation_k", 0.00008)),
        tm_capacity_per_day=int(cfg["ops"].get("tm_capacity_per_day", 1200)),
        sla_softcap_connected=float(cfg["ops"].get("sla_softcap_connected", 0.85)),
    )
    panel = _prep_design(ch_raw, channels, scmc)
    media = _media_features(panel, channels)
    facts = ch_raw.copy()

    dates = sorted(pd.to_datetime(media["date"].unique()))

    conf = AdaptiveSCMConfig(
        channels=channels,
        tm_capacity_per_day=float(cfg["ops"].get("tm_capacity_per_day", 1200.0)),
        util_softcap=float(cfg["ops"].get("sla_softcap_connected", 0.85)),
        obs_var_leads=float(cfg.get("dynamic", {}).get("obs_var_leads", 0.20)),
        q_drift=float(cfg.get("dynamic", {}).get("q_drift", 0.01)),
        seed=int(cfg["model"].get("seed", 42)),
    )

    state_path = out_dyn / "adaptive_state.json"
    model = _load_or_bootstrap_model(conf, state_path, dates, media, facts, channels)

    # Drift detector on leads residuals
    drift_cfg = DriftConfig(
        z_threshold=float(cfg.get("ops_loop", {}).get("drift_z", 3.0)),
        window=int(cfg.get("ops_loop", {}).get("drift_window", 21)),
        persist_days=int(cfg.get("ops_loop", {}).get("drift_persist_days", 3)),
    )
    drift_state_path = out_dyn / "drift_state.json"
    drift = _load_drift_state(drift_state_path, drift_cfg)

    # Determine new days since last run
    last_date = pd.to_datetime(model.state.last_date) if model.state.last_date else None
    new_dates = [d for d in dates if (last_date is None or d > last_date)]

    logs = []
    for d in new_dates:
        t = int(model.state.t_index) + 1
        x_media = media.loc[media["date"] == d, channels].iloc[0].to_dict()
        day = facts[facts["date"] == d]
        info = model.update_day(
            date=d,
            x_media_by_ch=x_media,
            leads_total=float(day["leads"].sum()),
            tm_attempts_total=float(day["tm_attempts"].sum()),
            connected_by_ch={ch: float(day.loc[day["channel"] == ch, "tm_connected"].sum()) for ch in channels},
            contracts_by_ch={ch: float(day.loc[day["channel"] == ch, "contracts"].sum()) for ch in channels},
            premium_by_ch={ch: float(day.loc[day["channel"] == ch, "premium"].sum()) for ch in channels},
            t_index=t,
        )
        det = drift.update(float(info.get("leads_log_resid", 0.0)))
        info.update({"drift": det["drift"], "hard_drift": det.get("hard_drift", False), "drift_z": det.get("z", 0.0)})

        # If hard drift: increase process noise to adapt faster
        if det.get("hard_drift", False):
            # multiply Q for media coefficients (fast adapt)
            model.state.dlm.Q = model.state.dlm.Q * 1.5
            info["action"] = "Q_up"

        model.state.last_date = str(d.date())
        model.state.t_index = t
        logs.append(info)

    if logs:
        log_path = out_dyn / "online_fit_log.csv"
        df_new = pd.DataFrame(logs)
        if log_path.exists():
            df_old = pd.read_csv(log_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(log_path, index=False)

    save_adaptive_state(model, state_path)
    _save_drift_state(drift_state_path, drift)

    # --- Next-day budget recommendation (risk-aware + exploration) ---
    next_date = max(dates) + pd.Timedelta(days=1)
    t_next = int(model.state.t_index) + 1
    total_budget = float(cfg["budget"].get("total_budget", 1000000.0)) / 30.0
    bounds = BudgetBounds(
        min_share=float(cfg.get("dynamic", {}).get("budget_min_share", 0.05)),
        max_share=float(cfg.get("dynamic", {}).get("budget_max_share", 0.80)),
    )

    tm_attempts_next = float(cfg.get("dynamic", {}).get("tm_attempts_next_day", conf.tm_capacity_per_day * conf.util_softcap))
    base_split = {ch: total_budget / len(channels) for ch in channels}

    risk_alpha = float(cfg.get("ops_loop", {}).get("risk_alpha", 0.10))
    n_draws = int(cfg.get("ops_loop", {}).get("ppc_draws", 300))

    score_samples: Dict[str, np.ndarray] = {}
    for ch in channels:
        scales = [0.7, 1.0, 1.3]
        cvars = []
        for s in scales:
            x_media = {c: base_split[c] for c in channels}
            x_media[ch] = base_split[ch] * s
            sim = model.simulate_next_day_revenue(next_date, x_media, tm_attempts_next, t_next, n_draws=n_draws)
            # use CVaR(10%) proxy for risk-aware objective
            cvars.append(sim["rev_cvar10"])
        slope = (cvars[2] - cvars[0]) / (base_split[ch] * (scales[2] - scales[0]) + 1e-9)
        rng = np.random.default_rng(0)
        score_samples[ch] = np.clip(rng.normal(loc=slope, scale=max(abs(slope) * 0.35, 1e-6), size=800), 1e-9, None)

    # CVaR hard floor constraint (optional): reject allocations that violate downside limit
    cvar_floor_ratio = float(cfg.get("ops_loop", {}).get("cvar_floor_ratio", 0.0))
    cvar_floor_abs = float(cfg.get("ops_loop", {}).get("cvar_floor_abs", 0.0))

    # baseline risk under "no-change" allocation (equal split) as a conservative reference
    sim_floor_base = model.simulate_next_day_revenue(next_date, base_split, tm_attempts_next, t_next, n_draws=n_draws)
    floor = 0.0
    if cvar_floor_ratio and cvar_floor_ratio > 0:
        floor = max(floor, float(sim_floor_base["rev_cvar10"]) * float(cvar_floor_ratio))
    if cvar_floor_abs and cvar_floor_abs > 0:
        floor = max(floor, float(cvar_floor_abs))

    def _sample_reco(seed: int) -> Dict[str, float]:
        return recommend_budget_thompson_explore(
            channels,
            total_budget,
            score_samples,
            bounds,
            exploration_eps=float(cfg.get("ops_loop", {}).get("exploration_eps", 0.10)),
            temperature=float(cfg.get("ops_loop", {}).get("temperature", 1.0)),
            seed=seed,
        )

    best = None
    best_mean = -1e30
    n_candidates = int(cfg.get("ops_loop", {}).get("reco_candidates", 80))
    for k in range(n_candidates):
        cand = _sample_reco(seed=int(cfg["model"].get("seed", 42)) + k)
        sim = model.simulate_next_day_revenue(next_date, cand, tm_attempts_next, t_next, n_draws=n_draws)
        if floor > 0 and float(sim["rev_cvar10"]) < floor:
            continue
        if float(sim["rev_mean"]) > best_mean:
            best_mean = float(sim["rev_mean"])
            best = cand

    reco = best if best is not None else _sample_reco(seed=int(cfg["model"].get("seed", 42)))

    # --- A/B holdout integration (optional): fix exploration into CONTROL/TREATMENT to strengthen identification ---
    ab_cfg = cfg.get("ab_holdout", {})
    ab_enabled = bool(ab_cfg.get("enabled", False))
    if ab_enabled:
        holdout_channels = list(ab_cfg.get("holdout_channels", []))
        delta = float(ab_cfg.get("delta", 0.20))
        groups = assign_weekly_groups(str(next_date.date()), channels, holdout_channels, seed=int(cfg["model"].get("seed", 42)))
        reco, mults = apply_holdout_multipliers(
            reco,
            groups,
            delta=delta,
            min_share=float(cfg.get("dynamic", {}).get("budget_min_share", 0.05)),
            max_share=float(cfg.get("dynamic", {}).get("budget_max_share", 0.80)),
        )
        # Save plan for next day + append assignment history
        ab_plan = pd.DataFrame(
            [
                {
                    "date": str(next_date.date()),
                    "channel": ch,
                    "group": groups.get(ch, "NA"),
                    "multiplier": float(mults.get(ch, 1.0)),
                    "budget": float(reco[ch]),
                }
                for ch in channels
                if ch in holdout_channels
            ]
        )
        (out_root / "ab_plan_next_day.csv").write_text(ab_plan.to_csv(index=False), encoding="utf-8")

        assign_path = out_root / "ab_assignments.csv"
        if assign_path.exists():
            old = pd.read_csv(assign_path)
            newa = pd.concat([old, ab_plan[["date", "channel", "group", "multiplier"]]], ignore_index=True)
        else:
            newa = ab_plan[["date", "channel", "group", "multiplier"]]
        newa.drop_duplicates(subset=["date", "channel"], keep="last", inplace=True)
        newa.to_csv(assign_path, index=False)

    reco_df = pd.DataFrame([{"date": str(next_date.date()), "channel": ch, "budget": float(reco[ch])} for ch in channels])
    reco_df.to_csv(out_dyn / "budget_reco_next_day.csv", index=False)
    (out_dyn / "budget_reco_next_day.json").write_text(json.dumps(reco, ensure_ascii=False, indent=2), encoding="utf-8")

    # append recommendation history (for later comparison with actual spend)
    reco_hist_path = out_root / "reco_history.csv"
    if reco_hist_path.exists():
        old = pd.read_csv(reco_hist_path)
        new = pd.concat([old, reco_df], ignore_index=True)
    else:
        new = reco_df
    new.drop_duplicates(subset=["date", "channel"], keep="last", inplace=True)
    new.to_csv(reco_hist_path, index=False)

    # --- Geo holdout (matched market) spend plan (optional) ---
    geo_cfg_raw = cfg.get("geo_holdout", {})
    geo_path = geo_cfg_raw.get("input_geo_channel_path", "data/input_daily_geo_channel.csv")
    if bool(geo_cfg_raw.get("enabled", False)) and Path(geo_path).exists():
        geo_daily = pd.read_csv(geo_path, parse_dates=["date"])
        geo_cfg = GeoHoldoutConfig(
            enabled=True,
            holdout_channels=list(geo_cfg_raw.get("holdout_channels", [])),
            delta=float(geo_cfg_raw.get("delta", 0.15)),
            min_geo_share=float(geo_cfg_raw.get("min_geo_share", 0.02)),
            max_geo_share=float(geo_cfg_raw.get("max_geo_share", 0.80)),
            treated_frac=float(geo_cfg_raw.get("treated_frac", 0.50)),
            min_control_geos=int(geo_cfg_raw.get("min_control_geos", 5)),
            prewindow_days=int(geo_cfg_raw.get("prewindow_days", 56)),
            ridge_lambda=float(geo_cfg_raw.get("mm_ridge_lambda", 1.0)),
            mm_max_geos=int(geo_cfg_raw.get("mm_max_geos", 40)),
            prefit_min_r2=float(geo_cfg_raw.get("mm_prefit_min_r2", 0.50)),
            prefit_max_rmse=float(geo_cfg_raw.get("mm_prefit_max_rmse", 1e18)),
            prefit_max_mape=float(geo_cfg_raw.get("mm_prefit_max_mape", 1e18)),
            seed=int(cfg["model"].get("seed", 42)),
        )
        plan_df, assign_df = build_geo_spend_plan(str(next_date.date()), reco, geo_daily, geo_cfg)
        if len(plan_df) > 0:
            plan_df.to_csv(out_root / "geo_plan_next_day.csv", index=False)
        if len(assign_df) > 0:
            ap = out_root / "geo_assignments.csv"
            if ap.exists():
                old = pd.read_csv(ap)
                allx = pd.concat([old, assign_df], ignore_index=True)
                allx.drop_duplicates(subset=["date", "channel", "geo"], keep="last", inplace=True)
            else:
                allx = assign_df
            allx.to_csv(ap, index=False)

    # --- Performance tracking: recommended vs actual (backtest) ---
    perf_path = out_root / "performance_daily.csv"

    # We compare for the latest observed day: yesterday = max(dates)
    d_last = max(dates)
    day_last = facts[facts["date"] == d_last]
    actual_rev = float(day_last["premium"].sum())
    actual_spend = {ch: float(day_last.loc[day_last["channel"] == ch, "spend"].sum()) for ch in channels}

    # Baseline: equal split (counterfactual estimate for same day using posterior)
    # Use x_media from features for that day as baseline proxy.
    x_base = media.loc[media["date"] == d_last, channels].iloc[0].to_dict()
    sim_base = model.simulate_next_day_revenue(d_last, x_base, float(day_last["tm_attempts"].sum()), int(model.state.t_index), n_draws=n_draws)

    row = {
        "date": str(d_last.date()),
        "actual_premium": actual_rev,
        "pred_premium_mean": sim_base["rev_mean"],
        "pred_premium_cvar10": sim_base["rev_cvar10"],
        "drift_flag": bool(logs[-1]["hard_drift"]) if logs else False,
        "drift_z": float(logs[-1]["drift_z"]) if logs else 0.0,
    }
    for ch in channels:
        row[f"actual_spend_{ch}"] = actual_spend[ch]

    # add recommended spend if available for that date
    try:
        reco_hist = pd.read_csv(reco_hist_path)
        reco_day = reco_hist[reco_hist["date"] == str(d_last.date())]
        if len(reco_day) == len(channels):
            for ch in channels:
                v = float(reco_day.loc[reco_day["channel"] == ch, "budget"].iloc[0])
                row[f"reco_spend_{ch}"] = v
    except Exception:
        pass

    df_row = pd.DataFrame([row])
    if perf_path.exists():
        df_old = pd.read_csv(perf_path)
        df_all = pd.concat([df_old, df_row], ignore_index=True)
        df_all.drop_duplicates(subset=["date"], keep="last", inplace=True)
    else:
        df_all = df_row
    df_all.to_csv(perf_path, index=False)

    logger.info("Ops daily run done. next-day reco in %s", out_dyn / "budget_reco_next_day.csv")


if __name__ == "__main__":
    main()
