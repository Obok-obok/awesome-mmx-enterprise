from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from src.mmx.bayesian_scm import SCMConfig, _prep_design, _sanitize_funnel_df
from src.data.generate_sample import make_sample_inputs, SampleSpec
from src.utils.logging_utils import silence_noise
from src.mmx_dynamic.adaptive_scm import AdaptiveSCM, AdaptiveSCMConfig
from src.mmx_dynamic.budget_optimizer import BudgetBounds, recommend_budget_thompson


logger = logging.getLogger(__name__)


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _media_features(panel: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    # Expect columns: date, channel, spend, x_media (if built). If missing, use spend.
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


def main(config_path: str = "configs/mmx.yaml") -> None:
    silence_noise()
    cfg = _load_cfg(config_path)
    channels = list(cfg["budget"]["channel_bounds"].keys())
    out_root = Path(cfg["data"]["out_dir"]) / "dynamic"
    _ensure_out(out_root)

    ch_path = cfg["data"]["input_channel_path"]
    if not Path(ch_path).exists():
        make_sample_inputs("data", SampleSpec(n_days=180, seed=int(cfg["model"].get("seed", 42))))

    # Load daily channel file (same one used by the original SCM)
    ch_raw = pd.read_csv(ch_path, parse_dates=["date"])
    ch_raw = _sanitize_funnel_df(ch_raw)

    # Reuse original adstock+saturation feature engineering
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

    # Aggregate to day-level dicts
    dates = pd.to_datetime(sorted(media["date"].unique()))
    conf = AdaptiveSCMConfig(
        channels=channels,
        tm_capacity_per_day=float(cfg["ops"].get("tm_capacity_per_day", 1200.0)),
        util_softcap=float(cfg["ops"].get("sla_softcap_connected", 0.85)),
        obs_var_leads=float(cfg.get("dynamic", {}).get("obs_var_leads", 0.20)),
        q_drift=float(cfg.get("dynamic", {}).get("q_drift", 0.01)),
        seed=int(cfg["model"].get("seed", 42)),
    )
    model = AdaptiveSCM(conf)

    logs = []
    state_rows = []

    for t, d in enumerate(dates):
        x_media = media.loc[media["date"] == d, channels].iloc[0].to_dict()

        day = facts[facts["date"] == d]
        leads_total = float(day["leads"].sum())
        tm_attempts_total = float(day["tm_attempts"].sum())

        connected_by = {ch: float(day.loc[day["channel"] == ch, "tm_connected"].sum()) for ch in channels}
        contracts_by = {ch: float(day.loc[day["channel"] == ch, "contracts"].sum()) for ch in channels}
        premium_by = {ch: float(day.loc[day["channel"] == ch, "premium"].sum()) for ch in channels}

        info = model.update_day(
            date=d,
            x_media_by_ch=x_media,
            leads_total=leads_total,
            tm_attempts_total=tm_attempts_total,
            connected_by_ch=connected_by,
            contracts_by_ch=contracts_by,
            premium_by_ch=premium_by,
            t_index=t,
        )
        logs.append(info)

        # Persist a light state snapshot (means only)
        row = {"date": d.date().isoformat()}
        row.update({f"theta_mean_{i}": float(model.state.dlm.m[i]) for i in range(model.k)})
        state_rows.append(row)

    pd.DataFrame(logs).to_csv(out_root / "online_fit_log.csv", index=False)
    pd.DataFrame(state_rows).to_csv(out_root / "dlm_state_means.csv", index=False)

    # --- Budget recommendation for next day (short-horizon) ---
    next_date = dates.max() + pd.Timedelta(days=1)
    t_next = len(dates)
    total_budget = float(cfg["budget"].get("total_budget", 1000000.0)) / 30.0
    bounds = BudgetBounds(
        min_share=float(cfg.get("dynamic", {}).get("budget_min_share", 0.05)),
        max_share=float(cfg.get("dynamic", {}).get("budget_max_share", 0.80)),
    )

    # Build score samples by simulating marginal returns around baseline split
    base_split = {ch: total_budget / len(channels) for ch in channels}
    tm_attempts_next = float(cfg.get("dynamic", {}).get("tm_attempts_next_day", conf.tm_capacity_per_day * conf.util_softcap))

    score_samples: Dict[str, np.ndarray] = {}
    for ch in channels:
        # 3-point finite difference: 0.7x, 1.0x, 1.3x for that channel only
        scales = [0.7, 1.0, 1.3]
        revs = []
        for s in scales:
            x_media = {c: base_split[c] for c in channels}
            x_media[ch] = base_split[ch] * s
            sim = model.simulate_next_day_revenue(next_date, x_media, tm_attempts_next, t_next, n_draws=200)
            revs.append(sim["rev_mean"])
        # approximate slope around 1.0
        slope = (revs[2] - revs[0]) / (base_split[ch] * (scales[2] - scales[0]) + 1e-9)
        # create pseudo posterior samples around slope with small noise
        score_samples[ch] = np.clip(np.random.default_rng(0).normal(loc=slope, scale=max(abs(slope) * 0.25, 1e-6), size=500), 1e-9, None)

    reco = recommend_budget_thompson(channels, total_budget, score_samples, bounds)
    pd.DataFrame([{"channel": k, "budget": v} for k, v in reco.items()]).to_csv(out_root / "budget_reco_next_day.csv", index=False)

    with open(out_root / "budget_reco_next_day.json", "w", encoding="utf-8") as f:
        json.dump(reco, f, ensure_ascii=False, indent=2)

    logger.info("Dynamic online run completed. Outputs in %s", out_root)


if __name__ == "__main__":
    main()
