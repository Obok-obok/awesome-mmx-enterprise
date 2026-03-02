from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


CHANNELS = ["google", "naver", "meta", "toss", "kakao"]
MESSAGE_TYPES = ["SA", "DA"]


def adstock(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + alpha * carry
        out[i] = carry
    return out


def saturation(x: np.ndarray, k: float) -> np.ndarray:
    return np.log1p(k * np.maximum(x, 0.0))


@dataclass
class SampleSpec:
    n_days: int = 180
    seed: int = 42
    n_campaigns_per_channel: int = 6
    tm_capacity: int = 1200
    contract_window_days: int = 30


def make_sample_inputs(out_dir: str, spec: SampleSpec = SampleSpec()) -> dict[str, str]:
    """Generate a realistic demo dataset for insurance Web→DB→TM→Contract.

    Outputs:
      - input_daily_channel.csv (date×channel)
      - input_daily_campaign.csv (date×channel×campaign×message)
      - raw_events.csv (lead-level, includes customer_id / lead_id / policy_id)
      - raw_leads.csv / raw_tm_calls.csv / raw_policies.csv (ops-ready raw tables for customer_id mapping demo)
      - spend_campaign.csv (date×channel×campaign×message spend)
    """

    rng = np.random.default_rng(spec.seed)
    start = date(2025, 8, 1)
    dates = [start + timedelta(days=i) for i in range(spec.n_days)]
    dow = np.array([d.weekday() for d in dates])

    # Channel base intensity
    base = {"google": 1.25, "naver": 1.05, "meta": 0.75, "toss": 0.45, "kakao": 0.55}
    season = 1.0 + 0.10 * np.sin(np.linspace(0, 6 * np.pi, spec.n_days))
    trend = np.linspace(0.9, 1.1, spec.n_days)

    # Spend → Leads elasticities (channel-level)
    beta_leads = {"google": 0.70, "naver": 0.60, "meta": 0.38, "toss": 0.28, "kakao": 0.32}
    # Lead quality multiplier (affects close rate)
    qual = {"google": 1.03, "naver": 1.00, "meta": 0.96, "toss": 1.12, "kakao": 0.98}

    # Message type effects (campaign-level): SA tends to yield higher intent, DA higher volume
    msg_lead_mult = {"SA": 0.92, "DA": 1.08}
    msg_close_mult = {"SA": 1.10, "DA": 0.93}

    # Campaign random effects within channel
    camp_re = {}
    campaigns = []
    for ch in CHANNELS:
        for j in range(spec.n_campaigns_per_channel):
            camp = f"{ch[:2].upper()}_C{j+1:02d}"
            # campaign effectiveness around 1.0
            camp_re[camp] = float(np.exp(rng.normal(0, 0.18)))
            campaigns.append((ch, camp))

    # Allocate spend across campaigns and messages each day
    rows_campaign_daily = []
    for ch in CHANNELS:
        # Weekend effect
        w_mult = np.where(dow >= 5, 0.85, 1.0)
        spend_ch = base[ch] * season * trend * w_mult * (1.0 + rng.normal(0, 0.08, spec.n_days))
        spend_ch = np.clip(spend_ch, 0.1, None) * 1_000_000

        # Split channel spend among campaigns (fixed weights + small daily noise)
        camp_list = [c for (cch, c) in campaigns if cch == ch]
        base_w = rng.dirichlet(np.ones(len(camp_list)) * 1.2)
        for i, d in enumerate(dates):
            w = np.clip(base_w + rng.normal(0, 0.02, len(camp_list)), 0.001, None)
            w = w / w.sum()

            # Message split (SA vs DA)
            msg_w = np.array([0.45, 0.55]) + rng.normal(0, 0.03, 2)
            msg_w = np.clip(msg_w, 0.05, None)
            msg_w = msg_w / msg_w.sum()

            for camp_idx, camp in enumerate(camp_list):
                for mt_idx, mt in enumerate(MESSAGE_TYPES):
                    spend = float(spend_ch[i] * w[camp_idx] * msg_w[mt_idx])
                    rows_campaign_daily.append(
                        {
                            "date": d,
                            "channel": ch,
                            "campaign_id": camp,
                            "message_type": mt,
                            "spend": int(round(spend)),
                        }
                    )

    df_camp = pd.DataFrame(rows_campaign_daily)

    # Generate campaign-level leads using adstock+saturation per campaign×message
    df_camp = df_camp.sort_values(["campaign_id", "message_type", "date"]).reset_index(drop=True)

    df_camp["leads"] = 0
    for (camp, mt), g in df_camp.groupby(["campaign_id", "message_type"], sort=False):
        x = saturation(adstock(g["spend"].to_numpy(dtype=float), 0.55), 0.00009)
        ch = g["channel"].iloc[0]
        # day-of-week seasonality term
        dow_local = np.array([pd.Timestamp(d).weekday() for d in g["date"].to_list()])
        mu = np.exp(
            2.4
            + beta_leads[ch] * x
            + 0.14 * np.sin(2 * np.pi * dow_local / 7)
        )
        mu = mu * camp_re[camp] * msg_lead_mult[mt]
        leads = rng.poisson(mu)
        df_camp.loc[g.index, "leads"] = leads.astype(int)

    # Build raw lead-level events with customer_id to enable campaign→contract mapping
    raw_rows = []
    lead_seq = 0
    customer_seq = 10_000_000
    for _, r in df_camp.iterrows():
        d = pd.to_datetime(r["date"]).date()
        ch = r["channel"]
        camp = r["campaign_id"]
        mt = r["message_type"]
        n_leads = int(r["leads"])
        for _k in range(n_leads):
            lead_seq += 1
            customer_seq += 1
            lead_id = f"L{d:%Y%m%d}_{lead_seq:06d}"
            customer_id = f"U{customer_seq}"
            raw_rows.append(
                {
                    "lead_id": lead_id,
                    "customer_id": customer_id,
                    "lead_date": d,
                    "channel": ch,
                    "campaign_id": camp,
                    "message_type": mt,
                }
            )

    raw = pd.DataFrame(raw_rows)

    # Simulate TM attempts / connections / contracts at lead-level with capacity constraints
    # Capacity acts at day level: total attempts capped; excess leads delayed (response delay) causing lower connect/close.
    # Create demand = leads, choose attempts up to capacity*0.60 per day.
    if len(raw) == 0:
        raise RuntimeError("Sample generation failed: no leads generated")

    # Determine per-lead attempt probability; DA tends to have slightly lower answer rate.
    base_attempt_p = 0.95
    base_connect_p = 0.42
    base_close_p = 0.12

    raw["attempted"] = 0
    raw["connected"] = 0
    raw["contracted"] = 0
    raw["policy_id"] = ""
    raw["contract_date"] = pd.NaT
    raw["premium"] = 0

    prem_mu = {"google": 52000, "naver": 50000, "meta": 47000, "toss": 61000, "kakao": 49000}

    for d, g in raw.groupby("lead_date", sort=True):
        idx = g.index.to_numpy()
        # Attempt capacity
        cap = int(spec.tm_capacity * 0.60)
        n = len(idx)
        # Choose which leads get attempted today
        # Attempt propensity depends on message_type (SA slightly prioritized)
        prio = g["message_type"].map({"SA": 1.05, "DA": 0.95}).to_numpy(dtype=float)
        prio = prio / prio.sum()
        m = min(cap, n)
        attempted_idx = rng.choice(idx, size=m, replace=False, p=prio)
        raw.loc[attempted_idx, "attempted"] = 1

        util = m / cap if cap > 0 else 1.0
        util_over = max(util - 0.85, 0.0)
        # Connection probability decreases if utilization is high (queueing / response delay proxy)
        # DA worse answer rate
        p_conn = base_connect_p * np.exp(-1.8 * util_over)
        p_conn_vec = (
            p_conn
            * g.loc[attempted_idx, "message_type"].map({"SA": 1.04, "DA": 0.93}).to_numpy(dtype=float)
        )
        p_conn_vec = np.clip(p_conn_vec, 0.05, 0.70)
        connected = rng.binomial(1, p_conn_vec)
        raw.loc[attempted_idx, "connected"] = connected

        # Close probability depends on channel quality and message type, slightly impacted by utilization
        p_close = base_close_p * np.exp(-0.9 * util_over)
        p_close_vec = (
            p_close
            * g.loc[attempted_idx, "channel"].map(qual).to_numpy(dtype=float)
            * g.loc[attempted_idx, "message_type"].map(msg_close_mult).to_numpy(dtype=float)
        )
        p_close_vec = np.clip(p_close_vec, 0.02, 0.28)
        contracted = rng.binomial(1, p_close_vec) * raw.loc[attempted_idx, "connected"].to_numpy(dtype=int)
        raw.loc[attempted_idx, "contracted"] = contracted

        # Assign policy ids and premiums with a random contract date within window
        contracted_idx = attempted_idx[contracted.astype(bool)]
        if len(contracted_idx) > 0:
            offsets = rng.integers(0, spec.contract_window_days + 1, size=len(contracted_idx))
            cdates = [pd.Timestamp(d + timedelta(days=int(o))) for o in offsets]
            raw.loc[contracted_idx, "contract_date"] = cdates
            raw.loc[contracted_idx, "policy_id"] = [
                f"P{pd.Timestamp(cd).strftime('%Y%m%d')}_{rng.integers(100000,999999)}" for cd in cdates
            ]
            prem = (
                raw.loc[contracted_idx, "channel"].map(prem_mu).astype(float)
                * rng.normal(1.0, 0.12, len(contracted_idx))
            ).clip(10000)
            raw.loc[contracted_idx, "premium"] = prem.round(0).astype(int)

    # Build ops-style raw tables
    raw_leads = raw[["lead_id", "customer_id", "lead_date", "channel", "campaign_id", "message_type"]].copy()
    raw_leads = raw_leads.rename(columns={"lead_date": "lead_ts"})
    raw_leads["lead_ts"] = pd.to_datetime(raw_leads["lead_ts"]) + pd.to_timedelta(rng.integers(8, 20, len(raw_leads)), unit="h")

    # TM calls: one call per attempted lead (simplified)
    attempted = raw[raw["attempted"] == 1].copy()
    tm_calls = attempted[["lead_id", "customer_id", "lead_date", "channel", "campaign_id", "message_type", "connected"]].copy()
    tm_calls = tm_calls.rename(columns={"lead_date": "call_ts", "connected": "connected_flag"})
    tm_calls["call_ts"] = pd.to_datetime(tm_calls["call_ts"]) + pd.to_timedelta(rng.integers(9, 21, len(tm_calls)), unit="h")
    tm_calls["call_id"] = [f"C{1000000+i}" for i in range(len(tm_calls))]
    tm_calls = tm_calls[["call_id", "customer_id", "call_ts", "connected_flag", "channel", "campaign_id", "message_type"]]

    # Policies raw
    pol = raw[raw["contracted"] == 1].copy()
    raw_policies = pol[["policy_id", "customer_id", "contract_date", "premium"]].copy()
    raw_policies = raw_policies.rename(columns={"contract_date": "contract_ts"})
    raw_policies["contract_ts"] = pd.to_datetime(raw_policies["contract_ts"]) + pd.to_timedelta(rng.integers(10, 18, len(raw_policies)), unit="h")

    # Aggregate to daily campaign panel (the modeling input)
    # tm_attempts = attempted leads
    # tm_connected = connected
    # contracts/premium based on contract_date, but attributed back to original campaign via customer_id mapping.
    # In ops, you'd enforce a window and matching rule; demo uses the generated mapping directly.
    df_camp2 = df_camp.copy()
    df_camp2["tm_attempts"] = 0
    df_camp2["tm_connected"] = 0
    df_camp2["contracts"] = 0
    df_camp2["premium"] = 0

    raw_attempts = (
        raw.groupby(["lead_date", "channel", "campaign_id", "message_type"], as_index=False)
        .agg(tm_attempts=("attempted", "sum"), tm_connected=("connected", "sum"))
        .rename(columns={"lead_date": "date"})
    )
    df_camp2 = df_camp2.merge(
        raw_attempts,
        on=["date", "channel", "campaign_id", "message_type"],
        how="left",
        suffixes=("", "_y"),
    )
    df_camp2["tm_attempts"] = df_camp2["tm_attempts_y"].fillna(0).astype(int)
    df_camp2["tm_connected"] = df_camp2["tm_connected_y"].fillna(0).astype(int)
    df_camp2 = df_camp2.drop(columns=["tm_attempts_y", "tm_connected_y"])

    # Contracts attributed to original campaign, but counted on contract_date
    contracts_attr = raw[raw["contracted"] == 1].copy()
    contracts_attr["date"] = contracts_attr["contract_date"].dt.date
    contracts_attr = (
        contracts_attr.groupby(["date", "channel", "campaign_id", "message_type"], as_index=False)
        .agg(contracts=("policy_id", "nunique"), premium=("premium", "sum"))
    )
    df_camp2 = df_camp2.merge(
        contracts_attr,
        on=["date", "channel", "campaign_id", "message_type"],
        how="left",
        suffixes=("", "_c"),
    )
    df_camp2["contracts"] = df_camp2["contracts_c"].fillna(0).astype(int)
    df_camp2["premium"] = df_camp2["premium_c"].fillna(0).astype(int)
    df_camp2 = df_camp2.drop(columns=["contracts_c", "premium_c"])

    # Channel daily is an aggregation of campaign daily
    df_ch = (
        df_camp2.groupby(["date", "channel"], as_index=False)
        .agg(
            spend=("spend", "sum"),
            leads=("leads", "sum"),
            tm_attempts=("tm_attempts", "sum"),
            tm_connected=("tm_connected", "sum"),
            contracts=("contracts", "sum"),
            premium=("premium", "sum"),
        )
        .sort_values(["date", "channel"])
        .reset_index(drop=True)
    )

    os.makedirs(out_dir, exist_ok=True)
    p_ch = os.path.join(out_dir, "input_daily_channel.csv")
    p_camp = os.path.join(out_dir, "input_daily_campaign.csv")
    p_raw = os.path.join(out_dir, "raw_events.csv")
    p_raw_leads = os.path.join(out_dir, "raw_leads.csv")
    p_raw_tm = os.path.join(out_dir, "raw_tm_calls.csv")
    p_raw_pol = os.path.join(out_dir, "raw_policies.csv")
    p_spend_camp = os.path.join(out_dir, "spend_campaign.csv")
    df_ch.to_csv(p_ch, index=False)
    df_camp2.to_csv(p_camp, index=False)
    raw.to_csv(p_raw, index=False)
    raw_leads.to_csv(p_raw_leads, index=False)
    tm_calls.to_csv(p_raw_tm, index=False)
    raw_policies.to_csv(p_raw_pol, index=False)
    df_camp[["date", "channel", "campaign_id", "message_type", "spend"]].to_csv(p_spend_camp, index=False)

    return {
        "channel": p_ch,
        "campaign": p_camp,
        "raw": p_raw,
        "raw_leads": p_raw_leads,
        "raw_tm_calls": p_raw_tm,
        "raw_policies": p_raw_pol,
        "spend_campaign": p_spend_camp,
    }
