from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from src.data_quality.gates import run_data_quality


@dataclass
class MappingSpec:
    window_days: int = 30
    dedupe_hours: int = 24


def _to_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)


def dedupe_leads(leads: pd.DataFrame, spec: MappingSpec) -> pd.DataFrame:
    """Deduplicate leads that are effectively the same event.

    Rule (practical default):
      - Within `dedupe_hours` for the same (customer_id, channel, campaign_id, message_type)
        keep the earliest lead and drop the rest.
    """
    leads = leads.copy()
    leads["lead_ts"] = _to_ts(leads["lead_ts"])
    leads = leads.sort_values(["customer_id", "channel", "campaign_id", "message_type", "lead_ts"])

    keep = np.ones(len(leads), dtype=bool)
    last_key = None
    last_ts = None
    for i, r in enumerate(leads.itertuples(index=False)):
        key = (r.customer_id, r.channel, r.campaign_id, r.message_type)
        ts = r.lead_ts
        if last_key == key and pd.notna(ts) and pd.notna(last_ts):
            if ts - last_ts <= pd.Timedelta(hours=spec.dedupe_hours):
                keep[i] = False
            else:
                last_ts = ts
        else:
            last_key = key
            last_ts = ts
    return leads.loc[keep].reset_index(drop=True)


def map_policy_to_lead(
    leads: pd.DataFrame,
    policies: pd.DataFrame,
    spec: MappingSpec,
) -> pd.DataFrame:
    """Map each policy to the most recent eligible lead (last-touch) within window.

    Handles:
      - 30-day attribution window
      - Multiple leads per customer (choose most recent before contract)
      - Multiple policies per customer (map each policy independently)
    """
    leads = leads.copy()
    policies = policies.copy()
    leads["lead_ts"] = _to_ts(leads["lead_ts"])
    policies["contract_ts"] = _to_ts(policies["contract_ts"])

    leads = leads.dropna(subset=["customer_id", "lead_ts"]).sort_values(["customer_id", "lead_ts"])
    leads = leads.sort_values(["customer_id", "lead_ts"], kind="mergesort").reset_index(drop=True)

    policies = policies.dropna(subset=["customer_id", "contract_ts", "policy_id"]).copy()
    policies = policies.sort_values(["customer_id", "contract_ts"], kind="mergesort").reset_index(drop=True)

    # Pandas merge_asof has strict global sorting requirements; implement a robust per-customer matcher.
    win = pd.Timedelta(days=spec.window_days)
    out_rows = []

    leads_g = {cid: g for cid, g in leads.groupby("customer_id", sort=False)}
    for cid, pol_g in policies.groupby("customer_id", sort=False):
        ldg = leads_g.get(cid)
        if ldg is None or len(ldg) == 0:
            continue
        lts = ldg["lead_ts"].to_numpy(dtype="datetime64[ns]")
        # Ensure sorted
        order = np.argsort(lts)
        ldg = ldg.iloc[order]
        lts = ldg["lead_ts"].to_numpy(dtype="datetime64[ns]")

        pol_g = pol_g.sort_values("contract_ts", kind="mergesort")
        for r in pol_g.itertuples(index=False):
            ct = r.contract_ts
            if pd.isna(ct):
                continue
            # last lead <= contract
            pos = np.searchsorted(lts, np.datetime64(ct), side="right") - 1
            if pos < 0:
                continue
            lead_row = ldg.iloc[int(pos)]
            age = pd.Timestamp(ct) - pd.Timestamp(lead_row["lead_ts"])
            if age < pd.Timedelta(0) or age > win:
                continue
            out = {
                **pol_g.loc[pol_g.index[0]].to_dict(),  # dummy to get columns? will overwrite
            }
            # safer: build dict from r + lead_row
            out = {
                "policy_id": getattr(r, "policy_id"),
                "customer_id": cid,
                "contract_ts": ct,
                "premium": getattr(r, "premium"),
                "lead_ts": lead_row["lead_ts"],
                "channel": lead_row.get("channel"),
                "campaign_id": lead_row.get("campaign_id"),
                "message_type": lead_row.get("message_type"),
            }
            out["attribution_window_days"] = int(age.days)
            out_rows.append(out)

    mapped = pd.DataFrame(out_rows)
    return mapped


def aggregate_inputs(
    mapped_policies: pd.DataFrame,
    tm_calls: pd.DataFrame,
    spend_campaign: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily campaign/channel inputs from mapped policies and TM calls.

    Notes:
      - TM calls are attributed to the most recent lead BEFORE the call within 30 days is recommended,
        but for simplicity we attribute calls via lead_id if present; else customer_id last-touch.
      - In this demo builder, we assume tm_calls already contain campaign_id/message_type via lead join.
    """

    # Campaign-level contracts/premium on contract_date
    mapped_policies["date"] = mapped_policies["contract_ts"].dt.date
    camp_contracts = (
        mapped_policies.groupby(["date", "channel", "campaign_id", "message_type"], as_index=False)
        .agg(contracts=("policy_id", "nunique"), premium=("premium", "sum"))
        .astype({"contracts": int, "premium": int})
    )

    # Leads on lead_date
    leads_daily = (
        mapped_policies.assign(lead_date=mapped_policies["lead_ts"].dt.date)
        .groupby(["lead_date", "channel", "campaign_id", "message_type"], as_index=False)
        .agg(leads=("customer_id", "count"))
        .rename(columns={"lead_date": "date"})
        .astype({"leads": int})
    )

    # TM calls: expect customer_id, call_ts, result, connected_flag
    tm_calls = tm_calls.copy()
    tm_calls["call_ts"] = _to_ts(tm_calls["call_ts"])
    tm_calls["date"] = tm_calls["call_ts"].dt.date
    # If tm_calls already have channel/campaign/message columns, use them.
    required = {"channel", "campaign_id", "message_type"}
    if not required.issubset(set(tm_calls.columns)):
        # Best-effort: use mapping from latest lead before call by customer.
        lead_cols = ["customer_id", "lead_ts", "channel", "campaign_id", "message_type"]
        leads_for_call = (
            mapped_policies[lead_cols]
            .drop_duplicates(subset=lead_cols)
            .sort_values(["customer_id", "lead_ts"], kind="mergesort")
            .reset_index(drop=True)
        )
        tm_calls = tm_calls.sort_values(["customer_id", "call_ts"], kind="mergesort").reset_index(drop=True)
        tm_calls = pd.merge_asof(
            tm_calls,
            leads_for_call,
            by="customer_id",
            left_on="call_ts",
            right_on="lead_ts",
            direction="backward",
            allow_exact_matches=True,
        )

    tm_daily = (
        tm_calls.groupby(["date", "channel", "campaign_id", "message_type"], as_index=False)
        .agg(
            tm_attempts=("call_id", "count"),
            tm_connected=("connected_flag", "sum"),
        )
        .fillna(0)
    )

    # Build campaign panel: outer merge then fill missing with 0
    df_camp = leads_daily.merge(tm_daily, on=["date", "channel", "campaign_id", "message_type"], how="outer")
    df_camp = df_camp.merge(camp_contracts, on=["date", "channel", "campaign_id", "message_type"], how="outer")
    df_camp = df_camp.fillna(0)

    for c in ["leads", "tm_attempts", "tm_connected", "contracts", "premium"]:
        if c in df_camp.columns:
            df_camp[c] = df_camp[c].astype(int)

    # Spend: if provided, merge; else set to 0
    if spend_campaign is not None:
        spend_campaign = spend_campaign.copy()
        spend_campaign["date"] = pd.to_datetime(spend_campaign["date"], errors="coerce").dt.date
        df_camp = df_camp.merge(
            spend_campaign[["date", "channel", "campaign_id", "message_type", "spend"]],
            on=["date", "channel", "campaign_id", "message_type"],
            how="left",
        )
        df_camp["spend"] = df_camp["spend"].fillna(0).astype(int)
    else:
        df_camp["spend"] = 0

    # Channel daily aggregation
    df_ch = (
        df_camp.groupby(["date", "channel"], as_index=False)
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

    return df_ch, df_camp


def build_inputs_from_raw(
    raw_dir: str,
    out_dir: str,
    spend_campaign_path: str | None,
    window_days: int = 30,
    dedupe_hours: int = 24,
) -> dict[str, str]:
    """Entry point: read raw tables, apply mapping rules, run DQ, write input csvs."""
    spec = MappingSpec(window_days=window_days, dedupe_hours=dedupe_hours)

    p_leads = os.path.join(raw_dir, "raw_leads.csv")
    p_tm = os.path.join(raw_dir, "raw_tm_calls.csv")
    p_pol = os.path.join(raw_dir, "raw_policies.csv")

    leads = pd.read_csv(p_leads)
    tm_calls = pd.read_csv(p_tm)
    policies = pd.read_csv(p_pol)

    # Normalize expected column names
    leads = leads.rename(columns={"lead_created_at": "lead_ts", "lead_date": "lead_ts"})
    policies = policies.rename(columns={"contract_date": "contract_ts"})

    # Minimal required columns
    need_leads = {"customer_id", "lead_ts", "channel", "campaign_id", "message_type"}
    need_pol = {"customer_id", "policy_id", "contract_ts", "premium"}
    need_tm = {"customer_id", "call_id", "call_ts", "connected_flag"}
    missing = (need_leads - set(leads.columns)) | (need_pol - set(policies.columns)) | (need_tm - set(tm_calls.columns))
    if missing:
        raise ValueError(f"Missing required columns in raw inputs: {sorted(missing)}")

    # Dedupe leads
    leads = dedupe_leads(leads, spec)

    # Map policies to leads (last-touch within window)
    mapped = map_policy_to_lead(leads, policies, spec)

    spend_campaign = None
    if spend_campaign_path:
        spend_campaign = pd.read_csv(spend_campaign_path)

    df_ch, df_camp = aggregate_inputs(mapped, tm_calls, spend_campaign)

    os.makedirs(out_dir, exist_ok=True)
    p_out_ch = os.path.join(out_dir, "input_daily_channel.csv")
    p_out_camp = os.path.join(out_dir, "input_daily_campaign.csv")

    df_ch.to_csv(p_out_ch, index=False)
    df_camp.to_csv(p_out_camp, index=False)

    # Stronger data quality gate on produced inputs
    dq = run_data_quality(df_ch, df_camp)
    with open(os.path.join(out_dir, "data_quality_report.json"), "w", encoding="utf-8") as f:
        f.write(dq)

    return {"channel": p_out_ch, "campaign": p_out_camp}
