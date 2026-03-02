from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class SCMConfig:
    draws: int = 400
    tune: int = 400
    chains: int = 2
    target_accept: float = 0.9
    seed: int = 42
    adstock_alpha: float = 0.5
    saturation_k: float = 0.00008
    tm_capacity_per_day: int = 1200
    sla_softcap_connected: float = 0.85
    # Fast-mode options
    inference: str = "mcmc"  # "mcmc" | "advi"
    advi_steps: int = 2000
    advi_draws: int = 300

def _sanitize_funnel_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize funnel metrics to avoid invalid likelihoods.

    - Coerce to numeric, fill missing with 0
    - Clip negatives to 0
    - Enforce funnel constraints:
        tm_attempts >= tm_connected >= contracts
      by adjusting upward on the left (minimally) so observed is feasible.
    - Premium is clipped to >=0 (and later only used when contracts>0).
    """
    df = df.copy()
    numeric_cols = ["spend","leads","tm_attempts","tm_connected","contracts","premium"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Fill missing with 0 (counts/spend), keep premium missing as 0 as well for stability
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Clip negatives
    for c in numeric_cols:
        df[c] = np.clip(df[c].astype(float), 0.0, None)

    # Cast integer-ish columns safely
    for c in ["leads","tm_attempts","tm_connected","contracts"]:
        df[c] = np.floor(df[c]).astype(int)

    # Enforce funnel feasibility
    df["tm_connected"] = np.minimum(df["tm_connected"], df["tm_attempts"])
    df["contracts"] = np.minimum(df["contracts"], df["tm_connected"])
    # If upstream was smaller than downstream due to data issues, lift upstream minimally
    df["tm_attempts"] = np.maximum(df["tm_attempts"], df["tm_connected"])
    df["tm_connected"] = np.maximum(df["tm_connected"], df["contracts"])

    # Premium cannot be negative
    df["premium"] = df["premium"].astype(float)
    df["premium"] = np.clip(df["premium"], 0.0, None)

    return df


def _prep_design(panel: pd.DataFrame, channels: list[str], cfg: SCMConfig) -> pd.DataFrame:
    from .transforms import adstock, saturation
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","channel"]).reset_index(drop=True)
    df["t"] = (df["date"] - df["date"].min()).dt.days.astype(int)
    df["dow"] = df["date"].dt.dayofweek.astype(int)

    x_series = []
    for ch in channels:
        idx = df.index[df["channel"]==ch]
        s = df.loc[idx, "spend"].values.astype(float)
        x = saturation(adstock(s, cfg.adstock_alpha), cfg.saturation_k)
        x_series.append(pd.Series(x, index=idx))
    df["x_media"] = pd.concat(x_series).sort_index().values

    daily = df.groupby("date", as_index=False).agg(tm_attempts_total=("tm_attempts","sum"))
    daily["util"] = daily["tm_attempts_total"] / float(cfg.tm_capacity_per_day)
    df = df.merge(daily[["date","util"]], on="date", how="left")
    return df

def _run_inference(model, cfg: SCMConfig):
    """Run posterior inference.

    - mcmc: NUTS sampling (slower, more accurate)
    - advi: Variational inference (very fast, approximate)
    """
    import pymc as pm

    if (cfg.inference or "mcmc").lower() == "advi":
        approx = pm.fit(n=int(cfg.advi_steps), method="advi", progressbar=False)
        trace = approx.sample(int(cfg.advi_draws))
        idata = pm.to_inference_data(trace=trace, model=model)
        return idata

    # default: MCMC
    idata = pm.sample(
        draws=int(cfg.draws),
        tune=int(cfg.tune),
        chains=int(cfg.chains),
        target_accept=float(cfg.target_accept),
        random_seed=int(cfg.seed),
        progressbar=False,
        init="jitter+adapt_diag",
        jitter_max_retries=10,
    )
    return idata


def fit_bayesian_funnel_scm(panel: pd.DataFrame, cfg: SCMConfig):
    import pymc as pm

    channels = sorted(panel["channel"].unique().tolist())
    df = _prep_design(panel, channels, cfg)
    df = _sanitize_funnel_df(df)

    ch_to_idx = {c:i for i,c in enumerate(channels)}
    ch_idx = df["channel"].map(ch_to_idx).values.astype(int)
    n_ch = len(channels)

    dow = df["dow"].values.astype(int)
    t = df["t"].values.astype(float)
    x_media = df["x_media"].values.astype(float)

    leads = df["leads"].values.astype(int)
    attempts = df["tm_attempts"].values.astype(int)
    connected = df["tm_connected"].values.astype(int)
    contracts = df["contracts"].values.astype(int)
    premium = df["premium"].values.astype(float)

    util = df["util"].values.astype(float)
    util_over = np.maximum(util - cfg.sla_softcap_connected, 0.0)

    with pm.Model() as model:
        # 1) Leads
        a0 = pm.Normal("a0", 0.0, 2.0)
        a_ch = pm.Normal("a_ch", 0.0, 0.7, shape=n_ch)
        b_ch = pm.Normal("b_ch", 0.0, 0.7, shape=n_ch)
        a_dow = pm.Normal("a_dow", 0.0, 0.3, shape=7)
        a_trend = pm.Normal("a_trend", 0.0, 0.01)
        mu_leads = pm.math.exp(a0 + a_ch[ch_idx] + b_ch[ch_idx]*x_media + a_dow[dow] + a_trend*t)
        pm.Poisson("leads_obs", mu=mu_leads, observed=leads)

        # 2) Connected
        c0 = pm.Normal("c0", 0.0, 2.0)
        c1 = pm.Normal("c1", 0.0, 0.6)
        c2 = pm.Normal("c2", -2.0, 1.0)
        log_leads = pm.math.log(leads + 1)
        p_conn = pm.Deterministic("p_conn", pm.math.sigmoid(c0 + c1*log_leads + c2*util_over))
        pm.Binomial("conn_obs", n=attempts, p=p_conn, observed=connected)

        # 3) Contracts
        k0 = pm.Normal("k0", 0.0, 2.0)
        k_ch = pm.Normal("k_ch", 0.0, 0.6, shape=n_ch)
        k1 = pm.Normal("k1", 0.0, 0.5)
        p_close = pm.Deterministic("p_close", pm.math.sigmoid(k0 + k_ch[ch_idx] + k1*log_leads))
        pm.Binomial("cont_obs", n=connected, p=p_close, observed=contracts)

        # 4) Premium per contract (hurdle: only when contracts>0)
        eps = 1e-6
        mask = contracts > 0
        prem_per = premium / (np.maximum(contracts, 1) + eps)
        prem_per = np.clip(prem_per, 1.0, None)

        m0 = pm.Normal("m0", np.log(50000), 0.5)
        m_ch = pm.Normal("m_ch", 0.0, 0.25, shape=n_ch)
        sigma = pm.HalfNormal("sigma", 0.4)
        mu_log = m0 + m_ch[ch_idx]

        # Use only valid rows to avoid lognormal on zeros
        pm.LogNormal("prem_obs", mu=mu_log[mask], sigma=sigma, observed=prem_per[mask])

        idata = _run_inference(model, cfg)

    return idata, df, channels


def _prep_design_campaign(camp_panel: pd.DataFrame, cfg: SCMConfig) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Prepare campaign-level design matrix.

    Expects columns:
      date, channel, campaign_id, message_type, spend, leads, tm_attempts, tm_connected, contracts, premium
    """
    from .transforms import adstock, saturation

    df = camp_panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "channel", "campaign_id", "message_type"]).reset_index(drop=True)
    df["t"] = (df["date"] - df["date"].min()).dt.days.astype(int)
    df["dow"] = df["date"].dt.dayofweek.astype(int)

    channels = sorted(df["channel"].unique().tolist())
    campaigns = sorted(df["campaign_id"].unique().tolist())
    msg_types = sorted(df["message_type"].unique().tolist())

    # media transform per campaign×message series
    x_series = []
    for (camp, mt), g in df.groupby(["campaign_id", "message_type"], sort=False):
        idx = g.index
        s = g["spend"].values.astype(float)
        x = saturation(adstock(s, cfg.adstock_alpha), cfg.saturation_k)
        x_series.append(pd.Series(x, index=idx))
    df["x_media"] = pd.concat(x_series).sort_index().values

    # TM utilization per day (across all campaigns)
    daily = df.groupby("date", as_index=False).agg(tm_attempts_total=("tm_attempts", "sum"))
    daily["util"] = daily["tm_attempts_total"] / float(cfg.tm_capacity_per_day)
    df = df.merge(daily[["date", "util"]], on="date", how="left")
    return df, channels, campaigns, msg_types


def fit_bayesian_funnel_scm_campaign(camp_panel: pd.DataFrame, cfg: SCMConfig):
    """Hierarchical Bayesian SCM at campaign level (campaign→contracts supported).

    - Channel random effects (baseline)
    - Message-type fixed effects (SA/DA)
    - Campaign random effects nested under channel
    """
    import pymc as pm

    df, channels, campaigns, msg_types = _prep_design_campaign(camp_panel, cfg)
    df = _sanitize_funnel_df(df)
    ch_to_idx = {c: i for i, c in enumerate(channels)}
    camp_to_idx = {c: i for i, c in enumerate(campaigns)}
    mt_to_idx = {m: i for i, m in enumerate(msg_types)}

    ch_idx = df["channel"].map(ch_to_idx).values.astype(int)
    camp_idx = df["campaign_id"].map(camp_to_idx).values.astype(int)
    mt_idx = df["message_type"].map(mt_to_idx).values.astype(int)

    n_ch = len(channels)
    n_camp = len(campaigns)
    n_mt = len(msg_types)

    dow = df["dow"].values.astype(int)
    t = df["t"].values.astype(float)
    x_media = df["x_media"].values.astype(float)

    leads = df["leads"].values.astype(int)
    attempts = df["tm_attempts"].values.astype(int)
    connected = df["tm_connected"].values.astype(int)
    contracts = df["contracts"].values.astype(int)
    premium = df["premium"].values.astype(float)

    util = df["util"].values.astype(float)
    util_over = np.maximum(util - cfg.sla_softcap_connected, 0.0)

    with pm.Model() as model:
        # --- Leads ---
        a0 = pm.Normal("a0", 0.0, 2.0)
        a_ch = pm.Normal("a_ch", 0.0, 0.6, shape=n_ch)
        a_mt = pm.Normal("a_mt", 0.0, 0.35, shape=n_mt)
        a_dow = pm.Normal("a_dow", 0.0, 0.25, shape=7)
        a_trend = pm.Normal("a_trend", 0.0, 0.01)

        # campaign random intercept around channel
        sigma_camp = pm.HalfNormal("sigma_camp", 0.35)
        a_camp_raw = pm.Normal("a_camp_raw", 0.0, 1.0, shape=n_camp)
        a_camp = pm.Deterministic("a_camp", a_camp_raw * sigma_camp)

        # campaign-level media elasticity (random slope)
        sigma_b = pm.HalfNormal("sigma_b", 0.30)
        b_ch = pm.Normal("b_ch", 0.0, 0.45, shape=n_ch)
        b_camp_raw = pm.Normal("b_camp_raw", 0.0, 1.0, shape=n_camp)
        b_camp = pm.Deterministic("b_camp", b_ch[ch_idx] + b_camp_raw[camp_idx] * sigma_b)

        mu_leads = pm.math.exp(
            a0
            + a_ch[ch_idx]
            + a_mt[mt_idx]
            + a_camp[camp_idx]
            + b_camp * x_media
            + a_dow[dow]
            + a_trend * t
        )
        pm.Poisson("leads_obs", mu=mu_leads, observed=leads)

        # --- Connected (given attempts) ---
        c0 = pm.Normal("c0", 0.0, 2.0)
        c_mt = pm.Normal("c_mt", 0.0, 0.35, shape=n_mt)
        c2 = pm.Normal("c2", -2.0, 1.0)  # utilization penalty
        log_leads = pm.math.log(leads + 1)
        c1 = pm.Normal("c1", 0.0, 0.5)
        p_conn = pm.Deterministic("p_conn", pm.math.sigmoid(c0 + c_mt[mt_idx] + c1 * log_leads + c2 * util_over))
        pm.Binomial("conn_obs", n=attempts, p=p_conn, observed=connected)

        # --- Contracts (given connected) ---
        k0 = pm.Normal("k0", 0.0, 2.0)
        k_ch = pm.Normal("k_ch", 0.0, 0.5, shape=n_ch)
        k_mt = pm.Normal("k_mt", 0.0, 0.35, shape=n_mt)
        k1 = pm.Normal("k1", 0.0, 0.45)
        p_close = pm.Deterministic(
            "p_close",
            pm.math.sigmoid(k0 + k_ch[ch_idx] + k_mt[mt_idx] + k1 * log_leads),
        )
        pm.Binomial("cont_obs", n=connected, p=p_close, observed=contracts)

        # --- Premium per contract (hurdle: only when contracts>0) ---
        eps = 1e-6
        mask = contracts > 0
        prem_per = premium / (np.maximum(contracts, 1) + eps)
        prem_per = np.clip(prem_per, 1.0, None)

        m0 = pm.Normal("m0", np.log(50000), 0.5)
        m_ch = pm.Normal("m_ch", 0.0, 0.22, shape=n_ch)
        m_mt = pm.Normal("m_mt", 0.0, 0.15, shape=n_mt)
        sigma = pm.HalfNormal("sigma", 0.4)
        mu_log = m0 + m_ch[ch_idx] + m_mt[mt_idx]

        pm.LogNormal("prem_obs", mu=mu_log[mask], sigma=sigma, observed=prem_per[mask])

        idata = _run_inference(model, cfg)

    return idata, df, channels, campaigns, msg_types

def posterior_summary(idata, channels: list[str]) -> pd.DataFrame:
    import arviz as az
    summ = az.summary(idata, var_names=["a0","a_ch","b_ch","c0","c1","c2","k0","k_ch","k1","m0","m_ch","sigma"], round_to=4)
    summ = summ.reset_index().rename(columns={"index":"param"})
    def pretty(p):
        for prefix in ["a_ch[","b_ch[","k_ch[","m_ch["]:
            if p.startswith(prefix):
                i = int(p.split("[")[-1].split("]")[0])
                return p.replace(f"[{i}]", f"[{channels[i]}]")
        return p
    summ["param"] = summ["param"].apply(pretty)
    return summ


def posterior_summary_campaign(idata, channels: list[str], campaigns: list[str], msg_types: list[str]) -> pd.DataFrame:
    import arviz as az

    summ = az.summary(
        idata,
        var_names=[
            "a0",
            "a_ch",
            "a_mt",
            "sigma_camp",
            "sigma_b",
            "b_ch",
            "c0",
            "c_mt",
            "c1",
            "c2",
            "k0",
            "k_ch",
            "k_mt",
            "k1",
            "m0",
            "m_ch",
            "m_mt",
            "sigma",
        ],
        round_to=4,
    )
    summ = summ.reset_index().rename(columns={"index": "param"})

    def pretty(p: str) -> str:
        for prefix, arr in [("a_ch[", channels), ("b_ch[", channels), ("k_ch[", channels), ("m_ch[", channels)]:
            if p.startswith(prefix):
                i = int(p.split("[")[-1].split("]")[0])
                return p.replace(f"[{i}]", f"[{arr[i]}]")
        for prefix, arr in [("a_mt[", msg_types), ("c_mt[", msg_types), ("k_mt[", msg_types), ("m_mt[", msg_types)]:
            if p.startswith(prefix):
                i = int(p.split("[")[-1].split("]")[0])
                return p.replace(f"[{i}]", f"[{arr[i]}]")
        return p

    summ["param"] = summ["param"].apply(pretty)
    return summ

def simulate_counterfactuals(idata, design_df: pd.DataFrame, channels: list[str], scenario: dict, cfg: SCMConfig) -> pd.DataFrame:
    from .transforms import adstock, saturation
    rng = np.random.default_rng(cfg.seed)
    df = design_df.copy().sort_values(["date","channel"]).reset_index(drop=True)

    scale = scenario.get("scale_spend", {}) or {}
    zero = set(scenario.get("zero_channels", []) or [])

    spend_mod = df["spend"].astype(float).values.copy()
    for ch in channels:
        idx = df.index[df["channel"]==ch].to_numpy()
        mult = float(scale.get(ch, 1.0))
        if ch in zero:
            mult = 0.0
        spend_mod[idx] *= mult
        x = saturation(adstock(spend_mod[idx], cfg.adstock_alpha), cfg.saturation_k)
        df.loc[idx, "x_media_cf"] = x

    x_media = df["x_media_cf"].values.astype(float)
    ch_to_idx = {c:i for i,c in enumerate(channels)}
    ch_idx = df["channel"].map(ch_to_idx).values.astype(int)
    dow = df["dow"].values.astype(int)
    t = df["t"].values.astype(float)
    util_over = np.maximum(df["util"].values.astype(float) - cfg.sla_softcap_connected, 0.0)
    attempts = df["tm_attempts"].values.astype(float)

    post = idata.posterior
    # flatten chain/draw
    def flat(name):
        return post[name].stack(sample=("chain", "draw")).transpose("sample", ...).values
    n_draws = flat("a0").shape[0]
    take = min(300, n_draws)
    idxs = rng.choice(n_draws, take, replace=False)

    a0 = flat("a0")[idxs]
    a_ch = flat("a_ch")[idxs]
    b_ch = flat("b_ch")[idxs]
    a_dow = flat("a_dow")[idxs]
    a_trend = flat("a_trend")[idxs]
    c0 = flat("c0")[idxs]
    c1 = flat("c1")[idxs]
    c2 = flat("c2")[idxs]
    k0 = flat("k0")[idxs]
    k_ch = flat("k_ch")[idxs]
    k1 = flat("k1")[idxs]
    m0 = flat("m0")[idxs]
    m_ch = flat("m_ch")[idxs]

    def sigmoid(z):
        return 1/(1+np.exp(-z))

    mu_leads = np.exp(a0[:,None] + a_ch[:,ch_idx] + b_ch[:,ch_idx]*x_media[None,:] + a_dow[:,dow] + a_trend[:,None]*t[None,:])
    p_conn = sigmoid(c0[:,None] + c1[:,None]*np.log(mu_leads+1) + c2[:,None]*util_over[None,:])
    conn = attempts[None,:] * p_conn
    p_close = sigmoid(k0[:,None] + k_ch[:,ch_idx] + k1[:,None]*np.log(mu_leads+1))
    cont = conn * p_close
    prem_per = np.exp(m0[:,None] + m_ch[:,ch_idx])
    prem = cont * prem_per

    # aggregate to day
    unique_dates = pd.to_datetime(sorted(df["date"].unique()))
    day_map = {d:i for i,d in enumerate(unique_dates)}
    g = np.array([day_map[pd.to_datetime(d)] for d in df["date"]])
    n_days = len(unique_dates)

    def agg(mat):
        out = np.zeros((mat.shape[0], n_days))
        for j in range(n_days):
            out[:,j] = mat[:, g==j].sum(axis=1)
        return out

    leads_day = agg(mu_leads)
    conn_day = agg(conn)
    cont_day = agg(cont)
    prem_day = agg(prem)

    spend_cf = pd.Series(spend_mod).groupby(df["date"]).sum().reindex(unique_dates).values.astype(float)

    def summarize(arr):
        return pd.DataFrame({
            "mean": arr.mean(axis=0),
            "p10": np.quantile(arr, 0.10, axis=0),
            "p90": np.quantile(arr, 0.90, axis=0),
        })

    out = pd.DataFrame({"date": unique_dates})
    out = out.join(summarize(leads_day).add_prefix("leads_"))
    out = out.join(summarize(conn_day).add_prefix("connected_"))
    out = out.join(summarize(cont_day).add_prefix("contracts_"))
    out = out.join(summarize(prem_day).add_prefix("premium_"))
    out["spend_cf"] = spend_cf
    out["scenario"] = scenario.get("name","scenario")
    return out