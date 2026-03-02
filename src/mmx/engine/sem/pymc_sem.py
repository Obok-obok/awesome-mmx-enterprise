from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import json

@dataclass(frozen=True)
class SEMConfig:
    max_lag_lead_to_attempt: int = 7
    max_lag_connected_to_contract: int = 14
    report_delay_max: int = 0  # 0 = OFF (Reporting delay for Leads observation)

@dataclass(frozen=True)
class PosteriorArtifact:
    model_version: str
    channel_params: Dict[str, Dict[str, Any]]
    globals: Dict[str, float]

def fit_sem_pymc(df: pd.DataFrame, config: SEMConfig, method: str = 'VI', seed: int = 7) -> Tuple[PosteriorArtifact, 'object']:
    """Bayesian SEM joint estimation with PyMC.

    Jointly estimates:
    - Spend→Lead: adstock decay + Hill saturation
      - Reporting delay (optional): lead_mean is latent, observed leads are delayed mixture of latent leads
    - Lead→Attempt: lag kernel
    - Attempt→Connected: direct (no lag)
    - Connected→Contract: lag kernel
    - Premium per contract: lognormal
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.scan import scan

    
    # --- Canonical mart schema mapping ---
    # The system canonical daily mart uses: spend, leads, attempts, connected, contracts, premium.
    # Some legacy paths may still provide call_attempt/call_connected naming.
    def _sget(row: pd.Series, *names: str, default: float = 0.0) -> float:
        for name in names:
            if name in row and row[name] is not None:
                try:
                    return float(row[name])
                except Exception:
                    pass
        return float(default)

df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    channels = sorted(df['channel'].unique().tolist())
    dates = sorted(df['date'].unique().tolist())
    T = len(dates)
    C = len(channels)
    ch2i = {c:i for i,c in enumerate(channels)}
    d2i = {d:i for i,d in enumerate(dates)}

    X_spend = np.zeros((C,T), dtype=float)
    Y_lead  = np.zeros((C,T), dtype=float)
    Y_att   = np.zeros((C,T), dtype=float)
    Y_con   = np.zeros((C,T), dtype=float)
    Y_ctr   = np.zeros((C,T), dtype=float)
    Y_prm   = np.zeros((C,T), dtype=float)

    for _, r in df.iterrows():
        ci = ch2i[r['channel']]
        ti = d2i[pd.Timestamp(r['date'])]
        X_spend[ci, ti] = float(r['spend'])
        Y_lead[ci, ti]  = float(r['leads'])
        Y_att[ci, ti]   = _sget(r, 'attempts', 'call_attempt', 'call_attempt_count')
        Y_con[ci, ti]   = _sget(r, 'connected', 'call_connected', 'call_connected_count')
        Y_ctr[ci, ti]   = float(r['contracts'])
        Y_prm[ci, ti]   = float(r['premium'])

    with pm.Model() as model:
        decay = pm.Beta('decay', alpha=8, beta=2, shape=C)
        ec50  = pm.LogNormal('ec50', mu=np.log(max(1.0, np.nanmedian(X_spend[X_spend>0]) if (X_spend>0).any() else 1000.0)), sigma=1.0, shape=C)
        hill  = pm.LogNormal('hill', mu=np.log(1.0), sigma=0.3, shape=C)
        alpha_lead = pm.HalfNormal('alpha_lead', sigma=max(10.0, float(np.nanstd(Y_lead)+1.0)), shape=C)

        w_la = pm.Dirichlet('w_lead_to_attempt', a=np.ones(config.max_lag_lead_to_attempt+1), shape=(C, config.max_lag_lead_to_attempt+1))
        w_cc = pm.Dirichlet('w_connected_to_contract', a=np.ones(config.max_lag_connected_to_contract+1), shape=(C, config.max_lag_connected_to_contract+1))

        rate_attempt = pm.HalfNormal('rate_attempt_per_lead', sigma=2.0, shape=C)
        rate_connect = pm.Beta('rate_connected_per_attempt', alpha=5, beta=5, shape=C)
        rate_contract= pm.Beta('rate_contract_per_connected', alpha=3, beta=7, shape=C)

        mu_ppc    = pm.Normal('mu_log_premium_per_contract', mu=np.log(max(1.0, float(np.nanmean((Y_prm+1)/(Y_ctr+1))))), sigma=1.0)
        sigma_ppc = pm.HalfNormal('sigma_log_premium_per_contract', sigma=1.0)

        X = pt.as_tensor_variable(X_spend)
        def adstock_scan(x_row, d):
            def step(x_t, a_prev, dd):
                return x_t + dd * a_prev
            # NOTE:
            # - PyMC itself does not expose `scan`; use PyTensor's scan.
            # - This keeps the adstock carryover differentiable for VI/MCMC.
            a, _ = scan(fn=step, sequences=[x_row], outputs_info=[pt.zeros(())], non_sequences=[d])
            return a
        A = pt.stack([adstock_scan(X[i], decay[i]) for i in range(C)], axis=0)

        lead_mean = alpha_lead[:, None] * (
            pt.power(pt.clip(A, 0, np.inf), hill[:, None]) /
            (pt.power(pt.clip(A, 0, np.inf), hill[:, None]) + pt.power(ec50[:, None], hill[:, None]) + 1e-9)
        )

        # Optional reporting delay on Leads (OFF by default)
        if int(config.report_delay_max) > 0:
            D = int(config.report_delay_max)
            d_delay = pm.Dirichlet('d_lead_reporting_delay', a=np.ones(D+1))  # global kernel
            lead_obs_mean_rows = []
            for i in range(C):
                y = pt.zeros((T,))
                for t in range(T):
                    s = 0
                    for k in range(D+1):
                        if t-k >= 0:
                            s = s + d_delay[k] * lead_mean[i, t-k]
                    y = pt.set_subtensor(y[t], s)
                lead_obs_mean_rows.append(y)
            lead_obs_mean = pt.stack(lead_obs_mean_rows, axis=0)
        else:
            d_delay = None
            lead_obs_mean = lead_mean

        phi_lead = pm.HalfNormal('phi_lead', sigma=20.0)
        pm.NegativeBinomial('leads', mu=lead_obs_mean + 1e-3, alpha=phi_lead, observed=Y_lead)

        # Lead→Attempt lag
        lead_lagged = []
        for i in range(C):
            y = pt.zeros((T,))
            for t in range(T):
                s = 0
                for k in range(config.max_lag_lead_to_attempt+1):
                    if t-k >= 0:
                        s = s + w_la[i, k] * Y_lead[i, t-k]
                y = pt.set_subtensor(y[t], s)
            lead_lagged.append(y)
        lead_lagged = pt.stack(lead_lagged, axis=0)

        attempt_mean = rate_attempt[:, None] * lead_lagged
        phi_att = pm.HalfNormal('phi_attempt', sigma=20.0)
        pm.NegativeBinomial('call_attempt', mu=attempt_mean + 1e-3, alpha=phi_att, observed=Y_att)

        # Attempt→Connected (no lag)
        connected_mean = rate_connect[:, None] * Y_att
        phi_con = pm.HalfNormal('phi_connected', sigma=20.0)
        pm.NegativeBinomial('call_connected', mu=connected_mean + 1e-3, alpha=phi_con, observed=Y_con)

        # Connected→Contract lag
        con_lagged = []
        for i in range(C):
            y = pt.zeros((T,))
            for t in range(T):
                s = 0
                for k in range(config.max_lag_connected_to_contract+1):
                    if t-k >= 0:
                        s = s + w_cc[i, k] * Y_con[i, t-k]
                y = pt.set_subtensor(y[t], s)
            con_lagged.append(y)
        con_lagged = pt.stack(con_lagged, axis=0)

        contract_mean = rate_contract[:, None] * con_lagged
        phi_ctr = pm.HalfNormal('phi_contract', sigma=20.0)
        pm.NegativeBinomial('contracts', mu=contract_mean + 1e-3, alpha=phi_ctr, observed=Y_ctr)

        ppc_obs = (Y_prm + 1e-3) / (Y_ctr + 1.0)
        pm.LogNormal('premium_per_contract', mu=mu_ppc, sigma=sigma_ppc, observed=ppc_obs)

        if method.upper() == 'VI':
            approx = pm.fit(random_seed=seed, method='advi')
            trace = approx.sample(2000, random_seed=seed)
        else:
            trace = pm.sample(1000, tune=1000, random_seed=seed, chains=2, target_accept=0.9)

    w_la_mean = trace.posterior['w_lead_to_attempt'].mean(dim=('chain','draw')).values
    w_cc_mean = trace.posterior['w_connected_to_contract'].mean(dim=('chain','draw')).values

    channel_params: Dict[str, Dict[str, Any]] = {}
    for i, ch in enumerate(channels):
        channel_params[ch] = {
            'decay': float(trace.posterior['decay'].mean(dim=('chain','draw')).values[i]),
            'ec50': float(trace.posterior['ec50'].mean(dim=('chain','draw')).values[i]),
            'hill': float(trace.posterior['hill'].mean(dim=('chain','draw')).values[i]),
            'alpha_lead': float(trace.posterior['alpha_lead'].mean(dim=('chain','draw')).values[i]),
            'rate_attempt_per_lead': float(trace.posterior['rate_attempt_per_lead'].mean(dim=('chain','draw')).values[i]),
            'rate_connected_per_attempt': float(trace.posterior['rate_connected_per_attempt'].mean(dim=('chain','draw')).values[i]),
            'rate_contract_per_connected': float(trace.posterior['rate_contract_per_connected'].mean(dim=('chain','draw')).values[i]),
            'w_lead_to_attempt': json.dumps([float(x) for x in w_la_mean[i].tolist()]),
            'w_connected_to_contract': json.dumps([float(x) for x in w_cc_mean[i].tolist()]),
        }

    globals_ = {
        'mu_log_premium_per_contract': float(trace.posterior['mu_log_premium_per_contract'].mean().values),
        'sigma_log_premium_per_contract': float(trace.posterior['sigma_log_premium_per_contract'].mean().values),
        'report_delay_max': int(config.report_delay_max),
    }

    if int(config.report_delay_max) > 0:
        d_mean = trace.posterior['d_lead_reporting_delay'].mean(dim=('chain','draw')).values
        globals_['d_lead_reporting_delay'] = json.dumps([float(x) for x in d_mean.tolist()])

    model_version = 'v' + pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
    return PosteriorArtifact(model_version=model_version, channel_params=channel_params, globals=globals_), trace
