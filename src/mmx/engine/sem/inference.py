from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import json

from mmx.engine.sem.adstock import adstock_geometric
from mmx.engine.sem.saturation import hill
from mmx.engine.sem.lag import apply_lag_kernel, default_kernel


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray | float:
    """Numerically stable logit."""
    p_ = np.clip(p, eps, 1.0 - eps)
    return np.log(p_ / (1.0 - p_))


def _adstock_warmup_days(decay: float, eps: float = 0.01, cap: int = 90) -> int:
    """Approximate warmup window so that carryover decays below eps.

    We find the smallest n with decay**n <= eps (capped).
    """
    if not (0.0 < decay < 1.0):
        return 0
    try:
        n = int(np.ceil(np.log(eps) / np.log(decay)))
    except Exception:
        return 0
    return int(max(0, min(cap, n)))


def _build_spend_series(
    df_daily: pd.DataFrame,
    channel: str,
    period_start: pd.Timestamp,
    pre_days: int,
    n_days: int,
    spend_total: float,
) -> np.ndarray:
    """Spend series for [period_start-pre_days, period_end] inclusive.

    - History uses observed spend from df_daily (missing -> 0)
    - Future allocates spend_total evenly across n_days.
    """
    df = df_daily.copy()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    hist_start = period_start - pd.Timedelta(days=int(pre_days))
    hist_dates = pd.date_range(start=hist_start, end=period_start - pd.Timedelta(days=1), freq="D")
    fut_dates = pd.date_range(start=period_start, periods=int(n_days), freq="D")

    hist = np.zeros(len(hist_dates), dtype=float)
    if len(hist_dates) and not df.empty:
        sub = df[df["channel"] == channel]
        if not sub.empty:
            sub = sub.groupby("date", as_index=True)["spend"].sum()
            for i, d in enumerate(hist_dates):
                if d in sub.index:
                    hist[i] = float(sub.loc[d])

    fut = np.full(len(fut_dates), float(spend_total) / max(1, int(n_days)), dtype=float)
    return np.concatenate([hist, fut], axis=0)

@dataclass(frozen=True)
class PosteriorSummary:
    model_version: str
    channel_params: Dict[str, Dict[str, Any]]
    globals: Dict[str, float]

def fit_posterior(df_daily: pd.DataFrame, backend: str, method: str, report_delay_max: int = 0) -> Tuple[PosteriorSummary, Optional[Any]]:
    backend = backend.upper()
    if backend == 'PYMC':
        from mmx.engine.sem.pymc_sem import fit_sem_pymc, SEMConfig
        art, trace = fit_sem_pymc(df_daily, config=SEMConfig(report_delay_max=report_delay_max), method=method)
        return PosteriorSummary(art.model_version, art.channel_params, art.globals), trace

    if backend == 'REFERENCE':
        channels = sorted(df_daily['channel'].unique().tolist()) if not df_daily.empty else []
        params: Dict[str, Dict[str, Any]] = {}
        # Pool latent-quality residuals across channels to build data-driven
        # safety rails (clip bounds) that prevent logit blow-ups.
        pooled_lq_resid: List[float] = []
        for ch in channels:
            sub = df_daily[df_daily['channel'] == ch]
            med_spend = float(np.nanmedian(sub['spend'])) if len(sub) else 1000.0
            med_leads = float(np.nanmedian(sub.get('leads', 0))) if len(sub) else 0.0

            # Calibrate funnel rates from observed data (train window).
            # This avoids pathological predictions (e.g., attempts=0 but contracts>0)
            # and keeps REFERENCE backend numerically plausible.
            leads_sum = float(np.nansum(sub.get('leads', 0))) if len(sub) else 0.0
            att_sum = float(np.nansum(sub.get('attempts', sub.get('call_attempt', 0)))) if len(sub) else 0.0
            con_sum = float(np.nansum(sub.get('connected', sub.get('call_connected', 0)))) if len(sub) else 0.0
            ctr_sum = float(np.nansum(sub.get('contracts', 0))) if len(sub) else 0.0

            def _clip_rate(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
                if not np.isfinite(x):
                    return float(lo)
                return float(min(max(x, lo), hi))

            rate_att = _clip_rate(att_sum / leads_sum) if leads_sum > 0 else 0.75
            rate_con = _clip_rate(con_sum / att_sum) if att_sum > 0 else 0.35
            rate_ctr = _clip_rate(ctr_sum / con_sum) if con_sum > 0 else 0.20

            # --- Latent Quality (LQ) calibration (v5.4+ groundwork) ---
            # We model a latent quality state q_{c,t} that perturbs *all* downstream
            # conversion logits. This allows rates to vary by time and by spend
            # intensity (optional), and provides a principled path to a full
            # Bayesian SEM with latent quality.
            #
            #   logit(p_x_{c,t}) = logit(base_p_x_c) + q_{c,t}
            #   q_{c,t} = rho_q * q_{c,t-1} + beta_q * log1p(spend_{c,t}) + eps
            #
            # REFERENCE backend estimates (rho_q, beta_q, q_last) using simple
            # robust moment/regression heuristics from the train window.
            base_logits = {
                "att": float(_logit(rate_att)),
                "con": float(_logit(rate_con)),
                "ctr": float(_logit(rate_ctr)),
            }

            q_last = 0.0
            rho_q = 0.5
            beta_q = 0.0
            sigma_q = 0.15
            try:
                # Daily residual in logit-space.
                s = sub.copy()
                s = s.sort_values("date")
                s_spend = s["spend"].astype(float).to_numpy()
                s_leads = s.get("leads", 0).astype(float).to_numpy()
                s_att = s.get("attempts", s.get("call_attempt", 0)).astype(float).to_numpy()
                s_con = s.get("connected", s.get("call_connected", 0)).astype(float).to_numpy()
                s_ctr = s.get("contracts", 0).astype(float).to_numpy()

                # Compute observed logits when denominators are valid.
                resid = []
                spend_feat = []
                for i in range(len(s)):
                    r_i = []
                    if s_leads[i] > 0:
                        r_i.append(float(_logit(_clip_rate(s_att[i] / s_leads[i])) - base_logits["att"]))
                    if s_att[i] > 0:
                        r_i.append(float(_logit(_clip_rate(s_con[i] / s_att[i])) - base_logits["con"]))
                    if s_con[i] > 0:
                        r_i.append(float(_logit(_clip_rate(s_ctr[i] / s_con[i])) - base_logits["ctr"]))
                    if r_i:
                        resid.append(float(np.mean(r_i)))
                        spend_feat.append(float(np.log1p(max(0.0, s_spend[i]))))

                if len(resid) >= 14:
                    y = np.asarray(resid, dtype=float)
                    x = np.asarray(spend_feat, dtype=float)
                    x = x - float(np.mean(x))

                    # Simple robust slope (beta_q) in logit residual space.
                    # If x is nearly constant, keep beta_q=0.
                    if float(np.std(x)) > 1e-6:
                        beta_q = float(np.cov(x, y, bias=True)[0, 1] / (np.var(x) + 1e-9))
                        beta_q = float(np.clip(beta_q, -0.5, 0.5))
                    y_d = y - beta_q * x

                    # Collect residuals for global clip bounds.
                    pooled_lq_resid.extend([float(v) for v in y_d if np.isfinite(v)])

                    # AR(1) coefficient via lag-1 autocorrelation.
                    if len(y_d) >= 3:
                        y0 = y_d[:-1]
                        y1 = y_d[1:]
                        denom = float(np.dot(y0, y0)) + 1e-9
                        rho_q = float(np.dot(y0, y1) / denom)
                        rho_q = float(np.clip(rho_q, 0.0, 0.95))

                    sigma_q = float(np.nanstd(y_d))
                    sigma_q = float(np.clip(sigma_q, 0.05, 0.50))
                    q_last = float(y_d[-1])
            except Exception:
                # Keep defaults if anything goes wrong; REFERENCE backend must be resilient.
                pass

            # Saturation ceiling: use a robust high percentile of observed leads.
            # Hill() outputs are capped at alpha_lead, so alpha_lead should reflect
            # the attainable daily lead level rather than an inflated constant.
            alpha_lead = float(max(1.0, np.nanpercentile(sub['leads'], 90))) if len(sub) else 10.0

            # Channel-specific saturation midpoint (ec50).
            # Use a simple hyperbolic calibration so different channels are not symmetric.
            # If leads ~= alpha * spend/(spend+ec50), then ec50 ~= spend * (alpha/leads - 1).
            if np.isfinite(med_spend) and med_spend > 0 and np.isfinite(med_leads) and med_leads > 0 and alpha_lead > med_leads:
                ec50 = float(med_spend * max(0.1, (alpha_lead / med_leads) - 1.0))
            else:
                ec50 = float(max(1.0, med_spend))
            params[ch] = {
                'decay': 0.8,
                'ec50': max(1.0, ec50),
                'hill': 1.0,
                'alpha_lead': alpha_lead,
                'rate_attempt_per_lead': rate_att,
                'rate_connected_per_attempt': rate_con,
                'rate_contract_per_connected': rate_ctr,
                # Latent quality parameters
                'lq_logit_base_attempt': base_logits["att"],
                'lq_logit_base_connected': base_logits["con"],
                'lq_logit_base_contract': base_logits["ctr"],
                'lq_rho': rho_q,
                'lq_beta_log1p_spend': beta_q,
                'lq_sigma': sigma_q,
                'lq_last_state': q_last,
                'w_lead_to_attempt': json.dumps(list(default_kernel(7))),
                'w_connected_to_contract': json.dumps(list(default_kernel(14))),
            }
        mv = 'v' + pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')

        # --- Latent Quality safety rails (data-driven) ---
        # We clip q_{c,t} in logit space to a robust range derived from the
        # observed train-window variability of logit-residuals.
        # This prevents pathological probabilities (near 0/1) that can explode
        # downstream counts/premium.
        if len(pooled_lq_resid) >= 50:
            q_clip_low = float(np.nanquantile(pooled_lq_resid, 0.01))
            q_clip_high = float(np.nanquantile(pooled_lq_resid, 0.99))
        else:
            # Conservative default: within ~[sigmoid(-1), sigmoid(+1)] swings.
            q_clip_low, q_clip_high = -1.0, 1.0
        # Hard sanity bounds.
        q_clip_low = float(np.clip(q_clip_low, -2.0, 0.0))
        q_clip_high = float(np.clip(q_clip_high, 0.0, 2.0))
        if q_clip_high <= q_clip_low:
            q_clip_low, q_clip_high = -1.0, 1.0

        # Loading coefficient that controls how strongly latent quality perturbs
        # downstream conversion logits. Keeping this <1 makes the model robust.
        alpha_rate_lq = 0.5

        # Calibrate premium-per-contract scale from observed data so predictions are on the same magnitude.
        # This matters for insurance-like domains where PPC can be large.
        ppc = None
        try:
            sub = df_daily.copy()
            sub = sub[(sub.get('contracts', 0) > 0) & (sub.get('premium', 0) > 0)]
            if not sub.empty:
                ppc = (sub['premium'].astype(float) / sub['contracts'].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
                ppc = ppc[ppc > 0]
        except Exception:
            ppc = None

        if ppc is not None and len(ppc) >= 10:
            log_ppc = np.log(ppc.to_numpy(dtype=float))
            mu_hat = float(np.mean(log_ppc))
            sigma_hat = float(np.std(log_ppc))
            sigma_hat = sigma_hat if np.isfinite(sigma_hat) and sigma_hat > 1e-6 else 0.35
        else:
            # Fallback if data is insufficient.
            mu_hat = float(np.log(1_000_000.0))
            sigma_hat = 0.35

        # --- PPC x Latent Quality coupling (v5.4.1) ---
        # We optionally couple premium-per-contract (PPC) to latent quality:
        #   log(PPC) = mu + gamma_ppc * q_{c,t} + eps
        # Estimate gamma_ppc with a strongly-shrunk regression on the implied
        # latent quality state (q-hat) derived from spend dynamics.
        gamma_ppc = 0.0
        try:
            d = df_daily.copy()
            d["date"] = pd.to_datetime(d["date"])
            d = d[(d.get("contracts", 0) > 0) & (d.get("premium", 0) > 0) & (d.get("spend", 0) >= 0)]
            if not d.empty:
                d = d.sort_values(["channel", "date"])
                d["ppc"] = (d["premium"].astype(float) / d["contracts"].astype(float)).replace([np.inf, -np.inf], np.nan)
                d = d.dropna(subset=["ppc"])
                d = d[d["ppc"] > 0]

                if len(d) >= 30:
                    q_vals = []
                    y_vals = []
                    for ch in sorted(d["channel"].unique().tolist()):
                        p = params.get(ch)
                        if not p:
                            continue
                        rho = float(p.get("lq_rho", 0.0))
                        beta = float(p.get("lq_beta_log1p_spend", 0.0))
                        q_prev = 0.0
                        sub = d[d["channel"] == ch]
                        for _, r in sub.iterrows():
                            s = float(r.get("spend", 0.0))
                            q_prev = rho * q_prev + beta * float(np.log1p(max(0.0, s)))
                            q_vals.append(float(q_prev))
                            y_vals.append(float(np.log(float(r["ppc"])) - mu_hat))

                    if len(q_vals) >= 30:
                        q = np.asarray(q_vals, dtype=float)
                        y = np.asarray(y_vals, dtype=float)
                        # Center to remove intercept leakage.
                        q = q - float(np.mean(q))
                        v = float(np.var(q))
                        if v > 1e-8:
                            gamma_ppc = float(np.cov(q, y, bias=True)[0, 1] / (v + 1e-12))
                            # Strong shrinkage / safety clamp (enterprise ops): keep effect small.
                            # PPC is noisy; an overly large coupling can explode Premium.
                            gamma_ppc = float(np.clip(gamma_ppc, -0.10, 0.10))
                            gamma_ppc = float(gamma_ppc * 0.20)  # shrink further
        except Exception:
            gamma_ppc = 0.0

        globals_ = {
            'mu_log_premium_per_contract': mu_hat,
            'sigma_log_premium_per_contract': float(sigma_hat),
            'gamma_ppc_lq': float(gamma_ppc),
            # Latent quality safety rails + loading
            'lq_clip_low': float(q_clip_low),
            'lq_clip_high': float(q_clip_high),
            'lq_alpha_rate': float(alpha_rate_lq),
        }
        return PosteriorSummary(mv, params, globals_), None

    raise ValueError(f'Unknown backend: {backend}')

def posterior_predict_premium(
    df_daily: pd.DataFrame,
    posterior: PosteriorSummary,
    budget_plan: Dict[str, float],
    *,
    period_start: Optional[pd.Timestamp] = None,
    n_days: int = 30,
    warm_start: bool = True,
    warm_start_days: Optional[int] = None,
    n_samples: int = 2000,
    seed: int = 17,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = np.zeros(n_samples, dtype=float)
    mu = float(posterior.globals.get('mu_log_premium_per_contract', np.log(1000.0)))
    sigma = float(posterior.globals.get('sigma_log_premium_per_contract', 0.5))
    gamma_ppc = float(posterior.globals.get('gamma_ppc_lq', 0.0))
    lq_clip_low = float(posterior.globals.get('lq_clip_low', -1.0))
    lq_clip_high = float(posterior.globals.get('lq_clip_high', 1.0))
    lq_alpha = float(posterior.globals.get('lq_alpha_rate', 0.5))

    for ch, spend_total in budget_plan.items():
        p = posterior.channel_params.get(ch)
        if not p:
            continue
        decay = float(p['decay'])
        ec50 = float(p['ec50'])
        hill_p = float(p.get('hill', 1.0))
        alpha_lead = float(p['alpha_lead'])
        # Base rates (REFERENCE); may be time-varying via latent quality.
        rate_att = float(p['rate_attempt_per_lead'])
        rate_con = float(p['rate_connected_per_attempt'])
        rate_ctr = float(p['rate_contract_per_connected'])

        # Latent Quality parameters (optional; defaults keep legacy behavior).
        lq_rho = float(p.get('lq_rho', 0.0))
        lq_beta = float(p.get('lq_beta_log1p_spend', 0.0))
        lq_last = float(p.get('lq_last_state', 0.0))
        lq_logit_att = float(p.get('lq_logit_base_attempt', _logit(rate_att)))
        lq_logit_con = float(p.get('lq_logit_base_connected', _logit(rate_con)))
        lq_logit_ctr = float(p.get('lq_logit_base_contract', _logit(rate_ctr)))

        w_la = np.array(json.loads(p['w_lead_to_attempt']), dtype=float)
        w_cc = np.array(json.loads(p['w_connected_to_contract']), dtype=float)

        max_lag = max(len(w_la) - 1, len(w_cc) - 1)
        if warm_start and period_start is not None:
            pre = int(warm_start_days) if warm_start_days is not None else max(max_lag, _adstock_warmup_days(decay))
            x_full = _build_spend_series(df_daily, ch, pd.to_datetime(period_start), pre_days=pre, n_days=int(n_days), spend_total=float(spend_total))
            a_full = adstock_geometric(x_full, decay=decay)
            leads_full = hill(a_full, alpha=alpha_lead, ec50=ec50, hill=hill_p)
            # Latent quality state over the full series (history + forecast window).
            q = np.zeros_like(x_full, dtype=float)
            q_prev = lq_last
            for i in range(len(x_full)):
                q_prev = lq_rho * q_prev + lq_beta * float(np.log1p(max(0.0, x_full[i])))
                q[i] = q_prev
            q = np.clip(q, lq_clip_low, lq_clip_high)
            p_att = _sigmoid(lq_logit_att + lq_alpha * q)
            p_con = _sigmoid(lq_logit_con + lq_alpha * q)
            p_ctr = _sigmoid(lq_logit_ctr + lq_alpha * q)

            lag_leads = apply_lag_kernel(leads_full, w_la, max_lag=len(w_la)-1)
            att_full = p_att * lag_leads
            con_full = p_con * att_full
            lag_con = apply_lag_kernel(con_full, w_cc, max_lag=len(w_cc)-1)
            ctr_full = p_ctr * lag_con
            ctr = ctr_full[-int(n_days):]
            q_win = q[-int(n_days):]
        else:
            x = np.full(int(n_days), float(spend_total) / max(1, int(n_days)))
            a = adstock_geometric(x, decay=decay)
            leads = hill(a, alpha=alpha_lead, ec50=ec50, hill=hill_p)
            q = np.zeros_like(x, dtype=float)
            q_prev = lq_last
            for i in range(len(x)):
                q_prev = lq_rho * q_prev + lq_beta * float(np.log1p(max(0.0, x[i])))
                q[i] = q_prev
            q = np.clip(q, lq_clip_low, lq_clip_high)
            p_att = _sigmoid(lq_logit_att + lq_alpha * q)
            p_con = _sigmoid(lq_logit_con + lq_alpha * q)
            p_ctr = _sigmoid(lq_logit_ctr + lq_alpha * q)

            lag_leads = apply_lag_kernel(leads, w_la, max_lag=len(w_la)-1)
            att = p_att * lag_leads
            con = p_con * att
            lag_con = apply_lag_kernel(con, w_cc, max_lag=len(w_cc)-1)
            ctr = p_ctr * lag_con
            q_win = q

        # PPC mean depends (optionally) on latent quality; use the period-average q.
        q_bar = float(np.mean(q_win)) if len(q_win) else 0.0
        mu_eff = float(mu + gamma_ppc * q_bar)
        ppc = rng.lognormal(mean=mu_eff, sigma=sigma, size=n_samples)
        total += float(np.sum(ctr)) * ppc

    return np.clip(total, 0.0, None)


def posterior_predict_funnel(
    posterior: PosteriorSummary,
    budget_plan: Dict[str, float],
    *,
    df_daily: Optional[pd.DataFrame] = None,
    period_start: Optional[pd.Timestamp] = None,
    n_days: int = 30,
    warm_start: bool = True,
    warm_start_days: Optional[int] = None,
    n_samples: int = 2000,
    seed: int = 19,
) -> Dict[str, np.ndarray]:
    """Posterior predictive samples for funnel totals: leads, attempts, connected, contracts, premium."""
    rng = np.random.default_rng(seed)
    lq_clip_low = float(posterior.globals.get('lq_clip_low', -1.0))
    lq_clip_high = float(posterior.globals.get('lq_clip_high', 1.0))
    lq_alpha = float(posterior.globals.get('lq_alpha_rate', 0.5))
    leads_s = np.zeros(n_samples)
    att_s = np.zeros(n_samples)
    con_s = np.zeros(n_samples)
    ctr_s = np.zeros(n_samples)
    prem_s = np.zeros(n_samples)

    mu = float(posterior.globals.get("mu_log_premium_per_contract", np.log(1000.0)))
    sigma = float(posterior.globals.get("sigma_log_premium_per_contract", 0.5))
    gamma_ppc = float(posterior.globals.get("gamma_ppc_lq", 0.0))

    import json as _json
    for ch, spend_total in budget_plan.items():
        p = posterior.channel_params.get(ch)
        if not p:
            continue
        decay = float(p["decay"])
        ec50 = float(p["ec50"])
        hill_p = float(p.get("hill", 1.0))
        alpha_lead = float(p["alpha_lead"])
        # Base rates; may be time-varying via latent quality.
        rate_att = float(p["rate_attempt_per_lead"])
        rate_con = float(p["rate_connected_per_attempt"])
        rate_ctr = float(p["rate_contract_per_connected"])

        lq_rho = float(p.get('lq_rho', 0.0))
        lq_beta = float(p.get('lq_beta_log1p_spend', 0.0))
        lq_last = float(p.get('lq_last_state', 0.0))
        lq_logit_att = float(p.get('lq_logit_base_attempt', _logit(rate_att)))
        lq_logit_con = float(p.get('lq_logit_base_connected', _logit(rate_con)))
        lq_logit_ctr = float(p.get('lq_logit_base_contract', _logit(rate_ctr)))

        w_la = np.array(_json.loads(p["w_lead_to_attempt"]), dtype=float)
        w_cc = np.array(_json.loads(p["w_connected_to_contract"]), dtype=float)

        max_lag = max(len(w_la) - 1, len(w_cc) - 1)
        if warm_start and period_start is not None and df_daily is not None:
            pre = int(warm_start_days) if warm_start_days is not None else max(max_lag, _adstock_warmup_days(decay))
            x_full = _build_spend_series(df_daily, ch, pd.to_datetime(period_start), pre_days=pre, n_days=int(n_days), spend_total=float(spend_total))
            a_full = adstock_geometric(x_full, decay=decay)
            leads_full = hill(a_full, alpha=alpha_lead, ec50=ec50, hill=hill_p)
            q = np.zeros_like(x_full, dtype=float)
            q_prev = lq_last
            for i in range(len(x_full)):
                q_prev = lq_rho * q_prev + lq_beta * float(np.log1p(max(0.0, x_full[i])))
                q[i] = q_prev
            q = np.clip(q, lq_clip_low, lq_clip_high)
            p_att = _sigmoid(lq_logit_att + lq_alpha * q)
            p_con = _sigmoid(lq_logit_con + lq_alpha * q)
            p_ctr = _sigmoid(lq_logit_ctr + lq_alpha * q)

            lag_leads = apply_lag_kernel(leads_full, w_la, max_lag=len(w_la)-1)
            att_full = p_att * lag_leads
            con_full = p_con * att_full
            lag_con = apply_lag_kernel(con_full, w_cc, max_lag=len(w_cc)-1)
            ctr_full = p_ctr * lag_con
            leads = leads_full[-int(n_days):]
            att = att_full[-int(n_days):]
            con = con_full[-int(n_days):]
            ctr = ctr_full[-int(n_days):]
            q_win = q[-int(n_days):]
        else:
            x = np.full(int(n_days), float(spend_total) / max(1, int(n_days)))
            a = adstock_geometric(x, decay=decay)
            leads = hill(a, alpha=alpha_lead, ec50=ec50, hill=hill_p)
            q = np.zeros_like(x, dtype=float)
            q_prev = lq_last
            for i in range(len(x)):
                q_prev = lq_rho * q_prev + lq_beta * float(np.log1p(max(0.0, x[i])))
                q[i] = q_prev
            q = np.clip(q, lq_clip_low, lq_clip_high)
            p_att = _sigmoid(lq_logit_att + lq_alpha * q)
            p_con = _sigmoid(lq_logit_con + lq_alpha * q)
            p_ctr = _sigmoid(lq_logit_ctr + lq_alpha * q)

            lag_leads = apply_lag_kernel(leads, w_la, max_lag=len(w_la)-1)
            att = p_att * lag_leads
            con = p_con * att
            lag_con = apply_lag_kernel(con, w_cc, max_lag=len(w_cc)-1)
            ctr = p_ctr * lag_con
            q_win = q

        leads_total = float(np.sum(leads))
        att_total = float(np.sum(att))
        con_total = float(np.sum(con))
        ctr_total = float(np.sum(ctr))

        q_bar = float(np.mean(q_win)) if len(q_win) else 0.0
        mu_eff = float(mu + gamma_ppc * q_bar)
        ppc = rng.lognormal(mean=mu_eff, sigma=sigma, size=n_samples)

        leads_s += leads_total
        att_s += att_total
        con_s += con_total
        ctr_s += ctr_total
        prem_s += ctr_total * ppc

    # --- Safety: enforce funnel monotonicity at the aggregate level ---
    # Even with a consistent generative model, downstream schema mismatches or
    # numerical issues can surface as violations (e.g., contracts > connected).
    # We clamp to preserve causal ordering.
    leads_s = np.clip(leads_s, 0, None)
    att_s = np.clip(att_s, 0, None)
    con_s = np.clip(con_s, 0, None)
    ctr_s = np.clip(ctr_s, 0, None)
    prem_s = np.clip(prem_s, 0, None)

    att_s = np.minimum(att_s, leads_s)
    con_s = np.minimum(con_s, att_s)
    ctr_s = np.minimum(ctr_s, con_s)

    return {
        "leads": leads_s,
        # Canonical keys (aligned with mart): attempts / connected
        "attempts": att_s,
        "connected": con_s,
        "contracts": ctr_s,
        "premium": prem_s,
    }
