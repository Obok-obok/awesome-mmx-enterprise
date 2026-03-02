from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dlm_kalman import DLMState, build_Q, online_step
from .bayes_updates import BetaPosterior, NIGPosterior, make_rate_posteriors


def _dow_onehot(dates: pd.Series) -> np.ndarray:
    dow = pd.to_datetime(dates).dt.dayofweek.values  # 0=Mon
    X = np.zeros((len(dow), 7), dtype=float)
    X[np.arange(len(dow)), dow] = 1.0
    return X


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0.0, None))


@dataclass
class AdaptiveSCMConfig:
    channels: List[str]
    tm_capacity_per_day: float = 1000.0
    util_softcap: float = 0.85
    obs_var_leads: float = 0.20  # variance on log1p(leads)
    q_drift: float = 0.01
    seed: int = 42


@dataclass
class AdaptiveSCMState:
    dlm: DLMState
    conn_rate: Dict[str, Dict[str, BetaPosterior]]
    close_rate: Dict[str, BetaPosterior]
    prem_per_contract: Dict[str, NIGPosterior]
    # ops meta
    last_date: str | None = None
    t_index: int = 0


class AdaptiveSCM:
    """Dynamic Bayesian SCM with online updates.

    - Leads: dynamic linear model on log1p(leads) with time-varying channel media coefficients.
    - Connected rate: beta-binomial by channel and utilization bucket.
    - Close rate: beta-binomial by channel.
    - Premium per contract: normal-inverse-gamma on log premium/contract by channel.
    """

    def __init__(self, cfg: AdaptiveSCMConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Design: [intercept, trend, 7 dow, x_media per channel]
        self.k = 1 + 1 + 7 + len(cfg.channels)
        self.idx_media = list(range(1 + 1 + 7, self.k))
        Q = build_Q(self.k, drift_idx=self.idx_media, q_drift=cfg.q_drift, q_static=0.0)
        m0 = np.zeros(self.k)
        C0 = np.eye(self.k) * 10.0  # diffuse
        self.state = AdaptiveSCMState(
            dlm=DLMState(m=m0, C=C0, R=cfg.obs_var_leads, Q=Q),
            conn_rate=make_rate_posteriors(cfg.channels, ["OK", "OVER"]),
            close_rate={ch: BetaPosterior(a=2.0, b=18.0) for ch in cfg.channels},
            prem_per_contract={ch: NIGPosterior(m=8.0, k=1.0, a=2.0, b=2.0) for ch in cfg.channels},
        )

    def design_row(self, date: pd.Timestamp, x_media_by_ch: Dict[str, float], t_index: int) -> np.ndarray:
        F = np.zeros(self.k, dtype=float)
        F[0] = 1.0
        F[1] = float(t_index)
        dow = int(date.dayofweek)
        F[2 + dow] = 1.0
        for j, ch in enumerate(self.cfg.channels):
            F[self.idx_media[j]] = float(x_media_by_ch.get(ch, 0.0))
        return F

    def util_bucket(self, util: float) -> str:
        return "OVER" if util > self.cfg.util_softcap else "OK"

    def update_day(
        self,
        date: pd.Timestamp,
        x_media_by_ch: Dict[str, float],
        leads_total: float,
        tm_attempts_total: float,
        connected_by_ch: Dict[str, float],
        contracts_by_ch: Dict[str, float],
        premium_by_ch: Dict[str, float],
        t_index: int,
    ) -> Dict[str, float]:
        """Online update with one day of aggregated observations."""
        y = float(np.log1p(max(leads_total, 0.0)))
        F = self.design_row(date, x_media_by_ch, t_index)

        self.state.dlm, f, q = online_step(self.state.dlm, F, y)
        resid = float(y - f)

        util = float(tm_attempts_total) / float(self.cfg.tm_capacity_per_day) if self.cfg.tm_capacity_per_day > 0 else 0.0
        ub = self.util_bucket(util)

        # Update funnel posteriors per channel
        for ch in self.cfg.channels:
            conn = float(connected_by_ch.get(ch, 0.0))
            att = float(max(connected_by_ch.get(f"{ch}__attempts", tm_attempts_total / max(len(self.cfg.channels), 1)), 0.0))
            # attempts per channel may be unavailable; fall back to equal split
            self.state.conn_rate[ch][ub] = self.state.conn_rate[ch][ub].update(conn, att)

            cont = float(contracts_by_ch.get(ch, 0.0))
            self.state.close_rate[ch] = self.state.close_rate[ch].update(cont, max(conn, 0.0) + 1e-9)

            prem = float(premium_by_ch.get(ch, 0.0))
            if cont > 0 and prem > 0:
                per = prem / cont
                self.state.prem_per_contract[ch] = self.state.prem_per_contract[ch].update(np.array([np.log(per)]))

        return {
            "date": str(date.date()),
            "leads_log_pred": f,
            "leads_log_pred_var": q,
            "leads_log_obs": y,
            "leads_log_resid": resid,
            "util": util,
        }

    def simulate_next_day_revenue(
        self,
        date: pd.Timestamp,
        x_media_by_ch: Dict[str, float],
        tm_attempts_total: float,
        t_index: int,
        n_draws: int = 200,
    ) -> Dict[str, float]:
        """Short-horizon (1-day) posterior predictive simulation."""
        F = self.design_row(date, x_media_by_ch, t_index)
        m, C = self.state.dlm.m, self.state.dlm.C

        # Sample theta ~ N(m, C)
        thetas = self.rng.multivariate_normal(mean=m, cov=C + 1e-9 * np.eye(self.k), size=n_draws)
        ylog = thetas @ F
        # add obs noise
        ylog = ylog + self.rng.normal(0.0, np.sqrt(self.cfg.obs_var_leads), size=n_draws)
        leads = np.expm1(ylog)
        leads = np.clip(leads, 0.0, None)

        util = float(tm_attempts_total) / float(self.cfg.tm_capacity_per_day) if self.cfg.tm_capacity_per_day > 0 else 0.0
        ub = self.util_bucket(util)

        # For simplicity, split leads by channel proportional to x_media (fallback equal)
        weights = np.array([max(float(x_media_by_ch.get(ch, 0.0)), 0.0) for ch in self.cfg.channels], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        revenue_draws = np.zeros(n_draws)
        for j, ch in enumerate(self.cfg.channels):
            leads_ch = leads * weights[j]

            p_conn = self.state.conn_rate[ch][ub].sample(self.rng, size=n_draws)
            connected = tm_attempts_total * weights[j] * p_conn

            p_close = self.state.close_rate[ch].sample(self.rng, size=n_draws)
            contracts = connected * p_close

            mu_log, sigma2 = self.state.prem_per_contract[ch].sample_mu_sigma(self.rng, size=n_draws)
            prem_per = np.exp(self.rng.normal(mu_log, np.sqrt(sigma2), size=n_draws))

            revenue_draws += contracts * prem_per

        alpha = 0.10
        q_alpha = float(np.quantile(revenue_draws, alpha))
        cvar = float(np.mean(revenue_draws[revenue_draws <= q_alpha]))
        return {
            "rev_mean": float(np.mean(revenue_draws)),
            "rev_p10": float(np.quantile(revenue_draws, 0.10)),
            "rev_p90": float(np.quantile(revenue_draws, 0.90)),
            "rev_cvar10": cvar,
        }
