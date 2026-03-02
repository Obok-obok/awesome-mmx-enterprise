from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DLMState:
    """Filtered state for a Dynamic Linear Model.

    We model:
        y_t = F_t' * theta_t + v_t,   v_t ~ N(0, R)
        theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q)

    Where selected components of theta_t are allowed to drift (random walk).
    """

    m: np.ndarray  # mean (k,)
    C: np.ndarray  # covariance (k,k)
    R: float
    Q: np.ndarray  # (k,k)


def build_Q(k: int, drift_idx: List[int], q_drift: float, q_static: float = 0.0) -> np.ndarray:
    Q = np.eye(k) * q_static
    for i in drift_idx:
        Q[i, i] = q_drift
    return Q


def kalman_predict(state: DLMState) -> Tuple[np.ndarray, np.ndarray]:
    a = state.m
    R = state.C + state.Q
    return a, R


def kalman_update(a: np.ndarray, R: np.ndarray, F: np.ndarray, y: float, obs_var: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """One-step update.

    Returns (m, C, f, q) where f is predictive mean and q predictive variance.
    """
    F = F.reshape(-1, 1)
    f = float((F.T @ a.reshape(-1, 1)).squeeze())
    q = float((F.T @ R @ F).squeeze() + obs_var)
    e = y - f
    A = (R @ F) / q  # Kalman gain (k,1)
    m = a.reshape(-1, 1) + A * e
    C = R - A @ A.T * q
    return m.squeeze(), C, f, q


def run_filter(
    y: np.ndarray,
    Fmat: np.ndarray,
    m0: np.ndarray,
    C0: np.ndarray,
    Q: np.ndarray,
    obs_var: float,
) -> Dict[str, np.ndarray]:
    """Run Kalman filter for a DLM.

    Args:
        y: (T,) observed series
        Fmat: (T,k) design
        m0, C0: initial state
        Q: process noise
        obs_var: observation variance (scalar)

    Returns arrays of filtered means/covs and 1-step predictions.
    """
    T, k = Fmat.shape
    m = m0.copy()
    C = C0.copy()

    ms = np.zeros((T, k))
    Cs = np.zeros((T, k, k))
    fhat = np.zeros(T)
    qhat = np.zeros(T)

    for t in range(T):
        a, R = m, C + Q
        F = Fmat[t]
        m, C, f, q = kalman_update(a, R, F, float(y[t]), obs_var)
        ms[t] = m
        Cs[t] = C
        fhat[t] = f
        qhat[t] = q

    return {"m": ms, "C": Cs, "f": fhat, "q": qhat}


def online_step(state: DLMState, F: np.ndarray, y: float) -> Tuple[DLMState, float, float]:
    a, R = kalman_predict(state)
    m, C, f, q = kalman_update(a, R, F, y, state.R)
    return DLMState(m=m, C=C, R=state.R, Q=state.Q), f, q
