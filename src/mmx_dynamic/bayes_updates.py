from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class BetaPosterior:
    a: float
    b: float

    def update(self, successes: float, trials: float) -> "BetaPosterior":
        return BetaPosterior(a=self.a + float(successes), b=self.b + float(trials - successes))

    def mean(self) -> float:
        return self.a / (self.a + self.b) if (self.a + self.b) > 0 else 0.0

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        return rng.beta(self.a, self.b, size=size)


@dataclass
class NIGPosterior:
    """Normal-Inverse-Gamma prior for Normal with unknown mean/variance.

    For x ~ Normal(mu, sigma^2).
    Prior:
        mu | sigma^2 ~ Normal(m0, sigma^2/k0)
        sigma^2 ~ Inv-Gamma(a0, b0)
    """

    m: float
    k: float
    a: float
    b: float

    def update(self, x: np.ndarray) -> "NIGPosterior":
        x = np.asarray(x, dtype=float)
        n = x.size
        if n == 0:
            return self
        xbar = float(x.mean())
        ss = float(((x - xbar) ** 2).sum())
        k_n = self.k + n
        m_n = (self.k * self.m + n * xbar) / k_n
        a_n = self.a + n / 2.0
        b_n = self.b + 0.5 * ss + (self.k * n * (xbar - self.m) ** 2) / (2.0 * k_n)
        return NIGPosterior(m=m_n, k=k_n, a=a_n, b=b_n)

    def mean_mu(self) -> float:
        return self.m

    def mean_sigma2(self) -> float:
        # mean of Inv-Gamma(a,b) is b/(a-1) for a>1
        return self.b / (self.a - 1.0) if self.a > 1.0 else self.b

    def sample_mu_sigma(self, rng: np.random.Generator, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        # sample sigma2 from Inv-Gamma
        sigma2 = 1.0 / rng.gamma(shape=self.a, scale=1.0 / self.b, size=size)
        mu = rng.normal(loc=self.m, scale=np.sqrt(sigma2 / self.k), size=size)
        return mu, sigma2


def make_rate_posteriors(channels: list[str], util_buckets: list[str]) -> Dict[str, Dict[str, BetaPosterior]]:
    """Create nested posteriors for connected rate by channel x util bucket."""
    out: Dict[str, Dict[str, BetaPosterior]] = {}
    for ch in channels:
        out[ch] = {}
        for ub in util_buckets:
            out[ch][ub] = BetaPosterior(a=2.0, b=8.0)  # mildly conservative prior
    return out
