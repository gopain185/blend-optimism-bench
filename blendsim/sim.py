"""Pools of base models with Gaussian scores, where the true AUC of every linear blend is known exactly.

Each row has a label y and an unobserved propensity z = delta * y + u, u ~ N(0, 1). The noise u is the part of
the label no model can predict. Model k sees z through its own error:

    s_k = z + e_k,    e ~ N(0, D (f * A A^T + (1 - f) * I) D),    D = diag(sigma_k), rows of A unit-norm,

so the models share u completely, and share part of their own errors through a few common factors (weight
f). Blending can cancel e but never u, so no blend beats the ceiling AUC Phi(delta / sqrt(2)). With signal
vector d (here delta for every model) and total noise covariance C = 1 1^T + Cov(e), the blend w^T s is
Gaussian within each class and its population AUC is

    AUC(w) = Phi( w^T d / sqrt(2 w^T C w) ),

and the best possible linear blend is w* = C^{-1} d (Fisher's discriminant).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC by the rank-sum formula. Scores are continuous, so ties have probability zero."""
    y = np.asarray(y, dtype=bool)
    ranks = np.empty(len(s))
    ranks[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_to_d(a: float) -> float:
    """Signal-to-noise ratio of a binormal score with population AUC a."""
    return float(np.sqrt(2.0) * norm.ppf(a))


@dataclass
class Pool:
    d: np.ndarray    # (k,) signal of each model
    cov: np.ndarray  # (k, k) noise covariance

    @property
    def k(self) -> int:
        return len(self.d)

    def true_auc(self, w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        return float(norm.cdf(w @ self.d / np.sqrt(2.0 * w @ self.cov @ w)))

    def single_aucs(self) -> np.ndarray:
        return np.array([self.true_auc(np.eye(self.k)[i]) for i in range(self.k)])

    def optimal_weights(self) -> np.ndarray:
        w = np.linalg.solve(self.cov, self.d)
        return w / np.abs(w).sum()

    def optimal_auc(self) -> float:
        return self.true_auc(self.optimal_weights())

    def sample(self, rng: np.random.Generator, n: int, prevalence: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
        y = (rng.random(n) < prevalence).astype(np.int8)
        noise = rng.standard_normal((n, self.k)) @ np.linalg.cholesky(self.cov + 1e-12 * np.eye(self.k)).T
        return y, (y[:, None] * self.d + noise).astype(np.float32)


def make_pool(rng: np.random.Generator, k: int, best_auc: float = 0.96, spread: float = 0.002,
              ceiling: float = 0.961, f: float = 0.5, n_factors: int = 3) -> Pool:
    """k models whose single AUCs lie in [best_auc - spread, best_auc], under a ceiling AUC no blend can pass.

    The default ceiling leaves about 0.001 between the best model and the best possible blend; Playground
    blends typically gain a few 1e-4 over their best single model. Factor loadings are random unit vectors,
    so the errors of each pair of models are correlated differently.
    """
    assert ceiling > best_auc
    delta = auc_to_d(ceiling)
    aucs = best_auc - spread * rng.random(k)
    aucs[0] = best_auc
    sigma = np.sqrt((delta / np.array([auc_to_d(a) for a in aucs])) ** 2 - 1.0)
    A = rng.standard_normal((k, n_factors))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    cov_e = np.outer(sigma, sigma) * (f * A @ A.T + (1.0 - f) * np.eye(k))
    return Pool(d=np.full(k, delta), cov=np.ones((k, k)) + cov_e)
