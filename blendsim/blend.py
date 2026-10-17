"""Blending rules as used in tabular competitions. Each takes OOF scores and returns linear blend weights."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .sim import auc


def best_single(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    w = np.zeros(S.shape[1])
    w[np.argmax([auc(y, S[:, j]) for j in range(S.shape[1])])] = 1.0
    return w


def mean_all(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.full(S.shape[1], 1.0 / S.shape[1])


def hill_climb(y: np.ndarray, S: np.ndarray, max_steps: int = 25) -> np.ndarray:
    """Greedy forward selection with replacement (Caruana et al., 2004): start from the best model, then keep
    adding the model that most improves the AUC of the running average, until nothing improves it."""
    counts = best_single(y, S)
    best = auc(y, S @ counts)
    for _ in range(max_steps):
        n = counts.sum()
        scores = [auc(y, (S @ counts + S[:, j]) / (n + 1)) for j in range(S.shape[1])]
        j = int(np.argmax(scores))
        if scores[j] <= best:
            break
        counts[j] += 1
        best = scores[j]
    return counts / counts.sum()


def logistic_stack(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Unregularised logistic regression on the OOF scores; its decision function is linear in them."""
    model = LogisticRegression(C=1e4, max_iter=2000).fit(S, y)
    return model.coef_[0]


RULES = {"best single": best_single, "mean of all": mean_all, "hill-climb": hill_climb, "logistic stack": logistic_stack}


def nested_estimate(rule, y: np.ndarray, S: np.ndarray, n_folds: int = 5, seed: int = 0) -> float:
    """Honest estimate: fit the blend weights on n_folds - 1 folds, score the held-out fold, average."""
    fold = np.random.default_rng(seed).integers(0, n_folds, len(y))
    out = []
    for f in range(n_folds):
        tr, va = fold != f, fold == f
        out.append(auc(y[va], S[va] @ rule(y[tr], S[tr])))
    return float(np.mean(out))
