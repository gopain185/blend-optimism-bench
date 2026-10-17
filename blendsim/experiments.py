"""The experiments behind the README. Each returns one row per replicate and rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .blend import RULES, nested_estimate
from .sim import auc, make_pool

SEED = 20261018


def run_one(rng: np.random.Generator, k: int, n: int, ceiling: float = 0.961, nested: bool = True) -> list[dict]:
    pool = make_pool(rng, k, ceiling=ceiling)
    y, S = pool.sample(rng, n)
    best_true = pool.single_aucs().max()
    rows = []
    for name, rule in RULES.items():
        w = rule(y, S)
        row = dict(rule=name, k=k, n=n, ceiling=ceiling, oof=auc(y, S @ w), true=pool.true_auc(w),
                   best_single_true=best_true, oracle=pool.optimal_auc())
        if nested and name in ("hill-climb", "logistic stack"):
            row["nested"] = nested_estimate(rule, y, S, seed=int(rng.integers(1 << 31)))
        rows.append(row)
    return rows


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["optimism"] = df.oof - df.true
    # The mean of all models uses no labels, so its OOF - true has expectation zero; subtracting it removes
    # the luck of the sample that every rule shares and leaves the optimism caused by choosing the weights.
    base = df[df.rule == "mean of all"].set_index(["k", "n", "ceiling", "rep"]).optimism
    df["selection_optimism"] = df.optimism - base.reindex(pd.MultiIndex.from_frame(df[["k", "n", "ceiling", "rep"]])).to_numpy()
    df["gain_true"] = df.true - df.best_single_true
    df["gain_oof"] = df.groupby(["k", "n", "ceiling", "rep"]).oof.transform(lambda s: s - s.iloc[0])
    df["regret"] = df.oracle - df.true
    if "nested" in df:
        # same correction: the nested estimate is computed on the same sample, so it shares that sample's luck
        df["nested_bias"] = df.nested - df.true - (df.optimism - df.selection_optimism)
    return df


def pool_size(ks=(2, 5, 10, 20, 40), n: int = 100_000, reps: int = 8, seed: int = SEED) -> pd.DataFrame:
    """How optimistic is the OOF score of a blend as the pool of candidate models grows?"""
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        for rep in range(reps):
            rows += [dict(r, rep=rep) for r in run_one(rng, k, n, nested=k <= 20)]
    return _frame(rows)


def n_rows(ns=(10_000, 50_000, 200_000, 700_000), k: int = 16, reps=(10, 8, 4, 3), seed: int = SEED + 1) -> pd.DataFrame:
    """The same question by training-set size, ending at a Playground-sized 700k rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for n, r in zip(ns, reps):
        for rep in range(r):
            rows += [dict(x, rep=rep) for x in run_one(rng, k, n, nested=n <= 200_000)]
    return _frame(rows)


def headroom(ceilings=(0.9605, 0.961, 0.9615, 0.963), k: int = 16, n: int = 100_000, reps: int = 8, seed: int = SEED + 2) -> pd.DataFrame:
    """Does blending gain more, and overfit more, when there is more room above the best model?"""
    rng = np.random.default_rng(seed)
    rows = []
    for ceiling in ceilings:
        for rep in range(reps):
            rows += [dict(x, rep=rep) for x in run_one(rng, k, n, ceiling=ceiling, nested=False)]
    return _frame(rows)


def large_pool(ks=(40, 120), n: int = 700_000, reps: int = 10, seed: int = SEED + 3) -> pd.DataFrame:
    """Large public-OOF stacks on Playground-sized data: only the mean and the logistic stack (a hill-climb over
    120 models is too slow to replicate)."""
    from .blend import logistic_stack, mean_all

    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        for rep in range(reps):
            pool = make_pool(rng, k)
            y, S = pool.sample(rng, n)
            for name, rule in (("mean of all", mean_all), ("logistic stack", logistic_stack)):
                w = rule(y, S)
                rows.append(dict(rule=name, k=k, n=n, ceiling=0.961, rep=rep, oof=auc(y, S @ w), true=pool.true_auc(w),
                                 best_single_true=pool.single_aucs().max(), oracle=pool.optimal_auc()))
    df = pd.DataFrame(rows)
    df["optimism"] = df.oof - df.true
    base = df[df.rule == "mean of all"].set_index(["k", "rep"]).optimism
    df["selection_optimism"] = df.optimism - base.reindex(pd.MultiIndex.from_frame(df[["k", "rep"]])).to_numpy()
    df["gain_true"] = df.true - df.best_single_true
    df["regret"] = df.oracle - df.true
    return df


def summarise(df: pd.DataFrame, by: str) -> pd.DataFrame:
    cols = {c: "mean" for c in ("selection_optimism", "gain_oof", "gain_true", "regret") if c in df}
    if "nested_bias" in df:
        cols["nested_bias"] = "mean"
    out = df.groupby([by, "rule"], sort=False).agg(cols).reset_index()
    return out.round(6)
