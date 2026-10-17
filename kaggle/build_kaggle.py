"""Builds the Kaggle dataset and notebook for this project.

    python kaggle/build_kaggle.py [--public] [--notebook-only]

writes kaggle/dataset/ (one simulated model pool with its exact covariance, the result tables of the three
experiments and the datasheet) and kaggle/notebook/ (a CPU notebook that embeds the package source and reruns
the experiments).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from blendsim.experiments import SEED, headroom, large_pool, n_rows, pool_size  # noqa: E402
from blendsim.sim import make_pool  # noqa: E402

OUT = ROOT / "kaggle"
DATASET_ID = "locdunhat/blend-overfitting-simulations"
NOTEBOOK_ID = "locdunhat/does-your-blend-overfit-the-oof-a-simulation"
TITLE = "Does Your Blend Overfit the OOF? A Simulation"
K, N = 16, 100_000


def build_dataset(public: bool, rerun: bool = True) -> Path:
    out = OUT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([SEED, 0])
    pool = make_pool(rng, K)
    y, S = pool.sample(rng, N)
    names = [f"model_{j:02d}" for j in range(K)]
    rows = pd.concat([pd.DataFrame({"row_id": np.arange(N), "y": y.astype(int)}),
                      pd.DataFrame(S.round(5), columns=names)], axis=1)
    rows.to_csv(out / "pool_oof_scores.csv", index=False)
    pd.DataFrame({"model": names, "signal": pool.d, "true_auc": pool.single_aucs(),
                  "optimal_weight": pool.optimal_weights()}).to_csv(out / "pool_models.csv", index=False)
    pd.DataFrame(pool.cov, index=names, columns=names).to_csv(out / "pool_noise_covariance.csv")
    if rerun:
        for name, fn in [("exp1_pool_size", pool_size), ("exp2_n_rows", n_rows), ("exp3_headroom", headroom),
                         ("exp4_large_pool", large_pool)]:
            fn().to_csv(out / f"{name}.csv", index=False)
    else:  # reuse the tables written by `python -m blendsim results`
        for f in sorted((ROOT / "results").glob("exp*.csv")):
            (out / f.name).write_bytes(f.read_bytes())
    datasheet = (OUT / "DATASHEET.md").read_text(encoding="utf-8")
    (out / "DATASHEET.md").write_text(datasheet, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "Blend Overfitting Simulations",
        "id": DATASET_ID,
        "subtitle": "Simulated OOF model pools where the true AUC of every blend is known exactly",
        "description": datasheet,
        "isPrivate": not public,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["tabular", "benchmark", "classification"],
    }, indent=2), encoding="utf-8")
    return out


def _sources() -> dict[str, str]:
    files = sorted((ROOT / "blendsim").glob("*.py"))
    return {f.relative_to(ROOT).as_posix(): f.read_text(encoding="utf-8") for f in files}


def _cells() -> list[tuple[str, str]]:
    return [
        ("markdown", (OUT / "notebook_intro.md").read_text(encoding="utf-8")),
        ("code", "SOURCES = " + json.dumps(_sources(), indent=0) + """

import os, sys
ROOT = "/kaggle/working/blend-optimism-bench" if os.path.isdir("/kaggle/working") else os.path.abspath("blend-optimism-bench")
for path, text in SOURCES.items():
    os.makedirs(os.path.dirname(os.path.join(ROOT, path)), exist_ok=True)
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text)
sys.path.insert(0, ROOT)
"""),
        ("code", """import time, warnings
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from blendsim.experiments import pool_size, n_rows, headroom, large_pool, summarise
pd.set_option("display.precision", 6)
"""),
        ("markdown", "## 1. More candidate models\n\nPools of 2 to 40 models, 100k OOF rows each. Every rule "
                     "picks its weights on the OOF rows. `selection_optimism` is how much the OOF score of the "
                     "chosen blend overstates its true AUC, beyond the luck of the sample that all rules share. "
                     "`gain_oof` and `gain_true` are the gain over the best single model as the OOF shows it and "
                     "as it really is; `nested_bias` is the error of a nested-CV estimate of the blend."),
        ("code", "t = time.time(); e1 = pool_size(); print(round(time.time() - t), 's'); s1 = summarise(e1, 'k'); s1"),
        ("code", """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for rule, g in s1[s1.rule != "mean of all"].groupby("rule", sort=False):
    axes[0].plot(g.k, g.selection_optimism * 1e4, marker="o", label=rule)
for rule, g in s1.groupby("rule", sort=False):
    axes[1].plot(g.k, g.gain_true * 1e4, marker="o", label=rule)
axes[0].set(xscale="log", xlabel="models in the pool", ylabel="OOF - true AUC (1e-4)", title="optimism from choosing the weights")
axes[1].set(xscale="log", xlabel="models in the pool", ylabel="true gain over best single (1e-4)", title="what the blend really gains")
axes[0].axhline(0, color="k", lw=0.8); axes[0].legend(); axes[1].legend(); plt.tight_layout(); plt.show()
"""),
        ("markdown", "## 2. More OOF rows\n\n16 models, from 10k to 700k OOF rows (the size of a Playground "
                     "training set)."),
        ("code", "t = time.time(); e2 = n_rows(); print(round(time.time() - t), 's'); summarise(e2, 'n')"),
        ("markdown", "## 3. More room to blend\n\nThe ceiling is the best AUC any blend could reach. The higher it "
                     "is above the best single model (0.96), the more the models' own errors differ."),
        ("code", "t = time.time(); e3 = headroom(); print(round(time.time() - t), 's'); summarise(e3, 'ceiling')"),
        ("markdown", "## 4. Large stacks on 700k rows\n\nThe public OOF libraries of Playground competitions get "
                     "stacked 100+ members at a time. Here: 40 and 120 members, 700k OOF rows, plain mean against an "
                     "unregularised logistic stack."),
        ("code", "t = time.time(); e4 = large_pool(); print(round(time.time() - t), 's'); summarise(e4, 'k')"),
        ("markdown", "## 5. The pool in the dataset\n\nThe companion dataset holds one pool: 16 models' OOF "
                     "scores on 100k rows, and the exact noise covariance, so the true AUC of any blend you "
                     "build on it can be computed. Here a hill-climb and a logistic stack are fitted to it."),
        ("code", """import glob
from scipy.stats import norm
from blendsim.blend import hill_climb, logistic_stack
from blendsim.sim import auc
hits = glob.glob("/kaggle/input/**/pool_oof_scores.csv", recursive=True)
if hits:
    base = os.path.dirname(hits[0])
    rows = pd.read_csv(hits[0]); models = pd.read_csv(os.path.join(base, "pool_models.csv"))
    cov = pd.read_csv(os.path.join(base, "pool_noise_covariance.csv"), index_col=0).to_numpy()
    y, S = rows.y.to_numpy(), rows[models.model].to_numpy()
    true_auc = lambda w: norm.cdf(w @ models.signal.to_numpy() / np.sqrt(2 * w @ cov @ w))
    for name, w in [("best single", np.eye(len(models))[models.true_auc.idxmax()]), ("mean of all", np.ones(len(models))),
                    ("hill-climb", hill_climb(y, S)), ("logistic stack", logistic_stack(y, S)),
                    ("optimal", models.optimal_weight.to_numpy())]:
        print(f"{name:<15} OOF {auc(y, S @ w):.6f}   true {true_auc(w):.6f}")
else:
    print("dataset not attached")
"""),
        ("markdown", (OUT / "notebook_findings.md").read_text(encoding="utf-8")),
    ]


def build_notebook(public: bool) -> Path:
    out = OUT / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (kind, src) in enumerate(_cells()):
        c = {"cell_type": kind, "id": f"cell-{i}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        cells.append(c)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (out / "blend-optimism.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": NOTEBOOK_ID, "title": TITLE, "code_file": "blend-optimism.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": not public, "enable_gpu": False, "enable_internet": False,
        "dataset_sources": [DATASET_ID], "competition_sources": [], "kernel_sources": [],
        "keywords": ["evaluation", "benchmark"],
    }, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    public = "--public" in sys.argv
    if "--notebook-only" not in sys.argv:
        print("wrote", build_dataset(public, rerun="--reuse-results" not in sys.argv))
    print("wrote", build_notebook(public))
