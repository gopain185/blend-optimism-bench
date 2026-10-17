# blend-optimism-bench

How much does a hill-climb or a stack overfit the out-of-fold (OOF) predictions it is fitted on? This repository
simulates pools of base models for which the **true** AUC of every linear blend is known in closed form, and
measures the gap between a blend's OOF score and its true score.

- **Kaggle notebook:** [Does Your Blend Overfit the OOF? A Simulation](https://www.kaggle.com/code/locdunhat/does-your-blend-overfit-the-oof-a-simulation)
- **Kaggle dataset:** [Blend Overfitting Simulations](https://www.kaggle.com/datasets/locdunhat/blend-overfitting-simulations) (CC0)

## The model

Each row has a label `y` and an unobserved propensity `z = delta * y + u`. Model `k` scores it as
`s_k = z + e_k`. Every model shares `u` (the part of the label nobody can predict). The errors `e` are
correlated through a few random factors. Blending can cancel `e` but not `u`, so no blend passes a ceiling AUC
`Phi(delta / sqrt(2))`. With signal `d` and noise covariance `C`, the true AUC of the blend `w^T s` is

    AUC(w) = Phi( w^T d / sqrt(2 w^T C w) ),   best blend w* = C^{-1} d.

Defaults:

- Single models score 0.958-0.960.
- Their errors correlate at about 0.975.
- The ceiling is 0.961, so the best blend gains about 0.001 over the best model, a Playground-sized gain.

Four rules fit their weights on the OOF rows:

- the best single model;
- the mean of all models, which uses no labels and so cannot overfit;
- a hill-climb (greedy forward selection with replacement);
- an unregularised logistic stack.

**Selection optimism** is a blend's OOF AUC minus its true AUC, minus the same quantity for the mean of all
models. The subtraction removes the luck of the sample, which every rule shares.

## Results

Means over replicates, in units of 1e-5 AUC, with the standard error in brackets.

**1. Pool size** (100k OOF rows):

| models | hill-climb optimism | stack optimism | hill-climb true gain over the mean | stack true gain over the mean |
| --- | --- | --- | --- | --- |
| 2 | 0.6 (1.3) | 0.6 (0.9) | +8.0 (4.9) | +11.3 (4.7) |
| 5 | 0.4 (1.0) | 0.6 (0.7) | +8.7 (2.6) | +11.2 (2.5) |
| 10 | 0.7 (1.1) | 1.0 (1.4) | +11.1 (3.0) | +13.2 (3.1) |
| 20 | 5.4 (0.7) | 5.7 (0.7) | -0.9 (1.3) | +0.5 (1.2) |
| 40 | 8.9 (0.4) | 13.8 (0.5) | -2.8 (1.0) | -4.7 (0.6) |

**2. OOF rows** (16 models):

| rows | hill-climb optimism | stack optimism | hill-climb true gain over the mean | stack true gain over the mean |
| --- | --- | --- | --- | --- |
| 10,000 | 28.2 (1.4) | 47.3 (3.7) | -7.6 (2.6) | -18.0 (3.0) |
| 50,000 | 6.9 (0.9) | 8.0 (1.4) | -2.2 (1.3) | -0.3 (0.9) |
| 200,000 | 0.8 (1.1) | 0.9 (1.1) | +5.0 (1.2) | +5.7 (1.2) |
| 700,000 | 0.2 (0.4) | 0.1 (0.2) | +1.3 (0.7) | +3.1 (1.3) |

**3. Headroom** (16 models, 100k rows): with the ceiling between 0.9605 and 0.963, the optimism stays at
3-6 (about 1 standard error each), whatever the room to blend.

**4. Large stacks on Playground-sized data** (700k rows, logistic stack only, 10 replicates):

| members | stack optimism | stack true gain over the mean |
| --- | --- | --- |
| 40 | 1.2 (0.2) | +2.4 (1.0) |
| 120 | 5.2 (0.2) | -1.9 (0.3) |

**5. Nested CV.** A 5-fold nested estimate of the hill-climb or the stack is between +0.7 and -7.4 away from the truth, usually
negative (slightly pessimistic), because each fold's weights see 80% of the rows.

What it shows:

- **At Playground size the OOF score of a blend is honest.** With 700k OOF rows and 16 models, choosing the weights
  inflates the OOF AUC by about 1e-6, and an unregularised 120-member stack by 5e-5. Weight selection cannot
  explain a 1e-4 gap between a blend's OOF and the public LB.
- **Many models on few rows is where it bites.** At 100k rows the optimism jumps between 10 and 20 models. At 40
  models, both the hill-climb and the stack end up worse in truth than the plain mean, while their OOF says they
  are better. At 10k rows the stack overstates itself by 0.0005 and is truly 0.0002 worse than the mean.
- **Picking the best single model has the same problem** (0.0003 optimism at 10k rows): it is a selection too.
- **Nested CV removes the optimism** and costs only a small pessimistic bias.

## Usage

```bash
pip install -r requirements.txt
python -m blendsim results     # all four experiments, about 20 minutes on a busy CPU
python -m pytest -q            # 10 tests, about 1.5 minutes
python kaggle/build_kaggle.py --reuse-results   # Kaggle dataset + notebook from the saved tables
```

## Layout

```
blendsim/sim.py           pools with a closed-form blend AUC, rank-sum AUC, sampling
blendsim/blend.py         best single, mean, hill-climb, logistic stack, nested estimate
blendsim/experiments.py   the four experiments and their summaries
results/                  raw tables of the run above (seed 20261018)
tests/                    checks of the closed form, the rules and the main claim on small runs
kaggle/                   dataset and notebook builders, datasheet
```

All data is synthetic. License: MIT.
