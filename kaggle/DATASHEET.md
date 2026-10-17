# Blend Overfitting Simulations

Simulated out-of-fold (OOF) predictions of pools of base models, for which the **true** AUC of every linear blend
is known in closed form. Use them to check how much a hill-climb or a stack fitted on OOF predictions overstates
its own score, and to test blending code against ground truth.

Companion notebook: [Does Your Blend Overfit the OOF? A Simulation](https://www.kaggle.com/code/locdunhat/does-your-blend-overfit-the-oof-a-simulation)

## How the data is generated

Each row has a label `y` (positive rate 0.4) and an unobserved propensity `z = delta * y + u` with `u ~ N(0, 1)`.
Model `k` scores the row as `s_k = z + e_k`. The errors `e` are Gaussian with standard deviations `sigma_k` and
are correlated through three random factors (half of their variance) plus independent noise. So the models
share `u` completely and part of `e`. No blend can remove `u`, which puts a ceiling on every blend's AUC
(0.961 by default). Single models score between 0.958 and 0.960.

Let `d` be the signal vector (`delta` for every model) and `C` the full noise covariance (`1 1^T + Cov(e)`). For
a weight vector `w`, the true AUC of the blend `w^T s` is `Phi(w^T d / sqrt(2 w^T C w))`, and the best possible
blend is `w* = C^{-1} d`.

## Files

| file | rows | content |
| --- | --- | --- |
| `pool_oof_scores.csv` | 100,000 | `row_id`, `y`, `model_00` … `model_15` scores (5 decimals) of one 16-model pool |
| `pool_models.csv` | 16 | `model`, `signal` (entry of `d`), `true_auc` (population AUC of the model alone), `optimal_weight` (normalised `w*`) |
| `pool_noise_covariance.csv` | 16 | the exact noise covariance `C`, models as rows and columns |
| `exp1_pool_size.csv` | 160 | one row per replicate and rule, pools of 2 to 40 models, 100k rows |
| `exp2_n_rows.csv` | 100 | one row per replicate and rule, 16 models, 10k to 700k rows |
| `exp3_headroom.csv` | 128 | one row per replicate and rule, 16 models, 100k rows, ceilings 0.9605 to 0.963 |

Columns of the experiment tables:

- `rule`: best single, mean of all, hill-climb, logistic stack.
- `k`, `n`, `ceiling`, `rep`: the setting and replicate.
- `oof`: the blend's AUC on the rows its weights were fitted on.
- `true`: its population AUC.
- `best_single_true`: the best model's population AUC.
- `oracle`: the AUC of `w*`.
- `nested`: a 5-fold nested-CV estimate, only where it was computed.
- Derived columns: `optimism` (= `oof - true`), `selection_optimism` (optimism minus that of the mean of all models, which uses no labels), `gain_oof`, `gain_true`, `regret` (= `oracle - true`), `nested_bias`.

## Provenance

Everything is simulated by the `blendsim` package (repository `blend-optimism-bench` by Loc Nguyen, MIT), seed
`20261018`. The notebook embeds the full source and regenerates every table. No real people, competitions or
submissions are involved.

## Intended use

Teaching and checking intuitions about blend overfitting: how many models and how many OOF rows make a
hill-climb or a stack noticeably optimistic, whether nested CV fixes it, and how much a blend really gains. The
pool file is also a test case for blending code, since any weight vector can be scored against the truth.

## Limitations

- Scores are Gaussian given the label and blends are linear. Real models have heavier tails, and rank or logit
  blends are not exactly linear in the scores.
- The base models' own OOF predictions are honest by construction: only the choice of blend weights can overfit.
  A leaky base model adds optimism this data does not contain.
- One metric (ROC AUC) and one prevalence (0.4).

## License

CC0 1.0.
