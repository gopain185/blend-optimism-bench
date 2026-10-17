## What the simulation shows

All numbers are true AUC in units of 1e-5, from the tables above (8-10 replicates per setting at most sizes,
3-4 at the largest).

- **At Playground size, a blend's OOF score is honest.** With 700k OOF rows and 16 models, fitting the weights
  inflates the OOF AUC by about 0.1-0.2 (standard error 0.2-0.4). If your blend's OOF is 1e-4 above its public LB,
  weight selection is not the reason. Look at the base models and at the train/test match instead.
- **Many models on few rows is where it bites.** At 100k rows, the optimism stays below about 1 up to 10 models,
  then rises to 5-6 at 20 models and to 9-14 at 40. At 40 models, the hill-climb and the stack are truly worse
  than the plain mean of all models (-3 and -5), although their OOF says they are better.
- **Small data is worse still.** At 10k rows the stack overstates itself by 47 and is truly 18 worse than the mean.
- **The best single model is a selection too.** Choosing it by OOF overstates it by about 33 at 10k rows.
- **Nested CV fixes it.** Fitting the weights on four folds and scoring the fifth gives an estimate within a few
  1e-5 of the truth, slightly pessimistic.

What to do with it:

- On large Playground data, trust the OOF score of a hill-climb or stack.
- On small data or with large pools, use nested CV or a regularised stack, and compare against the plain mean.
- Don't blame blend overfitting for an OOF-to-LB gap it cannot produce.
