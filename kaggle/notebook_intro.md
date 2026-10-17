# Does Your Blend Overfit the OOF? A Simulation

A hill-climb or a stack picks its weights on the same out-of-fold (OOF) predictions that it then reports a
score on. That score has to be a little optimistic. How much? Enough to matter at the fourth decimal, which
is where Playground leaderboards are decided? In a real competition nobody can say, because nobody knows a
blend's true AUC. In a simulation you can.

This notebook builds pools of base models whose scores are Gaussian given the label, so the population AUC of
**every linear blend is known in closed form**:

    s_k = z + e_k,   z = delta * y + u,   AUC(w) = Phi( w^T d / sqrt(2 w^T C w) )

`u` is noise every model shares (the part of the label nobody can predict), `e_k` are the models' own errors,
correlated through a few common factors. Blending can cancel `e` but not `u`, so there is a ceiling no blend
passes. With the defaults, single models score 0.958-0.960, their errors correlate at about 0.975, and the best
possible blend gains about 0.001: the scale of a Playground competition.

Four rules are compared: the best single model, the plain mean of all models (it uses no labels, so it cannot
overfit), a hill-climb (greedy forward selection with replacement) and an unregularised logistic stack. For each
one the notebook measures:

1. how much its OOF score overstates its true AUC, as the pool grows from 2 to 40 models;
2. how that changes with 10k to 700k OOF rows;
3. how it depends on how much room there is to blend;
4. what happens with 40 and 120 members on 700k rows, the size of the public OOF stacks;
5. whether a nested-CV estimate of the blend removes the bias.

Everything runs on CPU in about 25 minutes, and the package source is embedded below.
