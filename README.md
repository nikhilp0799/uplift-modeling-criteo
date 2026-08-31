# Uplift Modeling: Who Does the Treatment Actually Change?

Meta-learners on the Criteo Uplift dataset — ~13.9M users from a randomised
advertising experiment — estimating the *incremental* effect of treatment
rather than the probability of the outcome.

The distinction matters commercially. A conversion model ranks users by how
likely they are to convert, which puts the people who would have converted
anyway at the top. Spending on them changes nothing. An uplift model ranks
by how much the treatment *changes* behaviour, separating:

- **Persuadables** — respond only if treated. The only group worth targeting.
- **Sure things** — respond either way. Budget wasted.
- **Lost causes** — never respond. Budget wasted.
- **Sleeping dogs** — respond *unless* treated. Targeting them is harmful.

---

## Why this is harder than classification

The individual treatment effect is never observable. A user is either
treated or not, so the counterfactual outcome does not exist in the data and
there is no per-row label to score against. That single fact drives every
design choice here:

**Evaluation is rank-based.** Qini and AUUC measure whether targeting the
top-k users by predicted uplift produces more incremental outcomes than
targeting at random. No per-user accuracy exists to compute.

**A random-targeting control is mandatory.** AUUC is uninterpretable on its
own — a model that simply rediscovers the average treatment effect scores
positive. The random baseline is what makes the number mean something.

**Decile lift is the honesty check.** If the top predicted decile does not
show a larger observed treated-minus-control gap than the bottom decile,
the ranking is not capturing treatment effect regardless of AUUC.

---

## Learners

| | Approach | Expected weakness on this data |
|---|---|---|
| **S-learner** | One model, treatment as a feature | Boosting can ignore the treatment column when outcome signal dominates the effect — and here it does |
| **T-learner** | Separate model per arm, take the difference | Control arm is ~15% of rows, so its variance dominates |
| **X-learner** | Impute the missing counterfactual, model the imputed effect, blend by propensity | Designed for unbalanced arms; most likely to win here |

---

## Layout

```
src/data.py       loading, train/test split, ATE sanity numbers
src/learners.py   S-, T- and X-learners
src/evaluate.py   Qini curve, AUUC vs random, decile lift
src/train.py      comparison entry point
results/          one JSON report per learner per seed
```

## Running

```bash
pip install pandas numpy scikit-learn xgboost
# place criteo-uplift-v2.1.csv in data/raw/

python -m src.train --learner all --outcome visit --seeds 3
```

Start with `visit` (~4.7% base rate). `conversion` is ~0.29%, so uplift
estimates there are dominated by noise unless you use the full dataset.

---

## Results

`visit` outcome, 2M-row sample, 3 seeds (0-2), 70/30 train/test split.
ATE on this sample was ~1.0-1.5% depending on seed. Values below are
mean across the 3 seeds, with the seed-to-seed spread; raw JSON per
learner per seed is in `results/`.

| Learner | AUUC (mean ± std) | Beats random? | Top decile (mean) | Bottom decile (mean) |
|---|---|---|---|---|
| Random | 28.8 ± 91.6 | control | +0.0117 | +0.0109 |
| S-learner | 1937.4 ± 15.5 | Yes | +0.0579 | −0.0012 |
| T-learner | 1694.3 ± 85.8 | Yes | +0.0564 | +0.0068 |
| X-learner | 1709.8 ± 167.9 | Yes | +0.0570 | +0.0048 |

Random targeting scores an AUUC of 28.8 with a standard deviation of 91.6 —
indistinguishable from zero, which is what a metric built to be null under
no signal should produce. All three meta-learners land between 1,600 and
1,950, well outside that noise band. No ratio is quoted against random
because dividing by a quantity consistent with zero produces a meaningless
multiple.

The S-learner posted
the highest mean AUUC and was also the most stable across seeds (±15,
versus ±86 for T and ±168 for X) — the opposite of this file's own
prediction that boosting would ignore the treatment column and collapse to
the average treatment effect. It did not: its predicted-uplift standard
deviation (~0.023) is the same order of magnitude as T's and X's (~0.03),
meaning it genuinely used the treatment feature to produce per-user
heterogeneity rather than a single global shift. T and X, meanwhile,
overlap enough in AUUC across seeds that neither is clearly ahead of the
other here. The decile check confirms real ranking signal for all three
learners: top-decile observed uplift (0.056-0.058) sits roughly an order
of magnitude above bottom-decile uplift (0.005-0.007 for T and X), while
random's own top and bottom deciles (0.0117 vs 0.0109) are statistically
indistinguishable, exactly as expected when the ranking carries no
information. The S-learner is the one case with a genuine sleeping-dogs
signal: its bottom decile averages -0.0012 across seeds (negative in 2 of
3), meaning the lowest-ranked ~10% of users show a small net negative
response to the ad — worth flagging even though the effect is small and
its sign flips in one of the three seeds.

## Data

Criteo Uplift Prediction Dataset —
https://ailab.criteo.com/criteo-uplift-prediction-dataset/
Diemert et al., *A Large Scale Benchmark for Uplift Modeling*, AdKDD 2018.
