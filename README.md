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

*(Fill from `results/` once runs complete. Report AUUC for each learner
against the random-targeting control, the top- and bottom-decile observed
uplift, and variance across seeds. If a learner fails to beat random, say
so — that is a finding about the method on this data, not a failure of the
project.)*

| Learner | AUUC | vs random | Top decile | Bottom decile |
|---|---|---|---|---|
| Random | | — | | |
| S-learner | | | | |
| T-learner | | | | |
| X-learner | | | | |

## Data

Criteo Uplift Prediction Dataset —
https://ailab.criteo.com/criteo-uplift-prediction-dataset/
Diemert et al., *A Large Scale Benchmark for Uplift Modeling*, AdKDD 2018.
