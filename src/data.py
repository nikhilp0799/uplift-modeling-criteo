"""
Criteo Uplift dataset loading.

~13.9M rows from a randomised advertising experiment. Columns:
    f0..f11    anonymised features
    treatment  1 = shown the ad, 0 = held out
    visit      did the user visit
    conversion did the user convert
    exposure   whether the ad was actually served (ignore; use `treatment`)

Download: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
Place criteo-uplift-v2.1.csv in data/raw/.

Note on the treatment split: the experiment is heavily imbalanced (~85%
treated). That is fine for estimating uplift but means the control model
sees far fewer rows, which is the main source of variance in a T-learner
here. Worth reporting rather than silently rebalancing.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/criteo-uplift-v2.1.csv")
FEATURES = [f"f{i}" for i in range(12)]


@dataclass
class UpliftData:
    X_train: np.ndarray
    X_test: np.ndarray
    t_train: np.ndarray          # treatment assignment
    t_test: np.ndarray
    y_train: np.ndarray          # observed outcome
    y_test: np.ndarray
    outcome: str


def load(outcome: str = "visit", sample: int | None = 2_000_000,
         test_frac: float = 0.3, seed: int = 0) -> UpliftData:
    """Load and split.

    outcome: 'visit' (~4.7% base rate) or 'conversion' (~0.29%).
             Start with visit — conversion is so sparse that uplift
             estimates are dominated by noise unless you use the full 13.9M
             rows, and even then the confidence intervals are wide.

    sample:  None loads all ~13.9M rows. 2M is enough to get stable Qini on
             `visit` and keeps iteration fast.

    The split is random, not temporal: treatment was randomised, so there is
    no time-ordering to respect. What matters is that a user appears in
    exactly one side.
    """
    df = pd.read_csv(RAW)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]

    X = df[FEATURES].to_numpy(dtype=np.float32)
    t = df["treatment"].to_numpy(dtype=np.int8)
    y = df[outcome].to_numpy(dtype=np.int8)

    return UpliftData(
        X_train=X[tr], X_test=X[te],
        t_train=t[tr], t_test=t[te],
        y_train=y[tr], y_test=y[te],
        outcome=outcome,
    )


def describe(d: UpliftData) -> dict:
    """Sanity numbers to print before modelling.

    The naive difference in outcome rates between treated and control is the
    Average Treatment Effect. Any uplift model that cannot beat targeting at
    random is not finding heterogeneity — it is just rediscovering the ATE.
    """
    tr_rate = float(d.y_train[d.t_train == 1].mean())
    ct_rate = float(d.y_train[d.t_train == 0].mean())
    return {
        "outcome": d.outcome,
        "n_train": int(len(d.y_train)),
        "n_test": int(len(d.y_test)),
        "treated_frac": float(d.t_train.mean()),
        "outcome_rate_treated": tr_rate,
        "outcome_rate_control": ct_rate,
        "average_treatment_effect": tr_rate - ct_rate,
    }
