"""Fit and compare S-, T- and X-learners against a random-targeting control.

Usage:
    python -m src.train --learner t --outcome visit
    python -m src.train --learner all --outcome visit --seeds 3
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import describe, load
from .evaluate import evaluate
from .learners import LEARNERS

RESULTS = Path("results")


def run_one(name: str, d, seed: int) -> dict:
    np.random.seed(seed)
    model = LEARNERS[name]().fit(d.X_train, d.t_train, d.y_train)
    uplift = model.predict(d.X_test)
    report = evaluate(d.y_test, d.t_test, uplift)
    report["learner"] = name
    report["seed"] = seed
    report["predicted_uplift_mean"] = float(uplift.mean())
    report["predicted_uplift_std"] = float(uplift.std())
    return report


def run_random(d, seed: int) -> dict:
    """Control: rank users at random.

    Any learner that does not beat this has found no heterogeneity. Without
    it, a positive AUUC is uninterpretable.
    """
    rng = np.random.default_rng(seed)
    uplift = rng.normal(size=len(d.y_test))
    report = evaluate(d.y_test, d.t_test, uplift)
    report["learner"] = "random"
    report["seed"] = seed
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--learner", default="all",
                   choices=["s", "t", "x", "random", "all"])
    p.add_argument("--outcome", default="visit",
                   choices=["visit", "conversion"])
    p.add_argument("--sample", type=int, default=2_000_000)
    p.add_argument("--seeds", type=int, default=1)
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    names = ["random", "s", "t", "x"] if args.learner == "all" else [args.learner]

    for seed in range(args.seeds):
        d = load(outcome=args.outcome, sample=args.sample, seed=seed)
        if seed == 0:
            print(json.dumps(describe(d), indent=2))

        for name in names:
            rep = run_one(name, d, seed) if name != "random" else run_random(d, seed)
            rep["data"] = describe(d)
            out = RESULTS / f"{name}_{args.outcome}_seed{seed}.json"
            out.write_text(json.dumps(rep, indent=2))
            print(f"{name:7s} seed{seed}  AUUC {rep['auuc']:10.2f}   "
                  f"top-decile {rep['top_decile_observed_uplift']:+.5f}   "
                  f"bottom {rep['bottom_decile_observed_uplift']:+.5f}")


if __name__ == "__main__":
    main()
