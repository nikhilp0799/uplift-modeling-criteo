"""Uplift evaluation: Qini, AUUC, and decile lift.

The individual treatment effect is never observable — a user is either
treated or not — so there is no per-row ground truth to score against.
Everything here works by ranking users by predicted uplift and measuring
the *incremental* outcome in the top-k against what random targeting would
have produced.
"""

import numpy as np


def qini_curve(y: np.ndarray, t: np.ndarray, uplift: np.ndarray,
               points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental outcomes as a function of population targeted.

    At each cut-off k, take the top-k users by predicted uplift and compute
        responders_treated - responders_control * (n_treated / n_control)
    The scaling term is what makes this a Qini curve rather than a naive
    difference: without it, the arm with more users wins automatically.
    """
    order = np.argsort(-uplift)
    y, t = y[order], t[order]

    n = len(y)
    ks = np.unique(np.linspace(1, n, points).astype(int))

    y_t = np.cumsum(y * (t == 1))
    y_c = np.cumsum(y * (t == 0))
    n_t = np.cumsum(t == 1)
    n_c = np.cumsum(t == 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_c > 0, n_t / np.maximum(n_c, 1), 0.0)
    gain = y_t - y_c * ratio

    return ks / n, gain[ks - 1]


def auuc(y: np.ndarray, t: np.ndarray, uplift: np.ndarray) -> float:
    """Area under the uplift curve, normalised against random targeting.

    Random targeting is the straight line from (0,0) to the total
    incremental outcome. Reporting the area *above* that line is what makes
    the number interpretable: > 0 means the model found real heterogeneity,
    ~0 means it is no better than treating everyone in a random order.
    """
    x, gain = qini_curve(y, t, uplift)
    random_line = gain[-1] * x
    return float(np.trapezoid(gain - random_line, x))


def decile_lift(y: np.ndarray, t: np.ndarray, uplift: np.ndarray,
                bins: int = 10) -> list[dict]:
    """Observed uplift within each predicted-uplift decile.

    This is the honesty check on the model. If the top decile does not show
    a larger treated-minus-control gap than the bottom decile, the ranking
    is not capturing treatment effect regardless of what AUUC says.

    Watch for negative uplift in the bottom deciles — those are 'sleeping
    dogs', users the treatment actively drives away. Finding them is worth
    as much as finding persuadables.
    """
    order = np.argsort(-uplift)
    y, t, u = y[order], t[order], uplift[order]
    edges = np.array_split(np.arange(len(y)), bins)

    out = []
    for i, idx in enumerate(edges):
        yt, yc = y[idx][t[idx] == 1], y[idx][t[idx] == 0]
        obs = (yt.mean() if len(yt) else 0.0) - (yc.mean() if len(yc) else 0.0)
        out.append({
            "decile": i + 1,
            "n": int(len(idx)),
            "n_treated": int(len(yt)),
            "n_control": int(len(yc)),
            "predicted_uplift": float(u[idx].mean()),
            "observed_uplift": float(obs),
        })
    return out


def evaluate(y: np.ndarray, t: np.ndarray, uplift: np.ndarray) -> dict:
    x, gain = qini_curve(y, t, uplift)
    deciles = decile_lift(y, t, uplift)
    return {
        "auuc": auuc(y, t, uplift),
        "total_incremental": float(gain[-1]),
        "top_decile_observed_uplift": deciles[0]["observed_uplift"],
        "bottom_decile_observed_uplift": deciles[-1]["observed_uplift"],
        "qini_curve": {"fraction_targeted": x.tolist(),
                       "cumulative_gain": gain.tolist()},
        "deciles": deciles,
    }
