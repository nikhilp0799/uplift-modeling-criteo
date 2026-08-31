"""Meta-learners: S, T and X.

All three estimate the Conditional Average Treatment Effect
    tau(x) = E[Y | X=x, T=1] - E[Y | X=x, T=0]
from data where only one of those two terms is ever observed per user.
They differ in how they route that impossibility around the modelling.
"""

import numpy as np
from sklearn.base import clone
from xgboost import XGBClassifier, XGBRegressor


def _clf(**kw):
    return XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        tree_method="hist", eval_metric="logloss", **kw
    )


def _reg(**kw):
    return XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        tree_method="hist", **kw
    )


class SLearner:
    """One model, treatment as a feature.

    Cheap and often weak: gradient boosting can simply ignore the treatment
    column if the outcome signal is much stronger than the treatment effect,
    which on this data it is. Included because that failure mode is the point.
    """

    def fit(self, X, t, y):
        self.m = _clf()
        self.m.fit(np.hstack([X, t.reshape(-1, 1)]), y)
        return self

    def predict(self, X):
        ones = np.ones((len(X), 1), dtype=X.dtype)
        zeros = np.zeros((len(X), 1), dtype=X.dtype)
        p1 = self.m.predict_proba(np.hstack([X, ones]))[:, 1]
        p0 = self.m.predict_proba(np.hstack([X, zeros]))[:, 1]
        return p1 - p0


class TLearner:
    """Two independent models, one per arm; uplift is their difference.

    Clean and unbiased, but the control model here trains on ~15% of the
    data, so its variance dominates the estimate.
    """

    def fit(self, X, t, y):
        self.m1 = _clf().fit(X[t == 1], y[t == 1])
        self.m0 = _clf().fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return (self.m1.predict_proba(X)[:, 1]
                - self.m0.predict_proba(X)[:, 1])


class XLearner:
    """Imputes the missing counterfactual, then models the imputed effect.

    For treated users, estimate what would have happened untreated using the
    control model, and take the difference as an imputed effect; mirror it
    for control users. Fit a regressor to each imputed effect and blend by
    propensity. Designed for exactly this situation — unbalanced arms.
    """

    def fit(self, X, t, y):
        X1, y1 = X[t == 1], y[t == 1]
        X0, y0 = X[t == 0], y[t == 0]

        self.m1 = _clf().fit(X1, y1)
        self.m0 = _clf().fit(X0, y0)

        # Imputed treatment effects, per arm.
        d1 = y1 - self.m0.predict_proba(X1)[:, 1]
        d0 = self.m1.predict_proba(X0)[:, 1] - y0

        self.tau1 = _reg().fit(X1, d1)
        self.tau0 = _reg().fit(X0, d0)

        # Propensity for blending. Randomised here, so this is near-constant,
        # but keep it general.
        self.e = _clf().fit(X, t)
        return self

    def predict(self, X):
        g = self.e.predict_proba(X)[:, 1]
        return g * self.tau0.predict(X) + (1 - g) * self.tau1.predict(X)


LEARNERS = {"s": SLearner, "t": TLearner, "x": XLearner}
