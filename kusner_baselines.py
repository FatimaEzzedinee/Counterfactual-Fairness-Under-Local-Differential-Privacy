"""
kusner_baselines.py -- predictor baselines from Kusner et al. (2017),
"Counterfactual Fairness" (NeurIPS), Section 5 (law-school experiment).

Unlike the augmentation methods (cf_generation.py), these change the model's
INPUTS rather than its training rows:

  unaware   -- the protected attribute is removed (set to a constant), every
               other feature kept as is. Not counterfactually fair whenever A
               has descendants among the features.
  fair_add  -- "Fair Add" (Kusner's Level 3): each SCM descendant X_j is
               replaced by its residual X_j - (b0_j + b1_j * A) under the
               additive SCM, and A is removed. The residuals are functions of
               the exogenous noise U only, so by Kusner's Lemma 1 the predictor
               is counterfactually fair w.r.t. that SCM by construction.

Kusner's Fair K (Level 2: a latent "knowledge" variable inferred by MCMC) is
not implemented: it needs a hand-built latent-variable model per dataset, and
under this pipeline's additive linear SCM its role is played by the residuals.

The mapping is applied inside predict / predict_proba (FeatureMapEstimator),
so every evaluation that calls the model on RAW features -- test metrics, the
flip and SCM counterfactual metrics in paper_export.py -- works unchanged.
The SCM is fit on the training data actually passed in (LDP-perturbed in the
privacy scenarios); with epsilon given, its coefficients are LDP-corrected
(dp_causal_analysis.fit_scm_corrected_ldp).
"""

from __future__ import annotations
import os, sys
from typing import Callable, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import FEATURE_COLS, PROTECTED_COL
from pipeline.cf_generation import SCMParams, fit_scm

_PROTECTED_IDX = FEATURE_COLS.index(PROTECTED_COL)

KUSNER_BASELINES = ["unaware", "fair_add"]
# LDP-only: Fair Add with the LDP-corrected SCM coefficients.
KUSNER_LDP_CORRECTED = ["fair_add_ldp_corrected"]


class FeatureMapEstimator:
    """Wrap an estimator so it is fit on, and predicts from, feature_map(X)."""

    def __init__(self, base, feature_map: Callable[[np.ndarray], np.ndarray]):
        self.base = base
        self.feature_map = feature_map

    def fit(self, X, y, sample_weight=None):
        self.base.fit(self.feature_map(np.asarray(X, dtype=float)), y, sample_weight=sample_weight)
        self.classes_ = getattr(self.base, "classes_", None)
        return self

    def predict(self, X):
        return self.base.predict(self.feature_map(np.asarray(X, dtype=float)))

    def predict_proba(self, X):
        return self.base.predict_proba(self.feature_map(np.asarray(X, dtype=float)))


def unaware_map(X: np.ndarray) -> np.ndarray:
    X_m = np.array(X, dtype=float, copy=True)
    X_m[:, _PROTECTED_IDX] = 0.0
    return X_m


def fair_add_map(scm: SCMParams) -> Callable[[np.ndarray], np.ndarray]:
    """Descendants -> residuals under `scm` (each row's own A), then drop A."""
    idx = {col: FEATURE_COLS.index(col) for col in scm.dependent_cols}

    def _map(X: np.ndarray) -> np.ndarray:
        X_m = np.array(X, dtype=float, copy=True)
        a = X_m[:, _PROTECTED_IDX]
        for col, j in idx.items():
            b0, b1 = scm.coef[col]
            X_m[:, j] = X_m[:, j] - (b0 + b1 * a)
        X_m[:, _PROTECTED_IDX] = 0.0
        return X_m

    return _map


def build_feature_map(method: str, X_train: np.ndarray,
                      epsilon: Optional[float] = None) -> Callable[[np.ndarray], np.ndarray]:
    """Feature map for one Kusner baseline, fit on X_train.

    method in KUSNER_BASELINES, or 'fair_add_ldp_corrected' (requires epsilon).
    """
    if method == "unaware":
        return unaware_map
    if method == "fair_add":
        return fair_add_map(fit_scm(X_train))
    if method == "fair_add_ldp_corrected":
        if epsilon is None:
            raise ValueError("fair_add_ldp_corrected requires an LDP epsilon.")
        from pipeline.dp_causal_analysis import fit_scm_corrected_ldp
        return fair_add_map(fit_scm_corrected_ldp(X_train, epsilon))
    raise ValueError("unknown Kusner baseline {!r}; expected one of {}".format(
        method, KUSNER_BASELINES + KUSNER_LDP_CORRECTED))
