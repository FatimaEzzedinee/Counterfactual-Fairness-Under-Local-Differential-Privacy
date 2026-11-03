from __future__ import annotations
import os, sys, warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import (
    RANDOM_STATE, MODEL_FAMILIES, LR_PARAMS, RF_PARAMS, XGB_PARAMS,
    FEATURE_COLS, TARGET_COL,
)

try:
    from xgboost import XGBClassifier as _XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn("xgboost not installed; using extra RandomForest.", ImportWarning)


@dataclass
class ModelResult:
    name: str; family: str; scenario: str; estimator: Any
    y_pred:  np.ndarray = field(default_factory=lambda: np.array([]))
    y_proba: np.ndarray = field(default_factory=lambda: np.array([]))
    metrics: Dict[str,float] = field(default_factory=dict)
    n_train: int = 0
    feature_cols: list = field(default_factory=list)

@dataclass
class ScenarioModels:
    scenario: str
    results:  Dict[str,ModelResult] = field(default_factory=dict)


def _build_estimator(family):
    if family == "logistic_regression": return LogisticRegression(**LR_PARAMS)
    if family == "random_forest":       return RandomForestClassifier(**RF_PARAMS)
    if family == "xgboost":
        if _XGB_AVAILABLE: return _XGBClassifier(**XGB_PARAMS)
        return RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300, max_depth=10)
    raise ValueError(f"Unknown model family {family!r}")


def _evaluate(estimator, X_test, y_test):
    y_pred  = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    return y_pred, y_proba, {
        "accuracy":  accuracy_score(y_test, y_pred),
        "auc_roc":   roc_auc_score(y_test, y_proba),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
    }


def train_scenario(scenario, X_train, y_train, X_test, y_test,
                   X_val=None, y_val=None, families=MODEL_FAMILIES,
                   sample_weight=None, verbose=True, feature_map=None):
    """Train one model per family and evaluate on the test set.

    Parameters
    ----------
    sample_weight : array-like of shape (n_train,) or None
        Per-sample weights passed to estimator.fit().  None = uniform weights
        (original behaviour).  Used to down-weight synthetic augmented rows
        when AUG_SYNTHETIC_WEIGHT is set in config.
    feature_map : callable or None
        If given, each estimator is wrapped so it is fit on and predicts from
        feature_map(X) (see kusner_baselines.FeatureMapEstimator); callers
        keep passing raw features everywhere.
    """
    sm = ScenarioModels(scenario=scenario)
    for family in families:
        if verbose:
            print(f"  [models] {scenario}/{family} on {len(X_train):,} rows ... ", end="")
        est = _build_estimator(family)
        if feature_map is not None:
            from pipeline.kusner_baselines import FeatureMapEstimator
            est = FeatureMapEstimator(est, feature_map)
        try:
            est.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            # Fallback: estimator does not support sample_weight (shouldn't
            # happen for LR / RF / XGB, but guards against edge cases).
            warnings.warn(f"[models] {family} does not support sample_weight — fitting without.")
            est.fit(X_train, y_train)
        y_pred, y_proba, metrics = _evaluate(est, X_test, y_test)
        sm.results[family] = ModelResult(
            name=f"{scenario}_{family}", family=family, scenario=scenario,
            estimator=est, y_pred=y_pred, y_proba=y_proba,
            metrics=metrics, n_train=len(X_train), feature_cols=FEATURE_COLS)
        if verbose:
            weight_note = f"  w_synth={sample_weight[0]:.2f}" if sample_weight is not None else ""
            print(f"AUC={metrics['auc_roc']:.3f}  Acc={metrics['accuracy']:.3f}{weight_note}")
    return sm


def train_all_scenarios(data, ldp_result, verbose=True):
    if verbose:
        print("" + "="*60)
        print("  MODEL TRAINING  (COMPAS)")
        print("="*60)
    scenarios = {}
    if verbose: print("[Scenario A] Baseline")
    scenarios["baseline"] = train_scenario(
        "baseline", data.X_train, data.y_train, data.X_test, data.y_test, verbose=verbose)
    if verbose: print("[Scenario B] Augmented (+ counterfactuals)")
    scenarios["augmented"] = train_scenario(
        "augmented", data.X_aug, data.y_aug, data.X_test, data.y_test, verbose=verbose)
    if verbose: print("[Scenario C] LDP-Augmented")
    scenarios["ldp"] = train_scenario(
        "ldp", ldp_result.X_aug_ldp, ldp_result.y_aug_ldp,
        data.X_test, data.y_test, verbose=verbose)
    if verbose:
        print("[models] Training complete.")
        hdr = f"{'Scenario':<12} {'Family':<22} {'Acc':>6} {'AUC':>6} {'F1':>6}"
        print(hdr); print("-"*len(hdr))
        for sc_name, sm in scenarios.items():
            for fam, res in sm.results.items():
                m = res.metrics
                print(f"{sc_name:<12} {fam:<22} {m['accuracy']:.3f} {m['auc_roc']:.3f} {m['f1']:.3f}")
    return scenarios
