from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_BASE_DIR, os.path.join(_BASE_DIR, "ensemble_mia-main")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.config import (
    CATEGORICAL_COLS, FEATURE_COLS, PROTECTED_COL,
    GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL, RANDOM_STATE, OUTPUT_DIR,
)
from pipeline.data_preparation import augment_data, load_and_split
from pipeline.models import _build_estimator


@dataclass
class AttackResult:
    attacker_name: str
    model_name: str
    input_type: str
    metrics: Dict[str, float] = field(default_factory=dict)
    fpr_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr_curve: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class AIAResult:
    model_name: str
    input_type: str
    attacks: Dict[str, AttackResult] = field(default_factory=dict)
    score_df: Optional[pd.DataFrame] = None
    true_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    runtime_sec: float = 0.0
    variant: str = "no_y"
    feature_set: str = "x_ns+outputs"


def _group_labels():
    """Human-readable names for the two values of the binarised sensitive attr.

    _infer_sensitive_binary maps values ``< midpoint`` -> 0 (low_val group) and
    ``>= midpoint`` -> 1 (high_val group). config.PROTECTED_UNPRIV/PROTECTED_PRIV
    are 0/1 respectively (see pipeline/datasets/base.py), so those map onto
    GROUP_UNPRIV_LABEL/GROUP_PRIV_LABEL -- the same dataset-driven labels
    fairness_metrics.py and cf_generation.py already use for whichever column
    config.PROTECTED_COL actually is. (This used to read a separate
    config.PROTECTED flag ('SEX' | 'RACE') that could silently disagree with
    PROTECTED_COL -- see pipeline/cf_generation.py::_protected_params for the
    same class of bug.)
    """
    return (GROUP_UNPRIV_LABEL, GROUP_PRIV_LABEL)


def _per_group_metrics(y_true, y_pred, scores, group_labels):
    """Return AIA metrics broken out per true-sensitive-group.

    For each group g in {0, 1} we treat g as the "positive" class and
    report the full confusion-matrix-derived attacker performance on that
    group. Because y_true == the sensitive attribute here, per-group metrics
    describe exactly how well the attacker re-identifies members of each group:

        n_<g>             = # rows with y_true == g
        accuracy_<g>      = P(y_pred == g | y_true == g)
                            (aka recall_<g>; group 0 -> TNR, group 1 -> TPR)
        error_rate_<g>    = 1 - accuracy_<g>
        precision_<g>     = P(y_true == g | y_pred == g)
                            (of everyone the attacker LABELLED g, how many were g?)
        recall_<g>        = P(y_pred == g | y_true == g)   (== accuracy_<g>)
        f1_<g>            = 2 * precision * recall / (precision + recall)
        pos_pred_rate_<g> = P(y_pred == 1 | y_true == g)
        mean_score_<g>    = mean attacker score for rows in that group

    We also emit a summary ``group_disparity`` = |accuracy_1 - accuracy_0|
    which quantifies how much more information the attacker leaks about one
    group than the other. Zero means the attacker is equally (in)accurate
    on both groups; large values mean one group is far easier to re-identify.
    Companion disparities ``precision_disparity`` and ``f1_disparity`` mirror
    the same idea for precision and F1.
    """
    out = {}
    g0_lbl, g1_lbl = group_labels

    # Full 2x2 confusion matrix so we can derive precision / recall / F1 for
    # each group by treating that group alternately as the "positive" class.
    n_total = int(len(y_true))
    for g, glbl in enumerate(group_labels):
        mask_true_g = (y_true == g)
        mask_pred_g = (y_pred == g)
        n_g = int(mask_true_g.sum())
        out[f"n_{glbl}"] = n_g

        tp = int((mask_true_g & mask_pred_g).sum())
        fp = int((~mask_true_g & mask_pred_g).sum())
        fn = int((mask_true_g & ~mask_pred_g).sum())

        if n_g == 0:
            for k in ("accuracy", "error_rate", "pos_pred_rate", "mean_score",
                      "precision", "recall", "f1"):
                out[f"{k}_{glbl}"] = float("nan")
            continue

        pred_of_group = y_pred[mask_true_g]
        score_of_group = scores[mask_true_g]

        acc_g   = float((pred_of_group == g).mean())          # = recall_g
        pp_g    = float((pred_of_group == 1).mean())
        prec_g  = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        rec_g   = acc_g                                       # by definition
        if np.isfinite(prec_g) and (prec_g + rec_g) > 0:
            f1_g = float(2.0 * prec_g * rec_g / (prec_g + rec_g))
        else:
            f1_g = 0.0 if (tp == 0 and fp == 0 and fn == 0) else float("nan")

        out[f"accuracy_{glbl}"]      = acc_g
        out[f"error_rate_{glbl}"]    = 1.0 - acc_g
        out[f"pos_pred_rate_{glbl}"] = pp_g
        out[f"mean_score_{glbl}"]    = float(np.nanmean(score_of_group))
        out[f"precision_{glbl}"]     = prec_g
        out[f"recall_{glbl}"]        = rec_g
        out[f"f1_{glbl}"]            = f1_g

    def _safe_abs(a, b):
        if np.isfinite(a) and np.isfinite(b):
            return float(abs(a - b))
        return float("nan")

    acc0 = out.get(f"accuracy_{g0_lbl}", float("nan"))
    acc1 = out.get(f"accuracy_{g1_lbl}", float("nan"))
    out["group_disparity"] = _safe_abs(acc1, acc0)
    out["precision_disparity"] = _safe_abs(
        out.get(f"precision_{g1_lbl}", float("nan")),
        out.get(f"precision_{g0_lbl}", float("nan")),
    )
    out["f1_disparity"] = _safe_abs(
        out.get(f"f1_{g1_lbl}", float("nan")),
        out.get(f"f1_{g0_lbl}", float("nan")),
    )
    # Which group is the attacker better at re-identifying?
    if np.isfinite(acc0) and np.isfinite(acc1):
        out["group_more_leaked"] = g1_lbl if acc1 > acc0 else g0_lbl
    else:
        out["group_more_leaked"] = ""
    return out


def _compute_metrics(true_labels, scores, group_labels=None):
    """Score one attacker against the true sensitive-attribute labels.

    Threshold policy: previously this used the *median* of the score
    distribution, which (a) forces the predicted-positive rate to ~50% by
    construction regardless of the attacker's actual behavior, and (b) has no
    principled connection to "advantage". We instead use the Youden's-J
    operating point -- threshold_t that maximizes TPR(t) - FPR(t) -- which is
    exactly the classic Yeom et al. (2018) definition of attacker "advantage":
    Adv = sup_t (TPR(t) - FPR(t)). accuracy/precision/recall/F1/balanced-accuracy
    are then reported at that *same* threshold, so "advantage" and "accuracy"
    describe one coherent attacker strategy instead of two unrelated ones.

    Because attribute inference "accuracy" is meaningless without knowing the
    class balance of the sensitive attribute in the eval set (e.g. guessing
    the majority class is always >=50% "accurate"), we also report the
    majority-class baseline and each attacker's accuracy over that baseline.
    """
    true_labels = np.asarray(true_labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(true_labels):
        return {}, np.array([]), np.array([])
    if np.isnan(scores).all() or np.unique(true_labels).size < 2:
        return {}, np.array([]), np.array([])

    n_eval = int(len(true_labels))
    positive_base_rate = float(np.mean(true_labels))
    majority_baseline_accuracy = float(max(positive_base_rate, 1.0 - positive_base_rate))

    fpr_curve, tpr_curve, thresh_curve = roc_curve(true_labels, scores)
    j_stat = tpr_curve - fpr_curve
    best_idx = int(np.argmax(j_stat))
    threshold = float(thresh_curve[best_idx])
    advantage = float(j_stat[best_idx])

    y_pred = (scores >= threshold).astype(int)
    tp = int(((true_labels == 1) & (y_pred == 1)).sum())
    fp = int(((true_labels == 0) & (y_pred == 1)).sum())
    tn = int(((true_labels == 0) & (y_pred == 0)).sum())
    fn = int(((true_labels == 1) & (y_pred == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = 1.0 - fpr
    balanced_accuracy = float((tpr + tnr) / 2.0)
    tpr_at = {f"tpr_at_fpr_{t}": float(np.interp(t, fpr_curve, tpr_curve)) for t in [0.0, 0.001, 0.01, 0.1]}

    accuracy = float(accuracy_score(true_labels, y_pred))
    # Brier score: mean squared error between (probability-like) score and true
    # label -- a calibration signal, not just discrimination. For the one hard
    # 0/1 attacker (target_pred) this degenerates to its plain error rate.
    brier_score = float(np.mean((scores - true_labels) ** 2))
    _privacy_gain = float(1.0 - advantage)
    metrics = {
        "n_eval": n_eval,
        "positive_base_rate": positive_base_rate,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "auc_roc": float(roc_auc_score(true_labels, scores)),
        "pr_auc": float(average_precision_score(true_labels, scores)),
        "threshold_used": threshold,
        "accuracy": accuracy,
        "accuracy_over_baseline": float(accuracy - majority_baseline_accuracy),
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision_score(true_labels, y_pred, zero_division=0)),
        "recall": float(recall_score(true_labels, y_pred, zero_division=0)),
        "f1": float(f1_score(true_labels, y_pred, zero_division=0)),
        "true_positive_rate": float(tpr),
        "false_positive_rate": float(fpr),
        "brier_score": brier_score,
        "aia_advantage": advantage,
        "mia_advantage": advantage,           # alias for visualization compatibility
        "attribute_privacy_gain": _privacy_gain,
        "privacy_gain": _privacy_gain,         # alias for visualization compatibility
        **tpr_at,
    }

    # Per-group breakdown (e.g. female vs male for SEX). Uses the SAME
    # Youden-threshold y_pred as the aggregate metrics above so the
    # per-group and aggregate numbers describe one coherent attacker.
    glbls = group_labels or _group_labels()
    metrics["group0_label"] = glbls[0]
    metrics["group1_label"] = glbls[1]
    metrics.update(_per_group_metrics(true_labels, y_pred, scores, glbls))
    return metrics, fpr_curve, tpr_curve


def _infer_sensitive_binary(values, low_val: float, high_val: float):
    threshold = (float(low_val) + float(high_val)) / 2.0
    return (np.asarray(values, dtype=float) >= threshold).astype(int)


def _drop_sensitive_column(X, sensitive_idx: int):
    return np.delete(np.asarray(X, dtype=float), sensitive_idx, axis=1)


def _build_attack_features(estimator, X, sensitive_idx: int, include_x_ns: bool = True):
    """Build the attacker's feature matrix.

    include_x_ns=True (default)
        Classical attribute-inference threat model (Fredrikson et al., Yeom et
        al.): the attacker knows every non-sensitive attribute of the record AND
        observes the target model's output. Features are [X_ns | proba, pred,
        confidence].

    include_x_ns=False
        Black-box-output-only attacker: features are [proba, pred, confidence]
        alone. Note these three are all deterministic functions of proba, so this
        variant gives the attack a single degree of freedom.

    Either way X_ns is what the *target* model is queried on; include_x_ns only
    controls whether it is also handed to the attack model.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.empty((0, 0)), np.array([]), np.array([]), np.array([])

    X_ns = _drop_sensitive_column(X, sensitive_idx)
    proba = estimator.predict_proba(X_ns)[:, 1]
    pred = estimator.predict(X_ns).astype(float)
    confidence = np.maximum(proba, 1.0 - proba)
    outputs = np.column_stack([proba, pred, confidence])
    features = np.column_stack([X_ns, outputs]) if include_x_ns else outputs
    return features, proba, pred, confidence


def _build_ensemble_score(score_map: Dict[str, np.ndarray], eval_size: int):
    """Combine attack scores into one ensemble score.

    Prefer actual attacker outputs when they exist. Fallback to probability-type
    baseline scores only when no attacker model could be trained. target_pred is
    always excluded because it is a hard binary label (0/1), not a probability,
    and averaging it with probability scores corrupts the ensemble signal.
    """
    attack_keys = [
        key for key in ("attacker_lr", "attacker_rf")
        if key in score_map and len(score_map[key]) == eval_size
    ]
    if attack_keys:
        return np.mean(np.column_stack([score_map[key] for key in attack_keys]), axis=1)

    # Fallback: use only probability-type scores (exclude hard binary target_pred)
    prob_keys = ("target_prob_pos", "target_confidence")
    valid_scores = [
        score_map[k] for k in prob_keys
        if k in score_map and len(score_map[k]) == eval_size
    ]
    if valid_scores:
        return np.mean(np.column_stack(valid_scores), axis=1)
    return None


def _train_target_estimator(family: str, X_train, y_train, sensitive_idx: int):
    estimator = _build_estimator(family)
    estimator.fit(_drop_sensitive_column(X_train, sensitive_idx), y_train)
    return estimator


def _generate_target_cf(target_estimator, model_name: str, data, sensitive_idx: int,
                        query_source_X: np.ndarray,
                        verbose: bool = True):
    """Generate NiCE counterfactuals for *this* AIA target model.

    `pipeline.nice_cf.generate_nice_cfs_for_all_models` explains the
    fairness-ablation models, which are trained WITH the protected attribute
    as a feature. The AIA target model is a different, purpose-built model
    trained WITHOUT it (see `_train_target_estimator`), so those CFs are not
    valid queries for it -- the search space and decision boundary differ.
    This instead runs NiCE directly against `target_estimator` in the reduced
    (non-sensitive) feature space it actually operates on.

    `query_source_X` is the CF query pool: the AIA attacker's *train* set
    (`attack_X_train`, i.e. `data.X_val` -- the dataset's genuine held-out
    validation split, never seen by any target model), in full. It never
    includes rows from `attack_X_eval` (`data.X_test`, the AIA test set),
    which stays reserved for scoring the attack -- see
    `run_attribute_inference_analysis`. Every row in it is used as a query, so
    the number of CFs generated always equals the size of the AIA validation
    set. The resulting CFs become the "nice_cf" variant's attacker-training
    set (not its eval set) -- see `_resolve_nice_cf_train`.

    Each query's *original* protected-attribute value is reattached to its
    counterfactual afterwards: NiCE never sees or perturbs that column (it is
    dropped before the search), so the ground-truth sensitive label used to
    score the attacker is exactly the source row's real value.

    Returns an (n, len(FEATURE_COLS)) array of counterfactual rows in the
    same representation as `data.X_test`, or None if generation fails or
    NICEx is unavailable.
    """
    import pipeline.nice_cf as ncf

    if not ncf._NICE_AVAILABLE:
        warnings.warn(f"[AIA] NICEx not installed; skipping NiCE CF generation for {model_name}.")
        return None

    X_train_full = np.asarray(data.X_train, dtype=float)
    y_train = np.asarray(data.y_train, dtype=int)
    X_train_ns = _drop_sensitive_column(X_train_full, sensitive_idx)

    reduced_cols = [c for j, c in enumerate(FEATURE_COLS) if j != sensitive_idx]
    cat_feat = [i for i, c in enumerate(reduced_cols) if c in CATEGORICAL_COLS]
    num_feat = [i for i in range(len(reduced_cols)) if i not in set(cat_feat)]

    X_query_full = np.asarray(query_source_X, dtype=float)
    X_query_ns = _drop_sensitive_column(X_query_full, sensitive_idx)

    def predict_fn(X):
        return target_estimator.predict_proba(np.asarray(X, dtype=float))

    try:
        nm = ncf._NICELibrary(
            predict_fn=predict_fn, X_train=X_train_ns.copy(),
            cat_feat=cat_feat, num_feat=num_feat,
            y_train=y_train.copy(), optimization="proximity",
            justified_cf=True, distance_metric="HEOM",
            num_normalization="minmax",
        )
    except Exception as exc:
        warnings.warn(f"[AIA] NiCE init failed for {model_name}: {exc}")
        return None

    if verbose:
        print(f"  [AIA] Generating NiCE CFs for {model_name} ({len(X_query_ns)} queries)... ", end="", flush=True)
        
    cfs_ns, valid_idx = [], []
    for i, x in enumerate(X_query_ns):
        try:
            cf = nm.explain(x.reshape(1, -1).astype(float))
            if cf is not None and cf.shape == (1, X_query_ns.shape[1]):
                cfs_ns.append(cf[0])
                valid_idx.append(i)
        except Exception:
            pass
    if verbose:
        print(f"{len(cfs_ns)}/{len(X_query_ns)} valid.")

    if not cfs_ns:
        warnings.warn(f"[AIA] No NiCE CFs generated for {model_name}.")
        return None

    X_cf_ns = np.vstack(cfs_ns)
    orig_sensitive = X_query_full[valid_idx, sensitive_idx]
    return np.insert(X_cf_ns, sensitive_idx, orig_sensitive, axis=1)


def _resolve_nice_cf_train(nice_cf_results: Optional[dict], sc_name: str, family: str,
                           target_estimator, model_name: str, data, sensitive_idx: int,
                           generate_missing_cf: bool, attack_X_train: np.ndarray, verbose: bool):
    """Return an (n, len(FEATURE_COLS)) counterfactual training array, or None.

    This is the "nice_cf" AIA variant's attacker-training set: counterfactuals
    of the AIA validation set (`attack_X_train` = `data.X_val`), used in place of the raw
    validation rows so the attacker learns from CF-perturbed features. It is
    never used for scoring -- both AIA variants are evaluated on the same raw
    `attack_X_eval` (the AIA test set), so the comparison isolates the effect
    of CF-based training.

    Prefers a precomputed NiCEResult supplied via `nice_cf_results` (e.g. from
    `pipeline.nice_cf.generate_nice_cfs_for_all_models`, cached from a prior
    pipeline stage). Falls back to generating fresh CFs against this AIA
    target model when none is available and `generate_missing_cf` is True,
    querying every row of `attack_X_train`.
    """
    if nice_cf_results:
        nice_res = nice_cf_results.get(sc_name, {}).get(family)
        X_cf = getattr(nice_res, "X_cf", None)
        if X_cf is not None and len(X_cf) > 0:
            return np.asarray(X_cf, dtype=float)

    if not generate_missing_cf:
        return None

    return _generate_target_cf(
        target_estimator, model_name, data, sensitive_idx,
        query_source_X=attack_X_train, verbose=verbose,
    )


def run_attribute_inference_single(
    model_name,
    input_type,
    estimator,
    X_train,
    X_eval,
    sensitive_idx: int,
    low_val: float,
    high_val: float,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
    variant: str = "no_y",
    include_x_ns: bool = True,
):
    t0 = time.perf_counter()
    X_train = np.asarray(X_train, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    result = AIAResult(
        model_name=model_name,
        input_type=input_type,
        variant=variant,
        feature_set="x_ns+outputs" if include_x_ns else "outputs_only",
    )

    if X_train.size == 0 or X_eval.size == 0:
        result.runtime_sec = time.perf_counter() - t0
        return result

    y_train_sensitive = _infer_sensitive_binary(X_train[:, sensitive_idx], low_val, high_val)
    y_eval_sensitive = _infer_sensitive_binary(X_eval[:, sensitive_idx], low_val, high_val)
    result.true_labels = y_eval_sensitive

    F_train, prob_train, pred_train, conf_train = _build_attack_features(
        estimator, X_train, sensitive_idx, include_x_ns=include_x_ns
    )
    F_eval, prob_eval, pred_eval, conf_eval = _build_attack_features(
        estimator, X_eval, sensitive_idx, include_x_ns=include_x_ns
    )

    if len(y_eval_sensitive) == 0 or np.unique(y_eval_sensitive).size < 2:
        result.runtime_sec = time.perf_counter() - t0
        return result

    score_map: Dict[str, np.ndarray] = {
        "target_prob_pos": prob_eval,
        "target_pred": pred_eval,
        "target_confidence": conf_eval,
    }

    if np.unique(y_train_sensitive).size >= 2 and F_train.shape[0] > 1 and F_eval.shape[0] > 0:
        try:
            atk_lr = LogisticRegression(max_iter=400, random_state=random_state)
            atk_lr.fit(F_train, y_train_sensitive)
            score_map["attacker_lr"] = atk_lr.predict_proba(F_eval)[:, 1]
        except Exception as exc:
            warnings.warn(f"[AIA] attacker_lr failed: {exc}")

        try:
            atk_rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=random_state,
            )
            atk_rf.fit(F_train, y_train_sensitive)
            score_map["attacker_rf"] = atk_rf.predict_proba(F_eval)[:, 1]
        except Exception as exc:
            warnings.warn(f"[AIA] attacker_rf failed: {exc}")

    ensemble_score = _build_ensemble_score(score_map, len(y_eval_sensitive))
    if ensemble_score is not None:
        score_map["ensemble_mean"] = ensemble_score

    for att_name, score in score_map.items():
        try:
            metrics, fpr_curve, tpr_curve = _compute_metrics(y_eval_sensitive, score)
        except Exception as exc:
            warnings.warn(f"[AIA] metrics for {att_name} failed: {exc}")
            continue
        if not metrics:
            continue
        result.attacks[att_name] = AttackResult(
            attacker_name=att_name,
            model_name=model_name,
            input_type=input_type,
            metrics=metrics,
            fpr_curve=fpr_curve,
            tpr_curve=tpr_curve,
        )

    result.runtime_sec = time.perf_counter() - t0
    if verbose and result.attacks:
        best = max(result.attacks.values(), key=lambda r: r.metrics.get("auc_roc", 0.0))
        print(
            f"  [AIA] Best: {best.attacker_name}  AUC={best.metrics.get('auc_roc', float('nan')):.3f}  "
            f"Adv={best.metrics.get('aia_advantage', float('nan')):+.3f}  ({result.runtime_sec:.1f}s)"
        )
    return result


def _resolve_training_pair(training_sets: dict, scenario_name: str):
    if scenario_name not in training_sets:
        raise KeyError(f"Missing training set for scenario {scenario_name!r}")
    pair = training_sets[scenario_name]
    if isinstance(pair, dict):
        return pair["X_train"], pair["y_train"]
    return pair


def _unpack_pair(pair):
    if isinstance(pair, dict):
        if "X_train" in pair and "y_train" in pair:
            return pair["X_train"], pair["y_train"]
        if "X_eval" in pair and "y_eval" in pair:
            return pair["X_eval"], pair["y_eval"]
        raise KeyError("Expected a pair with X_train/y_train or X_eval/y_eval")
    return pair


def run_attribute_inference_analysis(
    training_sets: dict,
    families,
    data,
    nice_cf_results: Optional[dict] = None,
    sensitive_col: str = PROTECTED_COL,
    verbose: bool = True,
    include_x_ns: bool = True,
    include_nice_cf: bool = False,
    generate_missing_cf: bool = True,
):
    """Attribute inference on target models trained without the sensitive feature.

    Parameters
    ----------
    training_sets:
        Mapping scenario name -> (X_train, y_train) used to retrain the target
        model without the protected column.
    families:
        Model families to train for the target model and the attack model.
    data:
        DataBundle or compatible object providing X_val, X_test, and the
        protected feature layout.
    nice_cf_results:
        Optional precomputed NiCE counterfactual results, keyed the same way
        as `training_sets` x `families` (e.g. from
        `pipeline.nice_cf.generate_nice_cfs_for_all_models`). Only consulted
        when `include_nice_cf=True`.
    include_nice_cf:
        False -- (default) run only the attacker trained on the raw
                 validation set ("held_out_test_eval").
        True  -- also train a second attacker per (scenario, family) on
                 counterfactuals of the validation set, as a "nice_cf"
                 input_type alongside "held_out_test_eval". Both attackers
                 are scored on the same raw `attack_X_eval` (the AIA test
                 set), so the comparison isolates the effect of CF-based
                 training. Uses `nice_cf_results` when available for that
                 scenario/family, otherwise falls back to
                 `generate_missing_cf`.
    generate_missing_cf:
        True  -- (default, only relevant when include_nice_cf=True) generate
                 fresh NiCE counterfactuals against this AIA target model
                 (which is trained without the sensitive column) for any
                 scenario/family missing from `nice_cf_results`.
        False -- skip scenario/families with no precomputed CFs.
                 On-the-fly generation always queries every row of the AIA
                 validation set (`attack_X_train` = `data.X_val`) -- never
                 `attack_X_eval`, which is reserved for scoring the attack --
                 so the CF count equals the size of `data.X_val`.
    include_x_ns:
        True  -- (default) the attacker sees the non-sensitive attributes of the
                 record in addition to the target model's outputs. This is the
                 classical attribute-inference threat model.
        False -- the attacker sees only the target model's outputs.
        A bool runs a single feature-set variant. Pass an iterable, e.g.
        (True, False), to run every listed variant as an ablation -- the
        target model and any NiCE CF pool are trained/generated once per
        (scenario, family) and reused across variants, only the attacker's
        feature construction and fit differ per variant.
    """
    x_ns_variants = [include_x_ns] if isinstance(include_x_ns, bool) else list(dict.fromkeys(include_x_ns))
    multi_x_ns = len(x_ns_variants) > 1
    if verbose:
        print("\n" + "=" * 60)
        print("  ATTRIBUTE INFERENCE ATTACK  (target model excludes sensitive feature)")
        print("  attacker trained on data.X_val, scored on data.X_test")
        print("  attack feature-set variants: {}".format(
            [("x_ns+outputs" if v else "outputs_only") for v in x_ns_variants]))
        print("  nice_cf variant: {}".format("included" if include_nice_cf else "excluded"))
        print("=" * 60)

    if sensitive_col not in FEATURE_COLS:
        warnings.warn(f"[AIA] Unknown sensitive_col={sensitive_col!r}; using {PROTECTED_COL!r}.")
        sensitive_col = PROTECTED_COL
    sensitive_idx = FEATURE_COLS.index(sensitive_col)

    X_train_ref = np.asarray(next(iter(training_sets.values()))[0], dtype=float)
    col_train = X_train_ref[:, sensitive_idx]
    low_val = float(np.nanmin(col_train))
    high_val = float(np.nanmax(col_train))
    if not np.isfinite(low_val) or not np.isfinite(high_val) or low_val == high_val:
        low_val, high_val = 0.0, 1.0

    # The attacker must be trained and scored on rows NO target model ever
    # trained on. `training_sets` for every scenario is built from
    # `data.X_train` (or a row-superset thereof, for the causal-CF-augmented
    # scenarios -- see `data_preparation.augment_data`), so `data.X_train`
    # itself is never safe to use here. Instead:
    #   attack_X_train = data.X_val  -- the dataset's own held-out
    #       validation split (config.VAL_SIZE), carved out disjoint from BOTH
    #       X_train and X_test at load time (see data_preparation.load_and_split).
    #       No target model in the whole ablation is ever fit on it, so it is
    #       safe to use as the attacker's training data.
    #   attack_X_eval  = data.X_test -- the full real test set, i.e. the
    #       SAME evaluation surface every other scenario in the pipeline
    #       reports accuracy/fairness on, so AIA numbers are directly
    #       comparable to everything else in ablation_summary.csv.
    attack_X_train = np.asarray(data.X_val, dtype=float)
    attack_X_eval = np.asarray(data.X_test, dtype=float)
    if attack_X_train.shape[0] == 0:
        raise ValueError(
            "AIA requires a non-empty validation split (data.X_val is empty) "
            "-- set pipeline.config.VAL_SIZE > 0 before load_and_split()."
        )
    if attack_X_train.shape[0] < 100 and verbose:
        warnings.warn(
            f"[AIA] Only {attack_X_train.shape[0]} validation rows available to "
            "train the attacker; AUC estimates may be noisy."
        )

    all_results = {}
    for sc_name in training_sets.keys():
        X_train_scn, y_train_scn = _resolve_training_pair(training_sets, sc_name)
        X_train_scn = np.asarray(X_train_scn, dtype=float)
        y_train_scn = np.asarray(y_train_scn, dtype=int)
        all_results[sc_name] = {}

        for family in families:
            all_results[sc_name][family] = {}
            if verbose:
                print(f"\n[AIA] Scenario={sc_name}  Family={family}  attr={sensitive_col}")

            # Target model and (if requested) its NiCE CF pool depend only on
            # (scenario, family) -- not on the x_ns feature-set variant -- so
            # both are computed once here and reused across every variant
            # below, rather than being retrained/regenerated per variant.
            target_estimator = _train_target_estimator(family, X_train_scn, y_train_scn, sensitive_idx)
            model_name = f"{sc_name}_{family}_nosens"

            X_train_nice = None
            if include_nice_cf:
                X_train_nice = _resolve_nice_cf_train(
                    nice_cf_results, sc_name, family,
                    target_estimator=target_estimator, model_name=model_name,
                    data=data, sensitive_idx=sensitive_idx,
                    generate_missing_cf=generate_missing_cf,
                    attack_X_train=attack_X_train, verbose=verbose,
                )
                if X_train_nice is None or len(X_train_nice) == 0:
                    warnings.warn(f"[AIA] No NiCE CFs available for {sc_name}/{family} -- skipping.")

            for ix_ns in x_ns_variants:
                feature_set = "x_ns+outputs" if ix_ns else "outputs_only"
                suffix = f"__{feature_set}" if multi_x_ns else ""

                all_results[sc_name][family][f"test_rows{suffix}"] = run_attribute_inference_single(
                    model_name=model_name,
                    input_type="held_out_test_eval",
                    estimator=target_estimator,
                    X_train=attack_X_train,
                    X_eval=attack_X_eval,
                    sensitive_idx=sensitive_idx,
                    low_val=low_val,
                    high_val=high_val,
                    verbose=verbose,
                    include_x_ns=ix_ns,
                )

                if not include_nice_cf:
                    continue
                # Same eval set as above (attack_X_eval = data.X_test) --
                # only the attacker's training rows differ,
                # so "test_rows" and "nice_cf" are a controlled comparison:
                # attackers trained on raw vs. counterfactual versions of the
                # validation set, both scored on the same real, unseen test
                # rows.
                if X_train_nice is not None and len(X_train_nice) > 0:
                    all_results[sc_name][family][f"nice_cf{suffix}"] = run_attribute_inference_single(
                        model_name=model_name,
                        input_type="nice_cf",
                        estimator=target_estimator,
                        X_train=X_train_nice,
                        X_eval=attack_X_eval,
                        sensitive_idx=sensitive_idx,
                        low_val=low_val,
                        high_val=high_val,
                        verbose=verbose,
                        include_x_ns=ix_ns,
                    )
                else:
                    all_results[sc_name][family][f"nice_cf{suffix}"] = AIAResult(
                        model_name, "nice_cf", feature_set=feature_set,
                    )

    return all_results


def run_aia_analysis(*args, **kwargs):
    """Back-compat wrapper: legacy name now runs attribute inference."""
    return run_attribute_inference_analysis(*args, **kwargs)


def aia_results_to_dataframe(all_results):
    """Flatten nested {scenario: {family: {input_type: AIAResult}}} into a DataFrame.

    Emits one row per (scenario, family, input_type, attacker), unpacking every
    numeric metric on the AttackResult (including the per-group metrics added
    by _compute_metrics: accuracy_<g>, error_rate_<g>, group_disparity, etc.).
    """
    rows = []
    for sc, families in all_results.items():
        for fam, inputs in families.items():
            for itype, aia_res in inputs.items():
                for att_name, att_res in aia_res.attacks.items():
                    # getattr keeps this usable for MIA-shaped results, which
                    # carry .attacks but no .variant/.feature_set.
                    row = {
                        "scenario":        sc,
                        "family":          fam,
                        "input_type":      itype,
                        "attacker":        att_name,
                        "attack_variant":  getattr(aia_res, "variant", "n/a"),
                        "attack_features": getattr(aia_res, "feature_set", "n/a"),
                    }
                    row.update(att_res.metrics)
                    rows.append(row)
    return pd.DataFrame(rows)


def run_standalone_attack(
    methods=None,
    families=None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    include_x_ns: bool = True,
    include_nice_cf: bool = False,
):
    from pipeline.config import MODEL_FAMILIES, AUGMENTATION_METHOD

    methods = methods or [AUGMENTATION_METHOD]
    families = families or MODEL_FAMILIES
    out = output_dir or OUTPUT_DIR
    data = load_and_split(augmentation_method=methods[0], output_dir=out, verbose=verbose)

    training_sets = {"baseline": (data.X_train, data.y_train)}
    for method in methods:
        X_aug, y_aug, _ = augment_data(
            data.X_train, data.y_train, method,
            dataset_tag="compas", verbose=verbose,
        )
        training_sets[f"augmented_{method}"] = (X_aug, y_aug)

    results = run_attribute_inference_analysis(
        training_sets=training_sets,
        families=families,
        data=data,
        nice_cf_results={},
        verbose=verbose,
        include_x_ns=include_x_ns,
        include_nice_cf=include_nice_cf,
        generate_missing_cf=True,
    )
    df = aia_results_to_dataframe(results)
    suffix = "" if include_x_ns else "_outputs_only"
    path = os.path.join(out, "results", f"attribute_inference_results{suffix}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    if verbose:
        print(f"[AIA] Saved -> {path}")
    return results

'''
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the attribute inference attack independently.")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Augmentation methods to include (default: current config method).")
    parser.add_argument("--families", nargs="+",
                        choices=["logistic_regression", "random_forest", "xgboost", "lr", "rf", "xgb"],
                        default=None)
    parser.add_argument("--outputs-only", action="store_true",
                        help=("Attacker sees only target-model outputs. Default is the "
                              "classical threat model, where it also sees the record's "
                              "non-sensitive attributes."))
    parser.add_argument("--nice-cf", action="store_true",
                        help=("Also evaluate each attacker on NiCE counterfactual queries "
                              "(a 'nice_cf' input_type alongside the real held-out test "
                              "rows). Generated on the fly against each AIA target model "
                              "unless precomputed results are supplied programmatically. "
                              "On-the-fly generation always queries the full AIA validation "
                              "set (never the AIA test set). Requires the NICEx package."))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    alias = {"lr": "logistic_regression", "rf": "random_forest", "xgb": "xgboost"}
    families = [alias.get(n, n) for n in (args.families or [])] or None
    run_standalone_attack(methods=args.methods, families=families,
                          verbose=not args.quiet,
                          include_x_ns=not args.outputs_only,
                          include_nice_cf=args.nice_cf)
'''