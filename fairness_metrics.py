from __future__ import annotations
import os, sys, warnings
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import (
    PROTECTED_COL, PROTECTED_PRIV, PROTECTED_UNPRIV, TARGET_COL,
    FEATURE_COLS, GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL,
)


@dataclass
class FairnessResult:
    model_name: str
    group:      Dict[str, float] = field(default_factory=dict)
    individual: Dict[str, float] = field(default_factory=dict)


def _confusion_by_group(y_true, y_pred, mask):
    yt, yp = y_true[mask], y_pred[mask]
    tp = int(((yt==1)&(yp==1)).sum())
    fp = int(((yt==0)&(yp==1)).sum())
    tn = int(((yt==0)&(yp==0)).sum())
    fn = int(((yt==1)&(yp==0)).sum())
    tpr  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    fpr  = fp/(fp+tn) if (fp+tn)>0 else 0.0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    pos  = (tp+fp)/len(yt) if len(yt)>0 else 0.0
    return dict(tpr=tpr, fpr=fpr, precision=prec, pos_rate=pos)

# inequality / unfairness measure over whether each prediction was correct.
# Creates a binary vector:
#1.0 where prediction is correct
#0.0 where prediction is wrong
# So if predictions are:
#equally correct for everyone, inequality is low
#correct for some and wrong for others, inequality is higher
# mu = b.mean() This is the accuracy of the model.
# if mu == 0: return 0.0 If the model got everything wrong, the mean is 0, so dividing by mu would fail.
# r = b / mu Normalizes each entry by the average success rate.
# r[r==0] = 1e-10 Avoids log(0) in the next step.
# np.mean(r * np.log(r))
#This is the Theil index formula. 
# 0 means perfectly equal outcomes, Larger values mean more inequality in who gets correct predictions
# individual-level benefit inequality.
# Theil (individual inequality in prediction benefit/correctness)
#       Ideal: 0. Range: [0, +inf). Lower is better.
#       0 => more equal prediction outcomes across individuals.
#       Larger values => more unequal distribution of correct predictions.

def _theil(y_true, y_pred):
    b = (y_pred==y_true).astype(float)
    mu = b.mean()
    if mu==0: return 0.0
    r = b/mu; r[r==0]=1e-10
    return float(np.mean(r*np.log(r)))


# Interpretation:
# SPD   : ideal 0, range [-1,1], lower |.| is better; + => privileged gets more positives
# DI    : ideal 1, range [0,inf), closer to 1 is better; <1 => unprivileged gets fewer positives
# EOD   : ideal 0, range [-1,1], lower |.| is better; + => privileged has higher TPR
# AOD   : ideal 0, range [-1,1], lower |.| is better; + => privileged favored on avg
# EqOdds: ideal 0, range [0,1], lower is better; worst gap in TPR/FPR across groups
# PP    : ideal 0, range [-1,1], lower |.| is better; + => privileged has higher precision
# Theil : ideal 0, range [0,inf), lower is better; larger => more unequal correctness
# TPR_* / Precision_*: higher is better; FPR_*: lower is better; PosRate_*: compare across groups

# Group-specific outputs:
# TPR_*       : higher is better
# FPR_*       : lower is better
# PosRate_*   : not inherently better/worse; compare across groups
# Precision_* : higher is better
def compute_group_fairness(y_true, y_pred, protected,
                           priv_val=PROTECTED_PRIV, unpriv_val=PROTECTED_UNPRIV):
    ps = _confusion_by_group(y_true, y_pred, protected==priv_val)
    us = _confusion_by_group(y_true, y_pred, protected==unpriv_val)
    pl, ul = GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL
    # SPD   (Statistical Parity Difference) = pos_rate(priv) - pos_rate(unpriv)
    #       Ideal: 0. Range: [-1, 1]. Lower |value| is better.
    #       > 0 => privileged group gets more positive predictions.
    #       < 0 => unprivileged group gets more positive predictions.
    spd = ps["pos_rate"] - us["pos_rate"]  # Statistical Parity Difference

    # DI    (Disparate Impact) = pos_rate(unpriv) / pos_rate(priv)
    #       Ideal: 1. Range: [0, +inf). Closer to 1 is better.
    #       < 1 => unprivileged group receives fewer positive outcomes.
    #       > 1 => unprivileged group receives more positive outcomes
    di  = us["pos_rate"]/ps["pos_rate"] if ps["pos_rate"]>0 else 0.0

    # EOD   (Equal Opportunity Difference) = TPR(priv) - TPR(unpriv)
    #       Ideal: 0. Range: [-1, 1]. Lower |value| is better.
    #       > 0 => privileged group has higher true positive rate.
    #       < 0 => unprivileged group has higher true positive rate.
    eod = ps["tpr"] - us["tpr"]

    # AOD   (Average Odds Difference) = 0.5 * [(FPR(priv)-FPR(unpriv)) + (TPR(priv)-TPR(unpriv))]
    #       Ideal: 0. Range: [-1, 1]. Lower |value| is better.
    #       > 0 => privileged group favored on average in error/benefit rates.
    #       < 0 => unprivileged group favored on average.
    aod = 0.5*((ps["fpr"]-us["fpr"])+(ps["tpr"]-us["tpr"]))

    # EqOdds (Equalized Odds gap) = max(|TPR gap|, |FPR gap|)
    #       Ideal: 0. Range: [0, 1]. Lower is better.
    #       Measures worst-case group gap in TPR or FPR.
    eq  = max(abs(ps["tpr"]-us["tpr"]), abs(ps["fpr"]-us["fpr"]))

    # PP    (Precision Parity Difference) = Precision(priv) - Precision(unpriv)
    #       Ideal: 0. Range: [-1, 1]. Lower |value| is better.
    #       > 0 => positive predictions are more reliable for privileged group.
    #       < 0 => positive predictions are more reliable for unprivileged group.
    pp  = ps["precision"] - us["precision"]

    return {
        "SPD": float(spd), "DI": float(di), "EOD": float(eod), "AOD": float(aod),
        "EqOdds": float(eq), "PP": float(pp), "Theil": _theil(y_true, y_pred),
        f"TPR_{pl}": ps["tpr"], f"TPR_{ul}": us["tpr"],
        f"FPR_{pl}": ps["fpr"], f"FPR_{ul}": us["fpr"],
        f"PosRate_{pl}": ps["pos_rate"], f"PosRate_{ul}": us["pos_rate"],
        f"Precision_{pl}": ps["precision"], f"Precision_{ul}": us["precision"],
    }

# Counterfactual fairness: for every test point, flip the protected attribute
# (race_enc: 0=White ↔ 1=Black) and check whether the model’s prediction changes.
# Returns the fraction of predictions that stay the SAME after the flip.
# Ideal = 1.0 (model ignores race entirely). Lower = more race-sensitive.
#
# Safe to flip with (1 - value) since race_enc is already binary {0, 1}.
def compute_counterfactual_fairness(estimator, X_orig):
    prot_idx = FEATURE_COLS.index(PROTECTED_COL)
    col = X_orig[:, prot_idx]
    unique_vals = np.unique(col)
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        warnings.warn(
            f"[cf_fairness] Protected column values are not exactly {{0,1}} "
            f"(found {unique_vals}). Flip may be incorrect.")
    X_cf = X_orig.copy()
    X_cf[:, prot_idx] = 1.0 - col
    return float(np.mean(estimator.predict(X_orig) == estimator.predict(X_cf)))

# What it does:
# predicts labels for all samples
# finds the k nearest neighbors of each sample in feature space
# excludes the sample itself (idx[:, 1:])
# checks whether each sample’s prediction matches its neighbors’ predictions
# returns the average match rate
# Ideal: 1.0, Range: [0, 1], Higher is better
# 1.0 means every sample gets the same prediction as all of its nearest neighbors
# lower values mean nearby/similar samples often receive different predictions
def compute_consistency(estimator, X, k=5):
    y = estimator.predict(X)
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X)
    _, idx = nn.kneighbors(X)
    return float((y[idx[:, 1:]]==y[:, None]).mean())


def compute_fairness(model_result, X_test, y_test, protected_test,
                     k_consistency=5, verbose=True):
    if verbose:
        print(f"  [fairness] {model_result.name} ... ", end="")
    # Recompute predictions fresh from X_test so that ALL metrics
    # (group fairness, consistency, CF fairness) use exactly the same
    # predictions from the same X_test. This avoids any mismatch between
    # the y_pred stored at training time and the X_test passed here.
    y_pred = model_result.estimator.predict(X_test)
    group = compute_group_fairness(y_test, y_pred, protected_test)
    ind = {}
    ind["consistency"] = compute_consistency(model_result.estimator, X_test, k_consistency)
    # CF fairness: flip the race column directly on the (scaled) test set.
    # No external CF input needed — computed here unconditionally.
    ind["cf_fairness"] = compute_counterfactual_fairness(model_result.estimator, X_test)
    if verbose:
        spd = group["SPD"]; di = group["DI"]; eod = group["EOD"]
        con = ind["consistency"]; cf = ind["cf_fairness"]
        print(f"SPD={spd:+.3f}  DI={di:.3f}  EOD={eod:+.3f}  Consistency={con:.3f}  CF={cf:.3f}")
    return FairnessResult(model_name=model_result.name, group=group, individual=ind)


def run_fairness_analysis(scenarios, data, verbose=True):
    if verbose:
        print("" + "="*60)
        print("  FAIRNESS ANALYSIS  (protected: {} -- {} vs {})".format(
            PROTECTED_COL, GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL))
        print("="*60)
    prot_idx  = FEATURE_COLS.index(PROTECTED_COL)
    prot_test = data.X_test[:, prot_idx]
    # CF fairness is computed inside compute_fairness by flipping the race
    # column on the test set — no external CF array needed here.
    results = {}
    for sc_name, sm in scenarios.items():
        results[sc_name] = {}
        if verbose: print(f"[Scenario: {sc_name}]")
        for family, model_res in sm.results.items():
            results[sc_name][family] = compute_fairness(
                model_result=model_res, X_test=data.X_test, y_test=data.y_test,
                protected_test=prot_test, verbose=verbose)
    return results