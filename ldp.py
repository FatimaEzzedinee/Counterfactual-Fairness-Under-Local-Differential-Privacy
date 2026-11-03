from __future__ import annotations
import os, sys
from dataclasses import dataclass
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import (
    LDP_EPSILON, FEATURE_COLS, PROTECTED_COL,
    GROUP_UNPRIV_LABEL, RANDOM_STATE,
)

_PROTECTED_IDX = FEATURE_COLS.index(PROTECTED_COL)


@dataclass
class LDPResult:
    X_train_ldp:    np.ndarray
    y_train_ldp:    np.ndarray
    X_aug_ldp:      np.ndarray
    y_aug_ldp:      np.ndarray
    epsilon:        float
    flip_prob:      float
    n_flipped:      int


def randomised_response(values, epsilon=LDP_EPSILON, random_state=RANDOM_STATE):
    values = np.asarray(values, dtype=int).copy()
    rng = np.random.default_rng(random_state)
    p_flip = 1.0 / (1.0 + np.exp(epsilon))
    flip_mask = rng.random(len(values)) < p_flip
    values[flip_mask] = 1 - values[flip_mask]
    return values, p_flip


def apply_ldp(data, epsilon=LDP_EPSILON, verbose=True):
    p_flip = 1.0 / (1.0 + np.exp(epsilon))
    if verbose:
        print(f"[LDP] Applying Randomised Response to {PROTECTED_COL}")
        print(f"      eps={epsilon:.2f}  p_flip={p_flip:.4f} ({p_flip*100:.1f}% flipped on average)")

    X_train_ldp = data.X_train.copy()
    orig = X_train_ldp[:, _PROTECTED_IDX].copy()
    perturbed, _ = randomised_response(orig, epsilon=epsilon, random_state=RANDOM_STATE)
    X_train_ldp[:, _PROTECTED_IDX] = perturbed

    n_flipped = int((perturbed != orig).sum())
    if verbose:
        print(f"      Flipped {n_flipped:,}/{len(orig):,} values ({n_flipped/len(orig)*100:.1f}%)")
        print(f"      {GROUP_UNPRIV_LABEL} fraction: {orig.mean()*100:.1f}% -> {perturbed.mean()*100:.1f}%")


    # TO CHECK - FATIMA
    X_aug_ldp = np.vstack([X_train_ldp, data.X_cf_race])
    y_aug_ldp = np.concatenate([data.y_train, data.y_cf_race])

    if verbose:
        print(f"      Augmented LDP training set: {len(X_aug_ldp):,} rows")

    return LDPResult(
        X_train_ldp=X_train_ldp,
        y_train_ldp=data.y_train.copy(),
        X_aug_ldp=X_aug_ldp,
        y_aug_ldp=y_aug_ldp,
        epsilon=epsilon, flip_prob=p_flip, n_flipped=n_flipped,
    )

def apply_ldp_multi_eps(data, epsilons, verbose=True):
    """Apply Randomised Response LDP for each epsilon in *epsilons*.

    Parameters
    ----------
    data     : DataBundle from data_preparation.load_and_split().
    epsilons : Iterable of float epsilon values (privacy budgets).
    verbose  : Print per-epsilon summary.

    Returns
    -------
    Dict mapping epsilon_label (str) -> LDPResult.
    e.g. {"ldp_eps0.1": LDPResult(...), "ldp_eps1.0": LDPResult(...), ...}
    """
    results = {}
    for eps in epsilons:
        label = "ldp_eps{}".format(eps)
        if verbose:
            print("\n[LDP-multi] epsilon={}".format(eps))
        results[label] = apply_ldp(data, epsilon=eps, verbose=verbose)
    return results

