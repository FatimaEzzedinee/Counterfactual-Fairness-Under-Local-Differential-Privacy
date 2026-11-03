from __future__ import annotations
import os, sys, warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import (
    DATASET_CSV, FEATURE_COLS, TARGET_COL, PROTECTED_COL, PROTECTED_UNPRIV,
    GROUP_UNPRIV_LABEL, RANDOM_STATE, TEST_SIZE, VAL_SIZE, OUTPUT_DIR,
    DATASET_NAME, AUGMENTATION_METHOD, AUG_DROP_IDENTICAL_CFS,
    AUG_SYNTHETIC_WEIGHT, ACTIVE_DATASET_SPEC, SCM_DEPENDENT_COLS,
)
from pipeline.cf_generation import SCMParams, fit_scm, generate_training_race_cfs, generate_training_flip_cfs, main_MM

_MM_CF_DIR = os.path.join(OUTPUT_DIR, "results", "mm_cf_arrays")


@dataclass
class DataBundle:
    X_train:  np.ndarray = field(default_factory=lambda: np.array([]))
    y_train:  np.ndarray = field(default_factory=lambda: np.array([]))
    X_val:    np.ndarray = field(default_factory=lambda: np.array([]))
    y_val:    np.ndarray = field(default_factory=lambda: np.array([]))
    X_test:   np.ndarray = field(default_factory=lambda: np.array([]))
    y_test:   np.ndarray = field(default_factory=lambda: np.array([]))
    X_cf_race:   np.ndarray = field(default_factory=lambda: np.array([]))
    y_cf_race:   np.ndarray = field(default_factory=lambda: np.array([]))
    X_aug:    np.ndarray = field(default_factory=lambda: np.array([]))
    y_aug:    np.ndarray = field(default_factory=lambda: np.array([]))
    train_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    full_df:      Optional[pd.DataFrame] = None
    scm:          Optional[SCMParams]    = None
    feature_cols: list = field(default_factory=list)


def augment_data(X_train: np.ndarray, y_train: np.ndarray,
                 method: str,
                 dataset_tag: str = DATASET_NAME.lower(),
                 verbose: bool = True):
    """Return (X_aug, y_aug, sample_weight) for *method* applied to X_train / y_train.

    Single source of truth for augmentation shared between the initial training
    phase and the LDP phase in full_ablation_pipeline.py.

    Parameters
    ----------
    X_train     : raw (unscaled) training features -- the SCM (for method='SCM')
                  is always fit fresh on exactly this array, never on a
                  pre-computed one. When the caller passes clean X_train this
                  reproduces the true causal graph; when the caller passes an
                  LDP-perturbed X_train_ldp, the SCM is fit on the perturbed
                  protected column -- there is no hidden dependency on clean
                  data. (Previously this function accepted a pre-fit `scm`
                  parameter and reused it even when X_train was LDP-perturbed,
                  which silently leaked clean-data structure into the
                  'privacy + causal-fairness' scenarios. Fixed: the SCM is
                  now always refit on the X_train actually passed in.)
    y_train     : training labels
    method      : 'SCM' | 'update_labels' | 'add_comparators' | 'flip_only'
                  ('flip_only' is the non-causal baseline: twins flip only the
                  protected attribute, see generate_training_flip_cfs)
    dataset_tag : dataset key for main_MM config lookups ('compas' / 'adult')
    verbose     : passed through to generate_training_race_cfs

    Returns
    -------
    X_aug         : augmented feature matrix (original + synthetic rows)
    y_aug         : augmented labels
    sample_weight : per-row weight array, or None if AUG_SYNTHETIC_WEIGHT is None.
                    Original rows have weight 1.0; synthetic rows have weight
                    AUG_SYNTHETIC_WEIGHT (set in config.py).

    Config flags applied here (all default to original behaviour):
      AUG_DROP_IDENTICAL_CFS      — drop synthetic rows identical to source
      AUG_SYNTHETIC_WEIGHT        — down-weight synthetic rows during training
    AUG_RELABEL_DISTANCE_PERCENTILE is applied inside main_MM (cf_generation.py).
    """
    n_orig = len(X_train)

    if method == "SCM":
        if verbose:
            print("[data] Fitting SCM on the given training data and generating {} counterfactuals ...".format(PROTECTED_COL))
        scm = fit_scm(X_train, verbose=verbose)
        X_cf, y_cf, src_idx = generate_training_race_cfs(X_train, y_train, scm, verbose=verbose,
                                                         return_src_idx=True)

    elif method == "flip_only":
        if verbose:
            print("[data] Generating flip-only {} twins (no SCM) ...".format(PROTECTED_COL))
        X_cf, y_cf, src_idx = generate_training_flip_cfs(X_train, y_train, verbose=verbose,
                                                         return_src_idx=True)

    elif method == "update_labels":
        if verbose:
            print("[data] Running MM - update_labels ...")
        # update_labels returns a relabelled version of X_train (same rows,
        # updated labels) — no new rows are appended here.
        X_aug, y_aug = main_MM(X_train, y_train, "update_labels", dataset_tag)
        # No synthetic rows to weight or filter: return directly.
        sample_weight = None  # uniform (update_labels doesn't append rows)
        if verbose:
            n_changed = int((y_aug != y_train).sum())
            print(f"[data] Labels changed: {n_changed}/{n_orig} "
                  f"({n_changed/n_orig*100:.1f}%)")
            
        #X_aug = np.vstack([X_train, X_aug])
        #y_aug = np.concatenate([y_train, y_aug])

        return X_aug, y_aug, sample_weight

    else:  # add_comparators
        if verbose:
            print("[data] Running MM - add_comparators ...")
        X_cf, y_cf, src_idx = main_MM(X_train, y_train, "add_comparators", dataset_tag,
                                      return_src_idx=True)

    return finalize_cf_augmentation(X_train, y_train, X_cf, y_cf, src_idx=src_idx, verbose=verbose)


def finalize_cf_augmentation(X_train: np.ndarray, y_train: np.ndarray,
                             X_cf: np.ndarray, y_cf: np.ndarray,
                             src_idx: np.ndarray, verbose: bool = True):
    """Shared post-processing for any CF-row-appending augmentation method
    (X_cf[k] is the twin of X_train[src_idx[k]]): drop synthetic rows
    identical to their source
    (AUG_DROP_IDENTICAL_CFS), append the CF rows to the original training
    set, and build a sample_weight array (AUG_SYNTHETIC_WEIGHT) if
    configured.

    Extracted out of augment_data() (used there by 'SCM' / 'add_comparators')
    so other callers building their own CF rows from a custom-fit SCM --
    e.g. pipeline/dp_causal_analysis.py's LDP-aware SCM variants -- get the
    exact same drop/weight behaviour without duplicating it.

    Returns (X_aug, y_aug, sample_weight).
    """
    n_orig = len(X_train)

    # --- apply AUG_DROP_IDENTICAL_CFS ---
    # Synthetic rows that are bit-for-bit identical to their source add noise
    # without new information. Drop them when the flag is enabled.
    if AUG_DROP_IDENTICAL_CFS and len(X_cf) > 0:
        n_cf = len(X_cf)
        diff_mask = np.any(X_cf != np.asarray(X_train)[src_idx], axis=1)
        n_dropped = int((~diff_mask).sum())
        if n_dropped > 0 and verbose:
            print(f"[data] AUG_DROP_IDENTICAL_CFS: dropped {n_dropped}/{n_cf} identical CFs")
        X_cf = X_cf[diff_mask]
        y_cf = y_cf[diff_mask]

    X_aug = np.vstack([X_train, X_cf])
    y_aug = np.concatenate([y_train, y_cf])

    # --- build sample_weight ---
    # Original rows get weight 1.0; synthetic rows get AUG_SYNTHETIC_WEIGHT.
    # None means uniform weights (sklearn default — no weight array passed).
    if AUG_SYNTHETIC_WEIGHT is not None:
        sample_weight = np.ones(len(X_aug), dtype=float)
        sample_weight[n_orig:] = float(AUG_SYNTHETIC_WEIGHT)
        if verbose:
            print(f"[data] Sample weights: original=1.0  synthetic={AUG_SYNTHETIC_WEIGHT}")
    else:
        sample_weight = None

    return X_aug, y_aug, sample_weight


def _load_dataset():
    """Load the ACTIVE dataset (pipeline.config.ACTIVE_DATASET_SPEC) via its
    registered DatasetSpec.loader, then apply the schema-agnostic post
    processing every dataset needs (drop rows with missing feature/target
    values). Dataset-specific parsing/filtering/encoding lives in
    pipeline/datasets/<name>.py, not here.
    """
    df = ACTIVE_DATASET_SPEC.loader(DATASET_CSV)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return df


def load_and_split(augmentation_method: str = AUGMENTATION_METHOD,
                   output_dir: str = None,
                   verbose: bool = True):
    """Load the active dataset, split, fit SCM and build augmented training set.

    Parameters
    ----------
    augmentation_method : one of 'SCM', 'update_labels', 'add_comparators'
    output_dir          : where to cache MM-CF arrays (defaults to OUTPUT_DIR)
    verbose             : print progress
    """
    df = _load_dataset()
    if verbose:
        print(f"[data] Loaded {len(df):,} rows for dataset={DATASET_NAME} from {DATASET_CSV}")
        print(f"       {TARGET_COL}=1        {df[TARGET_COL].mean()*100:.1f}%")
        print(f"       {GROUP_UNPRIV_LABEL:<10} ({PROTECTED_COL}={PROTECTED_UNPRIV})  {(df[PROTECTED_COL] == PROTECTED_UNPRIV).mean()*100:.1f}%")

    X_all = df[FEATURE_COLS].values.astype(float)
    y_all = df[TARGET_COL].values.astype(int)
    strat = df[TARGET_COL].astype(str) + "_" + df[PROTECTED_COL].astype(str)

    X_tv, X_test, y_tv, y_test, idx_tv, _ = train_test_split(
        X_all, y_all, np.arange(len(df)), test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=strat)

    strat_tv = strat.values[idx_tv]
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE) if VAL_SIZE > 0 else 0.0

    if val_frac > 0:
        X_train, X_val, y_train, y_val, idx_train, _ = train_test_split(
            X_tv, y_tv, idx_tv, test_size=val_frac,
            random_state=RANDOM_STATE, stratify=strat_tv)
    else:
        X_train, y_train, idx_train = X_tv, y_tv, idx_tv
        X_val   = np.empty((0, len(FEATURE_COLS)), dtype=float)
        y_val   = np.empty(0, dtype=int)

    if verbose:
        print(f"[data] Train:{len(X_train):,}  Val:{len(X_val):,}  Test:{len(X_test):,}")

    # Counterfactual-fairness evaluation (fairness_metrics.py) flips the
    # protected column directly and does NOT need an SCM. `scm` is only
    # required by augment_data(..., method='SCM', ...) below, so only fit it
    # when the active dataset actually has an SCM spec -- fitting is not
    # possible (and not needed) for datasets without one.
    X_cf_race = np.empty((0, len(FEATURE_COLS)), dtype=float)
    y_cf_race = np.empty(0, dtype=int)

    if SCM_DEPENDENT_COLS:
        scm = fit_scm(X_train, verbose=verbose)
    else:
        scm = None
        if verbose:
            print(f"[data] No SCM spec for dataset={DATASET_NAME} -- "
                  f"augmentation_method='SCM' is unavailable (update_labels / "
                  f"add_comparators are unaffected).")

    if verbose:
        print(f"[data] Building augmented set: {augmentation_method} ...")

    X_aug, y_aug, _ = augment_data(X_train, y_train, augmentation_method, dataset_tag=DATASET_NAME.lower(), verbose=verbose)

    # Keep X_cf_race / y_cf_race for the DataBundle fields.
    # For SCM: the CFs are X_aug minus X_train.
    # For update_labels: there are no appended CFs (labels are updated in-place).
    # For add_comparators: the CFs are X_aug minus X_train.
    if augmentation_method == 'SCM' or augmentation_method == 'add_comparators':
        X_cf_race = X_aug[len(X_train):]
        y_cf_race = y_aug[len(y_train):]
    # else update_labels: X_cf_race stays as the empty arrays initialised above


    if verbose:
        print(f"[data] Augmented training set: {len(X_aug):,} rows ")

    # Persist MM-CF arrays for reuse — one subfolder per augmentation method
    _out = output_dir or OUTPUT_DIR
    _cache_dir = os.path.join(_out, "results", "mm_cf_arrays", augmentation_method)
    try:
        os.makedirs(_cache_dir, exist_ok=True)
        np.save(os.path.join(_cache_dir, "X_cf_race.npy"), X_cf_race)
        np.save(os.path.join(_cache_dir, "y_cf_race.npy"), y_cf_race)
        np.save(os.path.join(_cache_dir, "X_aug.npy"),     X_aug)
        np.save(os.path.join(_cache_dir, "y_aug.npy"),     y_aug)
        np.save(os.path.join(_cache_dir, "X_train.npy"),   X_train)
        np.save(os.path.join(_cache_dir, "y_train.npy"),   y_train)
        np.save(os.path.join(_cache_dir, "X_test.npy"),    X_test)
        np.save(os.path.join(_cache_dir, "y_test.npy"),    y_test)
        if verbose:
            print(f"[data] MM-CF arrays saved -> {_cache_dir}")
    except Exception as e:
        warnings.warn(f"[data] Could not save MM-CF arrays: {e}")

    return DataBundle(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_cf_race=X_cf_race, y_cf_race=y_cf_race,
        X_aug=X_aug, y_aug=y_aug,
        train_indices=idx_train,
        full_df=df, scm=scm,
        feature_cols=FEATURE_COLS,
    )