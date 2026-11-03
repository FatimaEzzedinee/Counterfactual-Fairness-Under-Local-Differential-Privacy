"""
full_ablation_pipeline.py -- Full privacy x fairness ablation study.

Dataset-pluggable: which dataset runs is controlled by pipeline/datasets/
(see pipeline/datasets/base.py for the contract). Defaults to COMPAS; pass
--dataset <name> or set ABLATION_DATASET=<name> to switch (e.g. "adult",
already registered as a worked second example). Everything below runs
against whichever dataset is active -- no other file needs to change.

This file does NOT reimplement any algorithm. It only orchestrates the
already-existing, already-tested building blocks in this package:

    data_preparation.py         -- dataset loading / splitting / SCM fitting
    cf_generation.py            -- SCM + "mutatis-mutandis" counterfactual
                                    augmentation (the proposed causal/
                                    counterfactual-reasoning fairness method)
    models.py                   -- LR / RF / XGB training + eval
    ldp.py                      -- Local Differential Privacy (randomised
                                    response) on the protected attribute
    fairlearn_pipeline.py       -- FairLearn constrained training
                                    (ExponentiatedGradient / ThresholdOptimizer)
    fairness_metrics.py         -- group + individual + counterfactual
                                    fairness metrics
    nice_cf.py                  -- NiCE counterfactual explanations (used as
                                    the MIA adversary's synthetic-data proxy
                                    and scored for their own CF quality)
    mia_analysis.py             -- Membership Inference Attack (21 attackers
                                    + ensembles, via ensemble_mia-main)
    attribute_inference_attack.py -- Attribute Inference Attack (LR/RF
                                    attacker models + confidence baselines)
    dp_causal_analysis.py       -- the LDP-only correction ('SCM_ldp_corrected':
                                    fits the causal SCM on the LDP-perturbed
                                    protected column, correcting for the known
                                    randomized-response flip probability) plus
                                    two classifier-free causal-graph
                                    comparisons (see that module's docstring):
                                    one across the Category-4 augmentation
                                    methods with no privacy involved, one
                                    across privacy budgets for the correction.
    dashboard.py                 -- single-page interactive HTML dashboard
                                    over every full_ablation_*/results/
                                    ablation_summary.csv found; auto-
                                    regenerated at the end of every run
                                    (pass --skip-dashboard to disable).
    research_dashboard.py        -- reshapes that same payload into one
                                    panel per research question (does SCM
                                    augmentation help; how LDP distorts the
                                    SCM; what remains recoverable; does the
                                    correction restore fairness toward the
                                    non-private ceiling); auto-regenerated
                                    alongside dashboard.py, same flag.

Ablation matrix
----------------
  1. no_privacy_no_fairness      : "baseline"
                                    -> plain LR/RF/XGB on raw training data.

  2. privacy_only                : "ldp_baseline_eps{e}"
                                    -> randomised-response LDP on race_enc at
                                       each epsilon in --ldp-epsilons.

  3. fairness_fairlearn_only     : "fairlearn_{constraint}"
                                    -> FairLearn baseline / EG-DemoParity /
                                       EG-EqOdds / EG-TPRParity /
                                       TO-DemoParity / TO-EqOdds.

  4. fairness_causal_only        : "augmented_{method}"
                                    -> proposed SCM / update_labels /
                                       add_comparators / add_comparators_bidir
                                       counterfactual-augmentation methods.
                                       'SCM' fits fit_scm() fresh on X_train
                                       here (clean data).

  4b. fairness_causal_combined   : "augmented_{m1}+{m2}[+{m3}]"   (opt-in: --method-ablation
                                    or --combos)
                                    -> the Category-4 methods applied together.
                                       update_labels relabels the original rows
                                       first; SCM / add_comparators[_bidir] each
                                       then generate their CF rows from those
                                       rows and all are appended (see
                                       _augment_combined). No privacy. Also
                                       written, with the baseline and the single
                                       methods, to method_ablation.csv.

  5. privacy_plus_fairness       : "ldp_{method}_eps{e}"              (LDP + causal CF,
                                       method in ALL_AUG_METHODS + dp_causal_analysis
                                       .LDP_CORRECTED_METHODS = SCM_ldp_corrected /
                                       update_labels_ldp_corrected /
                                       add_comparators_ldp_corrected /
                                       add_comparators_bidir_ldp_corrected)
                                    "fairlearn_ldp_eps{e}_{constraint}" (LDP + FairLearn)
                                    'SCM' here fits fit_scm() fresh on
                                    X_train_ldp (same function, same code
                                    path as category 4 -- just handed
                                    LDP-perturbed data instead of clean data,
                                    so it is the naive/uncorrected fit under
                                    privacy). Each '{method}_ldp_corrected'
                                    fits the same X_train_ldp but corrects
                                    for the known RR flip probability --
                                    'SCM_ldp_corrected' corrects fit_scm's
                                    single-covariate regression,
                                    'update_labels_ldp_corrected' /
                                    'add_comparators_ldp_corrected' correct
                                    main_MM's shared multi-covariate
                                    do_white_*/do_male_*.csv regression --
                                    ONE correction algorithm, two regressions
                                    (see dp_causal_analysis.py's docstring).

  +  AUGMENTATION CAUSAL-GRAPH COMPARISON (no privacy; classifier-free, its
     own CSVs): for each Category-4 method, refits the empirical
     protected -> dependent-column relationship on that method's augmented
     training data and compares it to the true SCM -- isolates how much
     each augmentation method itself distorts the causal graph, with no
     privacy involved. See augmentation_causal_graph_edges.csv /
     augmentation_causal_graph_summary.csv.

  +  DP CAUSAL-GRAPH COMPARISON (privacy; classifier-free, its own CSVs):
     repeated-sampling comparison (many independent RR draws per epsilon,
     NOT the one draw any Category-5a scenario trained on) of the
     uncorrected vs. corrected fit against the true fit, for EVERY
     augmentation method ('SCM', 'update_labels', 'add_comparators' --
     dp_causal_analysis.AUG_METHOD_CAUSAL_GRAPH). See
     dp_causal_graph_edges.csv / dp_causal_graph_summary.csv (columns
     aug_method, fit in {"uncorrected","corrected"}, epsilon, ...).

Every (scenario, family) pair trained anywhere above gets:
  * Model performance : accuracy, precision, recall, f1, auc_roc
  * Fairness           : SPD, DI, EOD, AOD, EqOdds, PP, Theil (group);
                         consistency, cf_fairness (individual / counterfactual)
  * NiCE CF quality     : flip_rate, proximity, plausibility, sparsity
  * Membership Inference Attack  : ~21 attackers + ensembles; auc_roc, pr_auc,
                         accuracy, precision, recall, f1, tpr, fpr,
                         mia_advantage, privacy_gain, tpr_at_fpr_{0,.001,.01,.1}
  * Attribute Inference Attack   : attacker_lr / attacker_rf / confidence
                         baselines / ensemble; same metric family as above
                         (aia_advantage / attribute_privacy_gain aliases)

All results are written as CSVs (one row per scenario x family x metric-group,
plus a wide `ablation_summary.csv` joining everything) to
`pipeline_outputs/full_ablation/results/`, together with a `run_manifest.json`
documenting the exact configuration, package versions, per-step timing and
any scenarios that failed.

Usage
-----
    python pipeline/full_ablation_pipeline.py                     # full run
    python pipeline/full_ablation_pipeline.py --quick-test         # fast smoke test
    python pipeline/full_ablation_pipeline.py --run-aia            # also run the attribute inference attack
    python pipeline/full_ablation_pipeline.py --ldp-epsilons 1 5
    python pipeline/full_ablation_pipeline.py --methods SCM add_comparators
    python pipeline/full_ablation_pipeline.py --method-ablation --no-ldp --no-fairlearn
    python pipeline/full_ablation_pipeline.py --combos update_labels+SCM
    python pipeline/full_ablation_pipeline.py --families lr rf
    python pipeline/full_ablation_pipeline.py --fl-constraints baseline eg_dp
    python pipeline/full_ablation_pipeline.py --dataset adult --quick-test
    ABLATION_DATASET=adult python pipeline/full_ablation_pipeline.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback
import warnings
from types import SimpleNamespace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_THIS_DIR)
for _p in [_BASE_DIR, os.path.join(_BASE_DIR, "ensemble_mia-main")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Dataset selection MUST happen before `import pipeline.config` --------
# pipeline.config resolves the active DatasetSpec (and every constant derived
# from it -- FEATURE_COLS, TARGET_COL, PROTECTED_COL, ...) once, at import
# time, from the ABLATION_DATASET env var. A --dataset CLI flag can only take
# effect if we read it BEFORE pipeline.config (or any module that imports it)
# is loaded, so we do a minimal manual scan of sys.argv here rather than
# waiting for argparse.
if "--dataset" in sys.argv:
    _idx = sys.argv.index("--dataset")
    if _idx + 1 < len(sys.argv):
        os.environ["ABLATION_DATASET"] = sys.argv[_idx + 1]
elif os.environ.get("ABLATION_DATASET") is None:
    for _arg in sys.argv:
        if _arg.startswith("--dataset="):
            os.environ["ABLATION_DATASET"] = _arg.split("=", 1)[1]

# --- Seed selection MUST also happen before `import pipeline.config` --------
# RANDOM_STATE (pipeline.config) is resolved once at import time from the
# ABLATION_SEED env var, and baked into module-level defaults (LR_PARAMS,
# RF_PARAMS, every `random_state=RANDOM_STATE` default) throughout the
# package -- same reasoning as the dataset pre-scan above. --seed lets a
# single run (or pipeline/run_multi_seed.py, one subprocess per seed) pick a
# reproducible seed other than the config default.
if "--seed" in sys.argv:
    _idx = sys.argv.index("--seed")
    if _idx + 1 < len(sys.argv):
        os.environ["ABLATION_SEED"] = sys.argv[_idx + 1]
elif os.environ.get("ABLATION_SEED") is None:
    for _arg in sys.argv:
        if _arg.startswith("--seed="):
            os.environ["ABLATION_SEED"] = _arg.split("=", 1)[1]

from pipeline.config import (
    OUTPUT_DIR, MODEL_FAMILIES, LDP_EPSILONS, NICE_MAX_SAMPLES, DATASET_NAME, FEATURE_COLS,
    PROTECTED_COL, RANDOM_STATE, ACTIVE_DATASET, ACTIVE_DATASET_SPEC,
)
import pipeline.config as _cfg
import pipeline.data_preparation as dp
import pipeline.models as ml
import pipeline.fairness_metrics as fm
import pipeline.ldp as ldp_mod
#import pipeline.mia_analysis as mia_mod
import pipeline.attribute_inference_attack as aia_mod
#import pipeline.nice_cf as ncf
import pipeline.cf_generation as cf_gen_mod
import pipeline.dp_causal_analysis as dpc_mod
import pipeline.dashboard as dashboard_mod
import pipeline.research_dashboard as research_dashboard_mod
import pipeline.paper_export as paper_mod
import pipeline.kusner_baselines as kusner_mod
from pipeline.data_preparation import augment_data

_PROTECTED_IDX = FEATURE_COLS.index(PROTECTED_COL)

# 'flip_only' is the non-causal baseline (twins flip only the protected
# attribute). It runs in Category 4 and 5a like the others, but is not in
# _COMBO_ORDER, so it never enters the method-ablation combinations, and it
# has no _ldp_corrected variant (it fits no regression to correct).
ALL_AUG_METHODS = ["SCM", "update_labels", "add_comparators", "add_comparators_bidir", "flip_only"]
ALL_FL_CONSTRAINTS = ["eg_dp", "eg_eo", "eg_tpr"] #, "to_dp", "to_eo"]

_CANONICAL_TO_TAG = {"logistic_regression": "lr", "random_forest": "rf", "xgboost": "xgb"}
_TAG_TO_CANONICAL = {v: k for k, v in _CANONICAL_TO_TAG.items()}

# Namespaced by dataset so `--dataset compas` and `--dataset adult` runs (or
# any other registered dataset) never overwrite each other's results.
_ABLATION_DIR = os.path.join(OUTPUT_DIR, "full_ablation_{}".format(ACTIVE_DATASET))
_RES_DIR = os.path.join(_ABLATION_DIR, "results")
_NICE_DIR = os.path.join(_ABLATION_DIR, "nice_cf_arrays")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    b = "=" * 70
    print("\n{}\n  {}\n{}".format(b, text, b))


def _sub_banner(text: str) -> None:
    print("\n  --- {} ---".format(text))


def _mkdirs() -> None:
    for d in (_ABLATION_DIR, _RES_DIR, _NICE_DIR):
        os.makedirs(d, exist_ok=True)


def _save_csv(df: pd.DataFrame, fname: str, verbose: bool = True) -> str:
    path = os.path.join(_RES_DIR, fname)
    df.to_csv(path, index=False)
    if verbose:
        print("  [save] {:<38} {:>6} rows -> {}".format(fname, len(df), path))
    return path


def _pkg_versions() -> dict:
    versions = {}
    for pkg in ["numpy", "pandas", "sklearn", "fairlearn", "xgboost", "statsmodels"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not installed"
    return versions


class _StepTimer:
    """Collects (label -> elapsed_seconds) for the run manifest."""

    def __init__(self):
        self.times = {}

    def __call__(self, label):
        return _StepTimerCtx(self, label)


class _StepTimerCtx:
    def __init__(self, parent, label):
        self.parent = parent
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.parent.times[self.label] = time.perf_counter() - self.t0


# ---------------------------------------------------------------------------
# FairLearn helper -- wraps train_fair_models into normalised ScenarioModels
# ---------------------------------------------------------------------------

def _train_fairlearn_block(data_obj, tag_prefix, constraints, family_tags, verbose=True):
    """Train FairLearn models and group them into {scenario: ScenarioModels}.

    Normalises FairLearn's short family tags ('lr'/'rf'/'xgb') to the
    canonical family names ('logistic_regression'/'random_forest'/'xgboost')
    used everywhere else in the pipeline, so downstream fairness / NiCE / MIA
    / AIA code sees ONE consistent family vocabulary.
    """
    try:
        from pipeline.fairlearn_pipeline import train_fair_models
    except ImportError as exc:
        warnings.warn("[fairlearn] pipeline.fairlearn_pipeline not importable: {}".format(exc))
        return {}

    try:
        fl_results = train_fair_models(data_obj, families=family_tags, verbose=verbose)
    except Exception as exc:
        warnings.warn("[fairlearn] training failed for prefix={}: {}\n{}".format(
            tag_prefix, exc, traceback.format_exc()))
        return {}

    by_constraint = {}
    for key, model_res in fl_results.items():
        ctag = getattr(model_res, "constraint_tag", key.rsplit("_", 1)[0])
        if constraints is not None and ctag not in constraints:
            continue
        canonical_fam = _TAG_TO_CANONICAL.get(model_res.family, model_res.family)
        sc_name = "{}_{}".format(tag_prefix, ctag)
        model_res.family = canonical_fam
        model_res.scenario = sc_name
        model_res.name = "{}_{}".format(sc_name, canonical_fam)
        by_constraint.setdefault(sc_name, ml.ScenarioModels(scenario=sc_name)).results[canonical_fam] = model_res
    return by_constraint


def _ldp_view(data, X_train_ldp: np.ndarray):
    """Lightweight DataBundle-like view that swaps in an LDP-perturbed X_train.

    Only carries the attributes fairlearn_pipeline.train_fair_models() and
    augment_data() actually read.
    """
    return SimpleNamespace(
        X_train=X_train_ldp, y_train=data.y_train,
        X_test=data.X_test, y_test=data.y_test, scm=data.scm,
        feature_cols=data.feature_cols,
    )


def _resolve_aug_method(method: str):
    """Map a (possibly virtual) method name to (real_method, bidir_flag)."""
    if method == "add_comparators_bidir":
        return "add_comparators", True
    if method == "add_comparators":
        return "add_comparators", False
    return method, None  # bidir flag irrelevant for SCM / update_labels


def _augment(X_train, y_train, method, verbose=True, epsilon=None):
    """Call augment_data() with the AUG_COMPARATORS_BIDIRECTIONAL flag set
    correctly for virtual method 'add_comparators_bidir', always restoring
    the flag afterwards (even on error).

    augment_data() always refits the SCM (for method='SCM') fresh on
    whichever X_train is passed in -- clean in Category 4, X_train_ldp in
    Category 5a -- so there is no separate clean-data-dependent code path
    here any more (see data_preparation.py::augment_data's docstring).

    method in dp_causal_analysis.LDP_CORRECTED_METHODS ('SCM_ldp_corrected' /
    'update_labels_ldp_corrected' / 'add_comparators_ldp_corrected') bypasses
    augment_data() entirely -- each needs epsilon (to know the RR flip
    probability to correct for), which augment_data() has no concept of.
    Only meaningful inside the LDP sweep, hence the required `epsilon`.
    """
    if method in dpc_mod.LDP_CORRECTED_METHODS:
        if epsilon is None:
            raise ValueError(
                f"_augment: method={method!r} requires an LDP epsilon -- it is only "
                f"valid inside the LDP sweep (Category 5a), never in Category 4."
            )
        return dpc_mod.augment_with_ldp_correction(
            X_train, y_train, method, epsilon, dataset_tag=DATASET_NAME.lower(), verbose=verbose)

    real_method, bidir = _resolve_aug_method(method)
    saved = _cfg.AUG_COMPARATORS_BIDIRECTIONAL
    try:
        if bidir is not None:
            _cfg.AUG_COMPARATORS_BIDIRECTIONAL = bidir
            cf_gen_mod.AUG_COMPARATORS_BIDIRECTIONAL = bidir
        return augment_data(X_train, y_train, real_method,
                            dataset_tag=DATASET_NAME.lower(), verbose=verbose)
    finally:
        _cfg.AUG_COMPARATORS_BIDIRECTIONAL = saved
        cf_gen_mod.AUG_COMPARATORS_BIDIRECTIONAL = saved


# ---------------------------------------------------------------------------
# Method ablation -- combining the augmentation methods (Category 4b)
# ---------------------------------------------------------------------------
# A combination is named by its components joined with '+', always in
# application order: update_labels (S2) first, then the row-appending
# generators SCM (S1) and add_comparators[_bidir] (S3). The two
# add_comparators variants are alternatives of the same method (S3), so a
# combination contains at most one of them.
_COMBO_ORDER = ["update_labels", "SCM", "add_comparators", "add_comparators_bidir"]
_S3_VARIANTS = {"add_comparators", "add_comparators_bidir"}


def _canonical_combo(combo: str) -> str:
    """Validate a '+'-joined combination and return it in canonical order."""
    parts = [p.strip() for p in combo.split("+") if p.strip()]
    unknown = [p for p in parts if p not in _COMBO_ORDER]
    if unknown:
        raise ValueError("combination {!r}: unknown method(s) {} -- expected components of {}".format(
            combo, unknown, _COMBO_ORDER))
    if len(set(parts)) < 2:
        raise ValueError("combination {!r} needs at least two distinct methods "
                         "(single methods already run in Category 4).".format(combo))
    if len(_S3_VARIANTS.intersection(parts)) > 1:
        raise ValueError("combination {!r}: 'add_comparators' and 'add_comparators_bidir' "
                         "are two variants of the same method -- use one.".format(combo))
    return "+".join(m for m in _COMBO_ORDER if m in parts)


def _method_combinations(methods: list) -> list:
    """Every combination of >= 2 of `methods` (at most one add_comparators
    variant each), smallest first -- the full method-ablation grid."""
    pool = [m for m in _COMBO_ORDER if m in methods]
    combos = []
    for r in range(2, len(pool) + 1):
        for c in itertools.combinations(pool, r):
            if len(_S3_VARIANTS.intersection(c)) <= 1:
                combos.append("+".join(c))
    return combos


def _augment_combined(X_train, y_train, combo, verbose=True):
    """Apply a '+'-joined combination of augmentation methods.

    1. update_labels (if present) relabels the ORIGINAL rows first, so every
       counterfactual row generated afterwards inherits the corrected label.
    2. Each row-appending method (SCM / add_comparators[_bidir]) is then run
       independently on the original features with those labels, and the
       synthetic rows of all of them are appended to the original rows.
       Generators are never chained: running add_comparators on SCM's output
       would refit its regression on data whose protected -> descendant
       effect SCM has already cancelled out.

    Each component goes through _augment(), i.e. exactly the code path (and
    config flags: AUG_DROP_IDENTICAL_CFS, AUG_COMPARATORS_BIDIRECTIONAL) it
    uses on its own in Category 4. AUG_SYNTHETIC_WEIGHT is applied to all
    appended rows. Returns (X_aug, y_aug, sample_weight).
    """
    components = combo.split("+")
    n_orig = len(X_train)
    y_used = y_train
    if "update_labels" in components:
        X_rel, y_used, _ = _augment(X_train, y_train, "update_labels", verbose=verbose)
        if X_rel.shape != X_train.shape or not np.array_equal(X_rel, X_train):
            raise RuntimeError("update_labels changed the feature rows -- cannot combine it safely.")
        y_used = np.asarray(y_used).astype(int)

    X_parts, y_parts = [X_train], [np.asarray(y_used)]
    for method in components:
        if method == "update_labels":
            continue
        X_m, y_m, _ = _augment(X_train, y_used, method, verbose=verbose)
        X_parts.append(X_m[n_orig:])
        y_parts.append(y_m[n_orig:])
        if verbose:
            print("  [combo] {}: +{} synthetic rows from {}".format(combo, len(X_m) - n_orig, method))

    X_aug = np.vstack(X_parts)
    y_aug = np.concatenate(y_parts)
    sample_weight = None
    if dp.AUG_SYNTHETIC_WEIGHT is not None:
        sample_weight = np.ones(len(X_aug), dtype=float)
        sample_weight[n_orig:] = float(dp.AUG_SYNTHETIC_WEIGHT)
    return X_aug, y_aug, sample_weight


_ABLATION_METRICS = ["accuracy", "f1", "auc_roc", "group_SPD", "group_DI", "group_EOD",
                     "group_AOD", "group_EqOdds", "ind_consistency", "ind_cf_fairness"]
_ABLATION_CATEGORIES = ["no_privacy_no_fairness", "fairness_causal_only", "fairness_causal_combined"]


_component_flags = paper_mod.component_flags


def _build_method_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, family) for the baseline, every single method
    and every combination (no privacy): which of S1/S2/S3 it uses, its
    metrics, and each metric's change relative to the same family's
    baseline (delta_<metric>)."""
    if summary.empty or "ablation_category" not in summary.columns:
        return pd.DataFrame()
    df = summary[summary["ablation_category"].isin(_ABLATION_CATEGORIES)].copy()
    if df.empty:
        return df
    flags = pd.DataFrame([_component_flags(m) for m in df["aug_method"]], index=df.index)
    metrics = [m for m in _ABLATION_METRICS if m in df.columns]
    out = pd.concat([df[["scenario", "family", "aug_method", "n_train"]], flags, df[metrics]], axis=1)
    base = out[out["scenario"] == "baseline"].set_index("family")
    for m in metrics:
        out["delta_" + m] = out[m] - out["family"].map(base[m]) if not base.empty else float("nan")
    s3_rank = {"none": 0, "one_sided": 1, "bidir": 2}
    out["_s3"] = out["S3_add_comparators"].map(s3_rank)
    out = out.sort_values(["family", "n_aug_components", "S2_update_labels", "S1_SCM", "_s3"])
    return out.drop(columns="_s3").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Row-flattening helpers
# ---------------------------------------------------------------------------

def _perf_rows(all_scenarios: dict, scenario_meta: dict) -> list:
    rows = []
    for sc, sm in all_scenarios.items():
        meta = scenario_meta.get(sc, {})
        for fam, res in sm.results.items():
            m = res.metrics or {}
            if not m:
                continue
            row = {"scenario": sc, "family": fam, "n_train": res.n_train}
            row.update(meta)
            row.update({
                "accuracy":  m.get("accuracy",  float("nan")),
                "precision": m.get("precision", float("nan")),
                "recall":    m.get("recall",    float("nan")),
                "f1":        m.get("f1",        float("nan")),
                "auc_roc":   m.get("auc_roc",   float("nan")),
            })
            rows.append(row)
    return rows


def _fairness_rows(fairness: dict, scenario_meta: dict) -> list:
    rows = []
    for sc, families in fairness.items():
        meta = scenario_meta.get(sc, {})
        for fam, fr in families.items():
            row = {"scenario": sc, "family": fam, "model": fr.model_name}
            row.update(meta)
            row.update({"group_" + k: v for k, v in fr.group.items()})
            row.update({"ind_" + k: v for k, v in fr.individual.items()})
            rows.append(row)
    return rows


def _nice_rows(nice_results: dict, scenario_meta: dict) -> list:
    rows = []
    for sc, families in nice_results.items():
        meta = scenario_meta.get(sc, {})
        for fam, nr in families.items():
            if not getattr(nr, "metrics", {}):
                continue
            row = {"scenario": sc, "family": fam, "model": nr.model_name}
            row.update(meta)
            yq = getattr(nr, "y_query", np.array([]))
            yc = getattr(nr, "y_cf", np.array([]))
            if len(yq) > 0 and len(yc) > 0:
                row["flip_rate"] = float((yq != yc).mean())
            row.update(nr.metrics)
            rows.append(row)
    return rows


def _attach_meta(df: pd.DataFrame, scenario_meta: dict) -> pd.DataFrame:
    if df.empty or "scenario" not in df.columns:
        return df
    meta_df = pd.DataFrame([dict(scenario=sc, **meta) for sc, meta in scenario_meta.items()])
    return df.merge(meta_df, on="scenario", how="left")


def _build_summary(perf_df, fair_df, nice_df, mia_df, aia_df) -> pd.DataFrame:
    """One row per (scenario, family) joining performance + fairness + NiCE
    quality + best-attacker MIA / AIA summary columns."""
    keys = ["scenario", "family"]
    if perf_df.empty:
        return pd.DataFrame()

    summary = perf_df.copy()

    if not fair_df.empty:
        fair_cols = keys + [c for c in fair_df.columns
                            if c.startswith("group_") or c.startswith("ind_")]
        summary = summary.merge(fair_df[fair_cols].drop_duplicates(subset=keys),
                                on=keys, how="left")

    if not nice_df.empty:
        nice_cols = [c for c in ["flip_rate", "proximity", "plausibility", "sparsity"]
                    if c in nice_df.columns]
        if nice_cols:
            summary = summary.merge(
                nice_df[keys + nice_cols].drop_duplicates(subset=keys),
                on=keys, how="left")

    for label, df in (("mia", mia_df), ("aia", aia_df)):
        if df is None or df.empty or "auc_roc" not in df.columns:
            continue
        valid = df.dropna(subset=["auc_roc"])
        if valid.empty:
            continue
        idx = valid.groupby(keys)["auc_roc"].idxmax()
        best = valid.loc[idx]
        rename = {"auc_roc": "{}_best_auc_roc".format(label)}
        cols = ["auc_roc"]
        if "attacker" in best.columns:
            rename["attacker"] = "{}_best_attacker".format(label)
            cols.append("attacker")
        adv_col = "mia_advantage" if "mia_advantage" in best.columns else None
        if adv_col:
            rename[adv_col] = "{}_best_advantage".format(label)
            cols.append(adv_col)
        priv_col = "privacy_gain" if "privacy_gain" in best.columns else None
        if priv_col:
            rename[priv_col] = "{}_best_privacy_gain".format(label)
            cols.append(priv_col)
        bal_col = "balanced_accuracy" if "balanced_accuracy" in best.columns else None
        if bal_col:
            rename[bal_col] = "{}_best_balanced_accuracy".format(label)
            cols.append(bal_col)
        base_col = "majority_baseline_accuracy" if "majority_baseline_accuracy" in best.columns else None
        if base_col:
            rename[base_col] = "{}_baseline_accuracy".format(label)
            cols.append(base_col)
        over_col = "accuracy_over_baseline" if "accuracy_over_baseline" in best.columns else None
        if over_col:
            rename[over_col] = "{}_best_accuracy_over_baseline".format(label)
            cols.append(over_col)
        sel = best[keys + cols].rename(columns=rename)
        summary = summary.merge(sel, on=keys, how="left")

    return summary


def _print_console_table(perf_rows: list, fairness: dict) -> None:
    if not perf_rows:
        return
    print("\n  {:<38} {:<22} {:>7} {:>7} {:>7} {:>7} {:>7}   {:>7} {:>7} {:>7}".format(
        "scenario", "family", "acc", "auc", "f1", "prec", "rec", "SPD", "EOD", "CF-fair"))
    print("  " + "-" * 130)
    for row in perf_rows:
        sc, fam = row["scenario"], row["family"]
        fr = fairness.get(sc, {}).get(fam)
        spd = fr.group.get("SPD", float("nan")) if fr else float("nan")
        eod = fr.group.get("EOD", float("nan")) if fr else float("nan")
        cf = fr.individual.get("cf_fairness", float("nan")) if fr else float("nan")
        print("  {:<38} {:<22} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f}   {:>+7.3f} {:>+7.3f} {:>7.3f}".format(
            sc[:38], fam[:22], row["accuracy"], row["auc_roc"], row["f1"],
            row["precision"], row["recall"], spd, eod, cf))


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_full_ablation(
    methods: list = None,
    families: list = None,
    ldp_epsilons: list = None,
    fl_constraints: list = None,
    nice_max_samples: int = None,
    include_cf_augmentation: bool = True,
    include_fairlearn: bool = True,
    include_ldp: bool = True,
    include_kusner_baselines: bool = True,
    method_combinations: list = None,
    ldp_corrected_methods: list = None,
    skip_dp_causal_graph: bool = False,
    dp_causal_graph_reps: int = 30,
    dp_causal_graph_epsilons: list = None,
    skip_nice: bool = False,
    skip_mia: bool = False,
    skip_aia: bool = True,
    skip_aia_nice_cf: bool = False,
    skip_dashboard: bool = False,
    skip_paper_export: bool = False,
    output_dir: str = None,
    verbose: bool = True,
) -> dict:
    """Run the complete privacy x fairness ablation study.

    method_combinations        : '+'-joined combinations of augmentation
                                  methods to train in Category 4b (e.g.
                                  ['update_labels+SCM']; see
                                  _augment_combined). None/[] skips 4b.
                                  _method_combinations(methods) gives the
                                  full grid. No privacy is applied to them.
    ldp_corrected_methods     : which of dp_causal_analysis.LDP_CORRECTED_METHODS
                                  ('SCM_ldp_corrected' / 'update_labels_ldp_corrected'
                                  / 'add_comparators_ldp_corrected') to add to
                                  Category 5a's method sweep (default: all;
                                  'SCM_ldp_corrected' is dropped automatically
                                  if the dataset has no SCMSpec, same as 'SCM').
                                  Dropped from Category 4 regardless -- each
                                  correction needs an LDP epsilon to correct
                                  for, so it has no meaning without privacy.
    skip_dp_causal_graph       : skip the separate classifier-free
                                  causal-graph comparison step entirely.
    dp_causal_graph_reps       : repetitions per epsilon for that step
                                  (independent RR draws; no model training,
                                  so this can be large -- default 30).
    dp_causal_graph_epsilons   : epsilon sweep for that step (default:
                                  dp_causal_analysis.DEFAULT_GRAPH_EPSILONS,
                                  a wider sweep than ldp_epsilons since this
                                  analysis is cheap).
    skip_dashboard             : skip auto-regenerating pipeline_outputs/dashboard.html
                                  at the end of the run (pipeline.dashboard.build_payload()
                                  + render_dashboard() over every full_ablation_*/results/
                                  ablation_summary.csv found, same as running
                                  `python -m pipeline.dashboard` by hand). A failure here
                                  is recorded in `failures` but never aborts the run --
                                  every CSV above is already saved by this point.

    Returns a dict with keys: data, scenarios, scenario_meta, fairness,
    nice_cf, mia, aia, perf_df, fairness_df, nice_df, mia_df, aia_df,
    summary, failures, manifest, dp_causal_graph_edges, dp_causal_graph_summary,
    dashboard_path (None if generation failed or was skipped).
    """
    methods        = methods        if methods        is not None else ALL_AUG_METHODS
    families       = families       if families       is not None else MODEL_FAMILIES
    ldp_epsilons   = ldp_epsilons   if ldp_epsilons   is not None else LDP_EPSILONS
    fl_constraints = fl_constraints if fl_constraints is not None else ALL_FL_CONSTRAINTS
    nice_max_samples = nice_max_samples if nice_max_samples is not None else NICE_MAX_SAMPLES
    ldp_corrected_methods = ldp_corrected_methods if ldp_corrected_methods is not None else list(dpc_mod.LDP_CORRECTED_METHODS)
    fl_family_tags = [_CANONICAL_TO_TAG[f] for f in families if f in _CANONICAL_TO_TAG]

    # 'SCM' / 'SCM_ldp_corrected' both need a per-dataset causal spec
    # (pipeline/datasets/<name>.py::SCMSpec) -- silently drop just those two
    # from an unfiltered default method list rather than crashing category
    # 4/5a partway through a long run; an explicit --methods SCM request
    # still surfaces the real NotImplementedError. 'update_labels'/
    # 'add_comparators' and their '_ldp_corrected' variants need no causal
    # spec (main_MM's regression needs no SCMSpec), so they are never
    # dropped here regardless of SCM_DEPENDENT_COLS.
    if "SCM" in methods and not _cfg.SCM_DEPENDENT_COLS:
        print("  [config] 'SCM' has no causal spec for dataset={!r} -- dropping it "
              "from the requested methods {}. Add pipeline.datasets.{}.SCMSpec to enable it.".format(
                  ACTIVE_DATASET, methods, ACTIVE_DATASET))
        methods = [m for m in methods if m != "SCM"]
    if "SCM_ldp_corrected" in ldp_corrected_methods and not _cfg.SCM_DEPENDENT_COLS:
        print("  [config] 'SCM_ldp_corrected' needs a causal spec for dataset={!r} -- dropping it "
              "(no SCMSpec). Add pipeline.datasets.{}.SCMSpec to enable it.".format(
                  ACTIVE_DATASET, ACTIVE_DATASET))
        ldp_corrected_methods = [m for m in ldp_corrected_methods if m != "SCM_ldp_corrected"]
    # Fair Add needs the dataset's SCMSpec; Unaware does not.
    kusner_methods = list(kusner_mod.KUSNER_BASELINES)
    if not _cfg.SCM_DEPENDENT_COLS:
        kusner_methods = [m for m in kusner_methods if m != "fair_add"]
    method_combinations = [_canonical_combo(c) for c in (method_combinations or [])]
    method_combinations = list(dict.fromkeys(method_combinations))  # de-duplicate, keep order
    if not _cfg.SCM_DEPENDENT_COLS and any("SCM" in c.split("+") for c in method_combinations):
        print("  [config] dropping combinations that use 'SCM' (no causal spec for dataset={!r}).".format(
            ACTIVE_DATASET))
        method_combinations = [c for c in method_combinations if "SCM" not in c.split("+")]

    global _RES_DIR, _NICE_DIR, _ABLATION_DIR
    if output_dir:
        _ABLATION_DIR = os.path.join(output_dir, "full_ablation_{}".format(ACTIVE_DATASET))
        _RES_DIR = os.path.join(_ABLATION_DIR, "results")
        _NICE_DIR = os.path.join(_ABLATION_DIR, "nice_cf_arrays")
    _mkdirs()

    timer = _StepTimer()
    t_start = time.perf_counter()
    failures = []

    all_scenarios: dict = {}   # scenario_name -> ScenarioModels
    training_sets: dict = {}   # scenario_name -> (X_train_used, y_train_used)
    scenario_meta: dict = {}   # scenario_name -> {ablation_category, ldp_epsilon, aug_method, fl_constraint}

    def _register(name, sm, X_used, y_used, meta):
        all_scenarios[name] = sm
        training_sets[name] = (X_used, y_used)
        scenario_meta[name] = meta

    _banner("FULL PRIVACY x FAIRNESS ABLATION STUDY -- dataset={}".format(ACTIVE_DATASET.upper()))
    print("  dataset         : {}".format(ACTIVE_DATASET))
    print("  methods         : {}".format(methods))
    print("  combinations    : {}  (Category 4b, no privacy)".format(method_combinations))
    print("  ldp_corrected_methods : {}  (Category 5a only)".format(ldp_corrected_methods))
    print("  families        : {}".format(families))
    print("  ldp_epsilons    : {}".format(ldp_epsilons))
    print("  fl_constraints  : {}".format(fl_constraints))
    print("  nice_max_samples: {}".format(nice_max_samples))
    print("  output_dir      : {}".format(_RES_DIR))

    # =======================================================================
    # STEP 0 -- Data
    # =======================================================================
    _banner("STEP 0 -- Data Loading, Splitting & SCM Fitting")
    with timer("step0_data"):
        first_method = methods[0] if (methods and include_cf_augmentation) else "add_comparators"
        data = dp.load_and_split(augmentation_method=first_method, output_dir=output_dir, verbose=verbose)
    prot_test = data.X_test[:, _PROTECTED_IDX]
    if verbose:
        print("  [data] train={:,}  test={:,}  features={}".format(
            len(data.X_train), len(data.X_test), FEATURE_COLS))

    # =======================================================================
    # STEP 0b -- DP causal-graph comparison (classifier-free)
    # =======================================================================
    # Independent of the scenario/family training matrix below: repeatedly
    # (over many independent RR draws, across a wide epsilon sweep -- NOT the
    # one draw any Category 5a scenario trains on) fits the uncorrected and
    # corrected SCM and compares their edge weights against the TRUE
    # (clean-data) SCM -- see dp_causal_analysis.py's module docstring for
    # why this is a separate statistical study, not a record of any specific
    # scenario's training data. No model training happens here, so this can
    # afford many more repetitions than the classifier-training categories
    # below.
    dp_causal_graph_edges = pd.DataFrame()
    dp_causal_graph_summary = pd.DataFrame()
    if not skip_dp_causal_graph:
        _banner("STEP 0b -- DP Causal-Graph Comparison (uncorrected vs. corrected SCM fit vs. true SCM)")
        with timer("dp_causal_graph"):
            try:
                dp_causal_graph_edges, dp_causal_graph_summary = dpc_mod.run_causal_graph_comparison(
                    data, epsilons=dp_causal_graph_epsilons, n_reps=dp_causal_graph_reps, verbose=verbose)
                dpc_mod.save_causal_graph_comparison(
                    dp_causal_graph_edges, dp_causal_graph_summary, _RES_DIR, verbose=verbose)
            except Exception as exc:
                failures.append({"scenario": "dp_causal_graph", "stage": "dp_causal_graph", "error": str(exc)})
                warnings.warn("[dp_causal_graph] comparison failed: {}\n{}".format(exc, traceback.format_exc()))
    else:
        print("\n[STEP 0b] Skipped (--skip-dp-causal-graph, or no SCMSpec for this dataset).")

    # =======================================================================
    # CATEGORY 1 -- no_privacy_no_fairness  (baseline)
    # =======================================================================
    _banner("CATEGORY 1/5 -- No Privacy, No Fairness (baseline)")
    with timer("category1_baseline"):
        try:
            sm = ml.train_scenario("baseline", data.X_train, data.y_train,
                                   data.X_test, data.y_test, families=families, verbose=verbose)
            _register("baseline", sm, data.X_train, data.y_train,
                      {"ablation_category": "no_privacy_no_fairness",
                       "ldp_epsilon": None, "aug_method": "none", "fl_constraint": "none"})
        except Exception as exc:
            failures.append({"scenario": "baseline", "stage": "train", "error": str(exc)})
            warnings.warn("[category1] baseline failed: {}\n{}".format(exc, traceback.format_exc()))

    # =======================================================================
    # CATEGORY 1b -- Kusner et al. (2017) predictor baselines (Unaware,
    # Fair Add). They change the model inputs, not the training rows, so they
    # train on X_train through a feature map (see kusner_baselines.py) and
    # are tagged like a Category-4 method so they sit next to the
    # augmentation methods in every table.
    # =======================================================================
    if include_kusner_baselines:
        _banner("CATEGORY 1b -- Kusner et al. Baselines ({})".format(", ".join(kusner_methods)))
        for kmethod in kusner_methods:
            sc_name = "kusner_{}".format(kmethod)
            with timer("category1b_{}".format(kmethod)):
                try:
                    fmap = kusner_mod.build_feature_map(kmethod, data.X_train)
                    sm = ml.train_scenario(sc_name, data.X_train, data.y_train, data.X_test, data.y_test,
                                           families=families, verbose=verbose, feature_map=fmap)
                    _register(sc_name, sm, data.X_train, data.y_train,
                              {"ablation_category": "fairness_causal_only",
                               "ldp_epsilon": None, "aug_method": kmethod, "fl_constraint": "none"})
                except Exception as exc:
                    failures.append({"scenario": sc_name, "stage": "train", "error": str(exc)})
                    warnings.warn("[category1b] {} failed: {}\n{}".format(kmethod, exc, traceback.format_exc()))

    # =======================================================================
    # CATEGORY 4 -- fairness_causal_only  (proposed CF / causal augmentation)
    # =======================================================================
    category4_augmented: dict = {}   # method -> (X_aug, y_aug), for the no-privacy causal-graph comparison below
    if include_cf_augmentation:
        _banner("CATEGORY 4/5 -- Fairness Only: Counterfactual / Causal Reasoning (proposed method)")
        for method in methods:
            sc_name = "augmented_{}".format(method)
            print("\n  [causal-CF] method={}".format(method))
            with timer("category4_{}".format(method)):
                try:
                    X_aug, y_aug, sample_weight = _augment(data.X_train, data.y_train, method, verbose=verbose)
                    category4_augmented[method] = (X_aug, y_aug)
                    sm = ml.train_scenario(sc_name, X_aug, y_aug, data.X_test, data.y_test,
                                           families=families, sample_weight=sample_weight, verbose=verbose)
                    _register(sc_name, sm, X_aug, y_aug,
                              {"ablation_category": "fairness_causal_only",
                               "ldp_epsilon": None, "aug_method": method, "fl_constraint": "none"})
                except Exception as exc:
                    failures.append({"scenario": sc_name, "stage": "train", "error": str(exc)})
                    warnings.warn("[category4] {} failed: {}\n{}".format(method, exc, traceback.format_exc()))
    else:
        print("\n[CATEGORY 4] Skipped (--no-cf-augmentation).")

    # =======================================================================
    # CATEGORY 4b -- fairness_causal_combined  (method ablation)
    # =======================================================================
    if include_cf_augmentation and method_combinations:
        _banner("CATEGORY 4b -- Method Ablation: Combined Causal Augmentation ({} combinations)".format(
            len(method_combinations)))
        for combo in method_combinations:
            sc_name = "augmented_{}".format(combo)
            print("\n  [combo] methods={}".format(combo))
            with timer("category4b_{}".format(combo)):
                try:
                    X_aug, y_aug, sample_weight = _augment_combined(
                        data.X_train, data.y_train, combo, verbose=verbose)
                    category4_augmented[combo] = (X_aug, y_aug)
                    sm = ml.train_scenario(sc_name, X_aug, y_aug, data.X_test, data.y_test,
                                           families=families, sample_weight=sample_weight, verbose=verbose)
                    _register(sc_name, sm, X_aug, y_aug,
                              {"ablation_category": "fairness_causal_combined",
                               "ldp_epsilon": None, "aug_method": combo, "fl_constraint": "none"})
                except Exception as exc:
                    failures.append({"scenario": sc_name, "stage": "train", "error": str(exc)})
                    warnings.warn("[category4b] {} failed: {}\n{}".format(combo, exc, traceback.format_exc()))

    # --- No-privacy causal-graph comparison: how much does each Category-4
    # augmentation method itself distort the protected -> dependent-column
    # relationship, with no privacy involved at all (see
    # dp_causal_analysis.py::run_augmentation_causal_graph_comparison). ---
    aug_causal_graph_edges = pd.DataFrame()
    aug_causal_graph_summary = pd.DataFrame()
    if category4_augmented and data.scm is not None:
        with timer("augmentation_causal_graph"):
            try:
                aug_causal_graph_edges, aug_causal_graph_summary = dpc_mod.run_augmentation_causal_graph_comparison(
                    data, category4_augmented, verbose=verbose)
                dpc_mod.save_augmentation_causal_graph_comparison(
                    aug_causal_graph_edges, aug_causal_graph_summary, _RES_DIR, verbose=verbose)
            except Exception as exc:
                failures.append({"scenario": "augmentation_causal_graph", "stage": "augmentation_causal_graph", "error": str(exc)})
                warnings.warn("[augmentation_causal_graph] comparison failed: {}\n{}".format(exc, traceback.format_exc()))

    # =======================================================================
    # CATEGORY 3 -- fairness_fairlearn_only
    # =======================================================================
    if include_fairlearn:
        _banner("CATEGORY 3/5 -- Fairness Only: FairLearn Constraints")
        with timer("category3_fairlearn"):
            try:
                fl_scenarios = _train_fairlearn_block(
                    data, tag_prefix="fairlearn", constraints=fl_constraints,
                    family_tags=fl_family_tags, verbose=verbose)
                for sc_name, sm in fl_scenarios.items():
                    ctag = sc_name.split("fairlearn_", 1)[1]
                    _register(sc_name, sm, data.X_train, data.y_train,
                              {"ablation_category": "fairness_fairlearn_only",
                               "ldp_epsilon": None, "aug_method": "none", "fl_constraint": ctag})
                if not fl_scenarios:
                    failures.append({"scenario": "fairlearn_*", "stage": "train",
                                     "error": "no fairlearn scenarios were produced"})
            except Exception as exc:
                failures.append({"scenario": "fairlearn_*", "stage": "train", "error": str(exc)})
                warnings.warn("[category3] fairlearn failed: {}\n{}".format(exc, traceback.format_exc()))
    else:
        print("\n[CATEGORY 3] Skipped (--no-fairlearn).")

    # =======================================================================
    # CATEGORY 2 & 5 -- privacy_only  +  privacy_plus_fairness
    # =======================================================================
    ldp_draws: dict = {}   # epsilon -> the LDP-perturbed X_train every Category 2/5 scenario at that epsilon trains on
    if include_ldp:
        _banner("CATEGORY 2/5 & 5/5 -- Privacy (LDP) Alone and Combined with Fairness")
        for eps in ldp_epsilons:
            eps_tag = "eps{}".format(eps)
            print("\n  [LDP] epsilon={}".format(eps))
            with timer("ldp_perturb_{}".format(eps_tag)):
                X_train_ldp = data.X_train.copy()
                perturbed, p_flip = ldp_mod.randomised_response(
                    X_train_ldp[:, _PROTECTED_IDX], epsilon=eps, random_state=RANDOM_STATE)
                X_train_ldp[:, _PROTECTED_IDX] = perturbed
                ldp_draws[eps] = X_train_ldp
                n_flip = int((perturbed != data.X_train[:, _PROTECTED_IDX]).sum())
                print("        p_flip={:.4f}  flipped={}/{}".format(p_flip, n_flip, len(perturbed)))

            # --- Category 2: LDP alone ---
            sc_name = "ldp_baseline_{}".format(eps_tag)
            with timer("category2_{}".format(eps_tag)):
                try:
                    sm = ml.train_scenario(sc_name, X_train_ldp, data.y_train,
                                           data.X_test, data.y_test, families=families, verbose=verbose)
                    _register(sc_name, sm, X_train_ldp, data.y_train,
                              {"ablation_category": "privacy_only",
                               "ldp_epsilon": eps, "aug_method": "none", "fl_constraint": "none"})
                except Exception as exc:
                    failures.append({"scenario": sc_name, "stage": "train", "error": str(exc)})
                    warnings.warn("[category2] {} failed: {}".format(sc_name, exc))

            # --- Category 5c: LDP + Kusner baselines (Fair Add naive and
            # with the LDP-corrected SCM; Unaware needs no SCM) ---
            if include_kusner_baselines:
                kusner_5c = list(kusner_methods) + (
                    kusner_mod.KUSNER_LDP_CORRECTED if "fair_add" in kusner_methods else [])
                for kmethod in kusner_5c:
                    sc_name = "ldp_kusner_{}_{}".format(kmethod, eps_tag)
                    with timer("category5c_{}_{}".format(kmethod, eps_tag)):
                        try:
                            fmap = kusner_mod.build_feature_map(kmethod, X_train_ldp, epsilon=eps)
                            sm = ml.train_scenario(sc_name, X_train_ldp, data.y_train, data.X_test, data.y_test,
                                                   families=families, verbose=verbose, feature_map=fmap)
                            _register(sc_name, sm, X_train_ldp, data.y_train,
                                      {"ablation_category": "privacy_plus_fairness",
                                       "ldp_epsilon": eps, "aug_method": kmethod, "fl_constraint": "none"})
                        except Exception as exc:
                            failures.append({"scenario": sc_name, "stage": "train", "error": str(exc)})
                            warnings.warn("[category5c] {} failed: {}".format(sc_name, exc))

            # --- Category 5a: LDP + causal-CF augmentation ---
            # methods_5a extends the Category-4 method list with the
            # LDP-only corrected variants (dp_causal_analysis
            # .LDP_CORRECTED_METHODS) -- these never appear in Category 4,
            # since fitting them without LDP applied would be meaningless
            # (there'd be nothing to correct for).
            if include_cf_augmentation:
                methods_5a = list(methods) + [m for m in ldp_corrected_methods if m not in methods]
                for method in methods_5a:
                    sc_name = "ldp_{}_{}".format(method, eps_tag)
                    with timer("category5a_{}_{}".format(method, eps_tag)):
                        try:
                            X_aug_ldp, y_aug_ldp, sw = _augment(
                                X_train_ldp, data.y_train, method, verbose=verbose, epsilon=eps)
                            sm = ml.train_scenario(sc_name, X_aug_ldp, y_aug_ldp, data.X_test, data.y_test,
                                                   families=families, sample_weight=sw, verbose=verbose)
                            _register(sc_name, sm, X_aug_ldp, y_aug_ldp,
                                      {"ablation_category": "privacy_plus_fairness",
                                       "ldp_epsilon": eps, "aug_method": method, "fl_constraint": "none"})
                        except Exception as exc:
                            failures.append({"scenario": sc_name, "stage": "train", "error": str(exc)})
                            warnings.warn("[category5a] {} failed: {}".format(sc_name, exc))

            # --- Category 5b: LDP + FairLearn ---
            if include_fairlearn:
                with timer("category5b_{}".format(eps_tag)):
                    try:
                        ldp_view = _ldp_view(data, X_train_ldp)
                        fl_ldp = _train_fairlearn_block(
                            ldp_view, tag_prefix="fairlearn_ldp_{}".format(eps_tag),
                            constraints=fl_constraints, family_tags=fl_family_tags, verbose=verbose)
                        for sc_name, sm in fl_ldp.items():
                            ctag = sc_name.split("_{}_".format(eps_tag), 1)[-1]
                            _register(sc_name, sm, X_train_ldp, data.y_train,
                                      {"ablation_category": "privacy_plus_fairness",
                                       "ldp_epsilon": eps, "aug_method": "none", "fl_constraint": ctag})
                        if not fl_ldp:
                            failures.append({"scenario": "fairlearn_ldp_{}_*".format(eps_tag),
                                             "stage": "train", "error": "no scenarios produced"})
                    except Exception as exc:
                        failures.append({"scenario": "fairlearn_ldp_{}_*".format(eps_tag),
                                         "stage": "train", "error": str(exc)})
                        warnings.warn("[category5b] eps={} failed: {}\n{}".format(
                            eps, exc, traceback.format_exc()))
    else:
        print("\n[CATEGORY 2 & 5] Skipped (--no-ldp).")

    n_scenarios = len(all_scenarios)
    n_models = sum(len(sm.results) for sm in all_scenarios.values())
    _banner("TRAINING COMPLETE -- {} scenarios / {} (scenario,family) models".format(
        n_scenarios, n_models))

    # =======================================================================
    # FAIRNESS  (every scenario x family, on the fixed original test set)
    # =======================================================================
    _banner("FAIRNESS ANALYSIS -- group, individual, counterfactual")
    fairness_results: dict = {}
    with timer("fairness"):
        for i, (sc_name, sm) in enumerate(all_scenarios.items(), 1):
            fairness_results[sc_name] = {}
            for fam, model_res in sm.results.items():
                try:
                    fairness_results[sc_name][fam] = fm.compute_fairness(
                        model_result=model_res, X_test=data.X_test, y_test=data.y_test,
                        protected_test=prot_test, verbose=verbose)
                except Exception as exc:
                    failures.append({"scenario": sc_name, "family": fam, "stage": "fairness", "error": str(exc)})
                    warnings.warn("[fairness] {}/{} failed: {}".format(sc_name, fam, exc))
            if verbose and i % 10 == 0:
                print("  [fairness] {}/{} scenarios done".format(i, n_scenarios))

    perf_rows = _perf_rows(all_scenarios, scenario_meta)
    if verbose:
        _print_console_table(perf_rows, fairness_results)

    perf_df = pd.DataFrame(perf_rows)
    fair_rows = _fairness_rows(fairness_results, scenario_meta)
    fair_df = pd.DataFrame(fair_rows)
    _save_csv(perf_df, "model_performance.csv", verbose)
    _save_csv(fair_df, "fairness_metrics.csv", verbose)

    # =======================================================================
    # ATTRIBUTE INFERENCE ATTACK
    # =======================================================================
    aia_df = pd.DataFrame()
    aia_results = {}
    if not skip_aia:
        _banner("ATTRIBUTE INFERENCE ATTACK -- {} scenarios x {} target families "
                "x 2 feature-set variants{}".format(
                    n_scenarios, len(families),
                    " x 2 query sources (real + NiCE CF)" if not skip_aia_nice_cf else ""))
        with timer("aia"):
            try:
                # Full ablation over both AIA axes in one call (target model
                # and, when enabled, its NiCE CF pool are each computed once
                # per scenario/family and reused across variants -- see
                # run_attribute_inference_analysis):
                #   1) feature-set:  attacker sees non-sensitive attributes +
                #      target outputs (x_ns+outputs) vs. target outputs only
                #      (outputs_only) -- include_x_ns=(True, False).
                #   2) query source: real held-out test rows only, vs. also
                #      evaluated on NiCE counterfactual queries generated
                #      against this AIA target model -- include_nice_cf.
                # nice_cf_results=None: the NiCE CFs generated above explain
                # the fairness-ablation models (trained WITH the protected
                # attribute), not the AIA's own no-sensitive target model, so
                # they aren't valid queries for it; generate_missing_cf=True
                # makes the AIA generate CFs against its own target model,
                # querying every row of the AIA validation set (never the AIA
                # test set, which stays reserved for scoring the attack).
                aia_results = aia_mod.run_attribute_inference_analysis(
                    training_sets=training_sets, families=families, data=data,
                    nice_cf_results=None, verbose=verbose,
                    include_x_ns=(True, False),
                    include_nice_cf=False, generate_missing_cf=False)
                aia_df = aia_mod.aia_results_to_dataframe(aia_results)
                aia_df = _attach_meta(aia_df, scenario_meta)
            except Exception as exc:
                failures.append({"scenario": "*", "stage": "aia", "error": str(exc)})
                warnings.warn("[aia] analysis failed: {}\n{}".format(exc, traceback.format_exc()))
        if not aia_df.empty:
            _save_csv(aia_df, "attribute_inference_attack.csv", verbose)
    else:
        print("\n[AIA] Skipped (off by default; pass --run-aia to enable).")

    # =======================================================================
    # Master summary
    # =======================================================================
    _banner("BUILDING ABLATION SUMMARY")
    #summary = _build_summary(perf_df, fair_df, nice_df, mia_df, aia_df)
    summary = _build_summary(perf_df, fair_df, pd.DataFrame(), pd.DataFrame(), aia_df)
    if not summary.empty:
        _save_csv(summary, "ablation_summary.csv", verbose)

    method_ablation = pd.DataFrame()
    if method_combinations:
        method_ablation = _build_method_ablation(summary)
        if not method_ablation.empty:
            _save_csv(method_ablation, "method_ablation.csv", verbose)

    # Paper tables: one row per trained model in every file (see pipeline/paper_export.py).
    paper_dir = os.path.join(_RES_DIR, "paper")
    if not skip_paper_export:
        _banner("PAPER TABLES -> {}".format(paper_dir))
        with timer("paper_export"):
            try:
                paper_mod.export_paper_tables(
                    paper_dir, all_scenarios, scenario_meta, fairness_results, data,
                    ldp_draws, dataset=ACTIVE_DATASET, seed=RANDOM_STATE, verbose=verbose)
            except Exception as exc:
                failures.append({"scenario": "*", "stage": "paper_export", "error": str(exc)})
                warnings.warn("[paper_export] failed: {}\n{}".format(exc, traceback.format_exc()))

    elapsed = time.perf_counter() - t_start
    manifest = {
        "config": {
            "methods": methods, "method_combinations": method_combinations,
            "families": families, "ldp_epsilons": ldp_epsilons,
            "fl_constraints": fl_constraints, "nice_max_samples": nice_max_samples,
            "include_cf_augmentation": include_cf_augmentation,
            "include_fairlearn": include_fairlearn, "include_ldp": include_ldp,
            "ldp_corrected_methods": ldp_corrected_methods,
            "skip_dp_causal_graph": skip_dp_causal_graph,
            "dp_causal_graph_reps": dp_causal_graph_reps,
            "dp_causal_graph_epsilons": dp_causal_graph_epsilons,
            "skip_nice": skip_nice, "skip_mia": skip_mia, "skip_aia": skip_aia,
            "skip_aia_nice_cf": skip_aia_nice_cf, "skip_dashboard": skip_dashboard,
            "skip_paper_export": skip_paper_export,
            "aia_ablation_variants": ["x_ns+outputs", "outputs_only"],
            "random_state": RANDOM_STATE, "dataset": DATASET_NAME,
        },
        "package_versions": _pkg_versions(),
        "n_scenarios": n_scenarios,
        "n_scenario_family_models": n_models,
        "step_timings_sec": {k: round(v, 2) for k, v in timer.times.items()},
        "elapsed_sec": round(elapsed, 2),
        "failures": failures,
        "output_dir": _RES_DIR,
    }
    manifest_path = os.path.join(_RES_DIR, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print("  [save] run_manifest.json -> {}".format(manifest_path))

    # =======================================================================
    # Dashboard -- auto-regenerated at the end of every run, over EVERY
    # full_ablation_*/results/ablation_summary.csv found under
    # pipeline_outputs/ (not just this run's dataset), exactly like manually
    # running `python -m pipeline.dashboard` afterwards. Never fails the run:
    # every CSV above is already saved by this point, so a dashboard bug is
    # recorded in `failures` and printed, not raised.
    # =======================================================================
    dashboard_path = None
    research_dashboard_path = None
    if not skip_dashboard:
        _banner("GENERATING DASHBOARD")
        try:
            dash_payload = dashboard_mod.build_payload()
            dashboard_path = os.path.join(_cfg.OUTPUT_DIR, "dashboard.html")
            dashboard_mod.render_dashboard(dash_payload, dashboard_path)
            print("  [save] dashboard.html -> {}  ({} records, {} run(s))".format(
                dashboard_path, len(dash_payload["records"]), len(dash_payload["runs"])))
        # SystemExit too: build_payload() raises it when no full_ablation*
        # run folder exists (always the case under run_multi_seed.py's
        # --output-dir), and it is not an Exception subclass.
        except (Exception, SystemExit) as exc:
            failures.append({"scenario": "*", "stage": "dashboard", "error": str(exc)})
            warnings.warn("[dashboard] generation failed: {}\n{}".format(exc, traceback.format_exc()))

        # Research-questions dashboard: reshapes the SAME payload above into
        # one panel per RQ1-RQ4 (see research_dashboard.py's module
        # docstring) instead of dashboard.py's general method-taxonomy
        # layout. Skipped together with --skip-dashboard since it depends on
        # dashboard.build_payload() having just run.
        try:
            research_payload = research_dashboard_mod.build_payload()
            research_dashboard_path = os.path.join(_cfg.OUTPUT_DIR, "research_dashboard.html")
            research_dashboard_mod.render_dashboard(research_payload, research_dashboard_path)
            print("  [save] research_dashboard.html -> {}  ({} run(s))".format(
                research_dashboard_path, len(research_payload["runs"])))
        except (Exception, SystemExit) as exc:
            failures.append({"scenario": "*", "stage": "research_dashboard", "error": str(exc)})
            warnings.warn("[research_dashboard] generation failed: {}\n{}".format(exc, traceback.format_exc()))
    else:
        print("\n[DASHBOARD] Skipped (--skip-dashboard).")

    _banner("FULL ABLATION COMPLETE  ({:.1f}s, {} failures)".format(elapsed, len(failures)))
    print("  Scenarios trained : {}".format(n_scenarios))
    print("  Models trained    : {}".format(n_models))
    print("  Output directory  : {}".format(_RES_DIR))
    if dashboard_path:
        print("  Dashboard         : {}".format(dashboard_path))
    if research_dashboard_path:
        print("  RQ dashboard      : {}".format(research_dashboard_path))
    if failures:
        print("\n  WARNING -- {} stage(s) failed (see run_manifest.json 'failures'):".format(len(failures)))
        for f in failures[:20]:
            print("    - {}".format(f))

    return {
        "data": data, "scenarios": all_scenarios, "scenario_meta": scenario_meta,
        "training_sets": training_sets,
        "fairness": fairness_results, # "nice_cf": nice_results,
        #"mia": mia_results, 
         "aia": aia_results,
        "perf_df": perf_df, "fairness_df": fair_df, 
         #"nice_df": nice_df,
         #"mia_df": mia_df,
          "aia_df": aia_df, "summary": summary, "method_ablation": method_ablation,
        "dp_causal_graph_edges": dp_causal_graph_edges,
        "dp_causal_graph_summary": dp_causal_graph_summary,
        "augmentation_causal_graph_edges": aug_causal_graph_edges,
        "augmentation_causal_graph_summary": aug_causal_graph_summary,
        "failures": failures, "manifest": manifest,
        "dashboard_path": dashboard_path,
        "research_dashboard_path": research_dashboard_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_FAM_ALIAS = {"lr": "logistic_regression", "rf": "random_forest", "xgb": "xgboost"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full privacy x fairness ablation study (dataset-pluggable -- "
                    "see pipeline/datasets/).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=ACTIVE_DATASET,
                        help="Registered dataset to run against (default: {!r}). "
                             "IMPORTANT: consumed before any other import happens "
                             "(see the sys.argv pre-scan at the top of this file), "
                             "so it works correctly even though argparse itself "
                             "runs after imports. Add new datasets in "
                             "pipeline/datasets/.".format(ACTIVE_DATASET))
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                        help="Random seed (default: config.RANDOM_STATE = {}). "
                             "Consumed before imports, same as --dataset -- see "
                             "the sys.argv pre-scan at the top of this file. "
                             "pipeline/run_multi_seed.py sets this per subprocess "
                             "to run the whole pipeline across N seeds.".format(RANDOM_STATE))
    parser.add_argument("--methods", nargs="+", default=None, choices=ALL_AUG_METHODS,
                        help="Counterfactual/causal augmentation methods (default: all 4).")
    parser.add_argument("--method-ablation", action="store_true",
                        help="Category 4b: also train every combination of >= 2 of --methods "
                             "(at most one add_comparators variant each), e.g. "
                             "update_labels+SCM, and write results/method_ablation.csv.")
    parser.add_argument("--combos", nargs="+", default=None, metavar="M1+M2[+M3]",
                        help="Category 4b with only these '+'-joined combinations, e.g. "
                             "--combos update_labels+SCM SCM+add_comparators_bidir. "
                             "Overrides the grid from --method-ablation.")
    parser.add_argument("--families", nargs="+", default=None,
                        choices=["logistic_regression", "random_forest", "xgboost", "lr", "rf", "xgb"],
                        help="Model families (default: all 3).")
    parser.add_argument("--ldp-epsilons", nargs="+", type=float, default=None,
                        help="LDP epsilon sweep (default: config.LDP_EPSILONS = {}).".format(LDP_EPSILONS))
    parser.add_argument("--fl-constraints", nargs="+", default=None, choices=ALL_FL_CONSTRAINTS,
                        help="FairLearn constraints to run (default: all 6).")
    parser.add_argument("--nice-max-samples", type=int, default=None,
                        help="Max test-set queries for NiCE CF generation (default: config.NICE_MAX_SAMPLES = {}).".format(NICE_MAX_SAMPLES))
    parser.add_argument("--ldp-corrected-methods", nargs="+", default=None, choices=dpc_mod.LDP_CORRECTED_METHODS,
                        help="Which LDP-corrected augmentation variants to add to Category 5a's "
                             "method sweep (default: all -- SCM_ldp_corrected, "
                             "update_labels_ldp_corrected, add_comparators_ldp_corrected; see "
                             "pipeline/dp_causal_analysis.py). Only meaningful combined with "
                             "LDP, so unlike --methods these never appear in Category 4.")
    parser.add_argument("--skip-dp-causal-graph", action="store_true",
                        help="Skip the separate classifier-free causal-graph comparison "
                             "(uncorrected vs. corrected fit vs. true fit edge weights, for "
                             "every augmentation method).")
    parser.add_argument("--dp-causal-graph-reps", type=int, default=30,
                        help="Repetitions per epsilon for the causal-graph comparison "
                             "(independent RR draws; no model training, so this can be "
                             "large -- default 30).")
    parser.add_argument("--dp-causal-graph-epsilons", nargs="+", type=float, default=None,
                        help="Epsilon sweep for the causal-graph comparison (default: "
                             "dp_causal_analysis.DEFAULT_GRAPH_EPSILONS = {}, wider than "
                             "--ldp-epsilons since this analysis is cheap).".format(
                                 dpc_mod.DEFAULT_GRAPH_EPSILONS))
    parser.add_argument("--no-cf-augmentation", action="store_true", help="Skip category 4 & 5a (causal-CF).")
    parser.add_argument("--no-fairlearn", action="store_true", help="Skip category 3 & 5b (FairLearn).")
    parser.add_argument("--no-ldp", action="store_true", help="Skip category 2 & 5 (LDP privacy).")
    parser.add_argument("--no-kusner-baselines", action="store_true",
                        help="Skip category 1b & 5c (Kusner et al. Unaware / Fair Add baselines).")
    parser.add_argument("--skip-nice", action="store_true", help="Skip NiCE CF generation (loads cache if present).")
    parser.add_argument("--skip-mia", action="store_true", help="Skip Membership Inference Attack.")
    parser.add_argument("--run-aia", action="store_true",
                        help="Run the Attribute Inference Attack (off by default).")
    parser.add_argument("--skip-aia", action="store_true",
                        help="Skip the Attribute Inference Attack (already the default; overrides --run-aia).")
    parser.add_argument("--skip-aia-nice-cf", action="store_true",
                        help="AIA always runs the full feature-set ablation "
                             "(x_ns+outputs vs outputs_only); by default it also evaluates "
                             "each variant on NiCE counterfactual queries generated against "
                             "its own target model (a 'nice_cf' input_type alongside the "
                             "real held-out test rows). Pass this to skip that CF axis and "
                             "only run the real-data evaluation. Requires NICEx; silently "
                             "skipped per scenario/family without it regardless of this flag.")
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="Skip auto-regenerating pipeline_outputs/dashboard.html at the "
                             "end of the run (equivalent to `python -m pipeline.dashboard`).")
    parser.add_argument("--skip-paper-export", action="store_true",
                        help="Skip writing results/paper/*.csv (one row per trained model; see "
                             "pipeline/paper_export.py).")
    parser.add_argument("--output-dir", default=None, help="Override OUTPUT_DIR.")
    parser.add_argument("--quick-test", action="store_true",
                        help="Fast smoke-test config: 1 method, 1 epsilon, 1 family, "
                             "2 fairlearn constraints, nice_max_samples=5.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")
    args = parser.parse_args()

    families_norm = [_FAM_ALIAS.get(f, f) for f in args.families] if args.families else None
    methods = args.methods
    ldp_epsilons = args.ldp_epsilons
    fl_constraints = args.fl_constraints
    nice_max_samples = args.nice_max_samples
    ldp_corrected_methods = args.ldp_corrected_methods
    dp_causal_graph_epsilons = args.dp_causal_graph_epsilons
    dp_causal_graph_reps = args.dp_causal_graph_reps
    if args.combos:
        try:
            method_combinations = [_canonical_combo(c) for c in args.combos]
        except ValueError as exc:
            parser.error(str(exc))
    elif args.method_ablation:
        method_combinations = _method_combinations(methods or ALL_AUG_METHODS)
    else:
        method_combinations = None

    if args.quick_test:
        methods = (methods or ALL_AUG_METHODS)[:1]
        ldp_epsilons = (ldp_epsilons or LDP_EPSILONS)[:1]
        families_norm = (families_norm or MODEL_FAMILIES)[:1]
        fl_constraints = (fl_constraints or ALL_FL_CONSTRAINTS)[:2]
        nice_max_samples = 5
        ldp_corrected_methods = (ldp_corrected_methods or dpc_mod.LDP_CORRECTED_METHODS)[:1]
        dp_causal_graph_epsilons = (dp_causal_graph_epsilons or dpc_mod.DEFAULT_GRAPH_EPSILONS)[:2]
        dp_causal_graph_reps = min(dp_causal_graph_reps, 3)
        method_combinations = method_combinations[:1] if method_combinations else None
        print("[quick-test] methods={} epsilons={} families={} fl_constraints={} nice_max_samples={} "
              "ldp_corrected_methods={} dp_causal_graph_epsilons={} dp_causal_graph_reps={} "
              "method_combinations={}".format(
            methods, ldp_epsilons, families_norm, fl_constraints, nice_max_samples,
            ldp_corrected_methods, dp_causal_graph_epsilons, dp_causal_graph_reps,
            method_combinations))

    run_full_ablation(
        methods=methods,
        families=families_norm,
        ldp_epsilons=ldp_epsilons,
        fl_constraints=fl_constraints,
        nice_max_samples=nice_max_samples,
        include_cf_augmentation=not args.no_cf_augmentation,
        include_kusner_baselines=not args.no_kusner_baselines,
        include_fairlearn=not args.no_fairlearn,
        include_ldp=not args.no_ldp,
        method_combinations=method_combinations,
        ldp_corrected_methods=ldp_corrected_methods,
        skip_dp_causal_graph=args.skip_dp_causal_graph,
        dp_causal_graph_reps=dp_causal_graph_reps,
        dp_causal_graph_epsilons=dp_causal_graph_epsilons,
        skip_nice=args.skip_nice,
        skip_mia=args.skip_mia,
        skip_aia=args.skip_aia or not args.run_aia,
        skip_aia_nice_cf=args.skip_aia_nice_cf,
        skip_dashboard=args.skip_dashboard,
        skip_paper_export=args.skip_paper_export,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


'''
    # =======================================================================
    # NiCE CF generation  (needed as the MIA synthetic-data proxy, and scored
    # in its own right for CF explanation quality)
    # =======================================================================
    nice_results: dict = {}
    if not skip_nice:
        _banner("NiCE COUNTERFACTUAL GENERATION -- {} scenarios, max_samples={}".format(
            n_scenarios, nice_max_samples))
        with timer("nice_cf"):
            try:
                nice_results = ncf.generate_nice_cfs_for_all_models(
                    scenarios=all_scenarios, data=data, max_samples=nice_max_samples, verbose=verbose)
                ncf.save_nice_cf_results(nice_results, _NICE_DIR)
            except Exception as exc:
                failures.append({"scenario": "*", "stage": "nice_cf", "error": str(exc)})
                warnings.warn("[nice_cf] generation failed: {}\n{}".format(exc, traceback.format_exc()))
    else:
        _banner("NiCE COUNTERFACTUAL GENERATION -- skipped, loading cache")
        nice_results = ncf.load_nice_cf_results(_NICE_DIR)

    nice_df = pd.DataFrame(_nice_rows(nice_results, scenario_meta))
    if not nice_df.empty:
        _save_csv(nice_df, "nice_cf_quality.csv", verbose)

    # =======================================================================
    # MEMBERSHIP INFERENCE ATTACK
    # =======================================================================
    mia_df = pd.DataFrame()
    mia_results = {}
    if not skip_mia:
        _banner("MEMBERSHIP INFERENCE ATTACK -- {} scenarios".format(n_scenarios))
        with timer("mia"):
            try:
                mia_results = mia_mod.run_mia_analysis(
                    scenarios=all_scenarios, data=data, nice_cf_results=nice_results, verbose=verbose)
                mia_df = mia_mod.mia_results_to_dataframe(mia_results)
                mia_df = _attach_meta(mia_df, scenario_meta)
            except Exception as exc:
                failures.append({"scenario": "*", "stage": "mia", "error": str(exc)})
                warnings.warn("[mia] analysis failed: {}\n{}".format(exc, traceback.format_exc()))
        if not mia_df.empty:
            _save_csv(mia_df, "membership_inference_attack.csv", verbose)
    else:
        print("\n[MIA] Skipped (--skip-mia).")
'''