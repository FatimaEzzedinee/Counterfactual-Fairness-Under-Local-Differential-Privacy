from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.config import (
    FEATURE_COLS, PROTECTED_COL, SCM_DEPENDENT_COLS, RANDOM_STATE,
    DATASET_NAME, datasets_info, sensitive_columns, female_values, white_values, LDP_EPSILONS
)
import pipeline.config as _cfg
import pipeline.cf_generation as cf_gen_mod
from pipeline.cf_generation import SCMParams, fit_scm, generate_training_race_cfs, main_MM
from pipeline.data_preparation import finalize_cf_augmentation
from pipeline.ldp import randomised_response
from pipeline.ldp_correction import fit_ols_naive, fit_ols_corrected_ldp, counterfactual_predictions

_PROTECTED_IDX = FEATURE_COLS.index(PROTECTED_COL)

# The four augmentation-method names this module adds to Category 5a -- one
# per augmentation method ('SCM', 'update_labels', 'add_comparators',
# 'add_comparators_bidir'), each corrected for the known RR flip probability
# via pipeline.ldp_correction.fit_ols_corrected_ldp. Only meaningful combined
# with LDP -- with no privacy applied there is no flip probability to
# correct for, so none of these appear in Category 4.
LDP_CORRECTED_METHODS = [
    "SCM_ldp_corrected", "update_labels_ldp_corrected",
    "add_comparators_ldp_corrected", "add_comparators_bidir_ldp_corrected",
]

# Which (no-suffix) augmentation method each corrected variant corrects, and
# which of main_MM's two return shapes it uses ('update_labels' relabels in
# place -- no CF rows to append; 'add_comparators'/'add_comparators_bidir'
# append CF rows, like 'SCM' does via generate_training_race_cfs).
# 'add_comparators_bidir_ldp_corrected' maps to the SAME base method as
# 'add_comparators_ldp_corrected' -- the difference between them is the
# AUG_COMPARATORS_BIDIRECTIONAL flag toggled in augment_with_ldp_correction
# below (mirroring full_ablation_pipeline.py::_resolve_aug_method's handling
# of the uncorrected 'add_comparators' / 'add_comparators_bidir' pair), NOT
# a different regression -- the do_white_*/do_male_*.csv correction itself
# is identical either way; only which CF rows get appended to the augmented
# training set afterwards changes.
_CORRECTED_TO_BASE = {
    "SCM_ldp_corrected": "SCM",
    "update_labels_ldp_corrected": "update_labels",
    "add_comparators_ldp_corrected": "add_comparators",
    "add_comparators_bidir_ldp_corrected": "add_comparators",
}

# Every augmentation method this module's causal-graph comparison can
# characterize under LDP. 'update_labels' and 'add_comparators' share
# EXACTLY the same design (main_MM's dep_col ~ female + nonwhite (+ age)
# regression backs both of their do_white_*/do_male_*.csv files) -- they are
# listed separately here only so each gets its own row in the resulting
# CSVs, matching how the ablation study already treats them as distinct
# methods; the underlying computation is intentionally duplicated once per
# name rather than special-cased, since both are cheap OLS fits.
AUG_METHOD_CAUSAL_GRAPH = ["SCM", "update_labels", "add_comparators"]

# Default epsilon sweep for run_causal_graph_comparison(). Deliberately
# wider than config.LDP_EPSILONS (the main ablation study's default [1, 5]):
# this analysis only ever re-fits OLS-equivalent group means (no model
# training), so a wide sweep is cheap, and the bias-vs-epsilon transition
# this module exists to characterize is only visible across a wide range.
# Very small epsilon (e.g. 0.1) pushes the RR flip probability so close to
# 0.5 that the correction's denominator (1 - 2p) approaches 0 and the
# corrected estimate becomes extremely high-variance -- a real
# bias-variance tradeoff, deliberately excluded from this default so it
# doesn't dominate downstream plots; pass it explicitly to study that regime.
DEFAULT_GRAPH_EPSILONS = LDP_EPSILONS


def fit_scm_corrected_ldp(X_train_ldp: np.ndarray, epsilon: float,
                          dependent_cols: Optional[List[str]] = None,
                          verbose: bool = False) -> SCMParams:
    """
    Fit the SCM's causal coefficients directly on the LDP-perturbed
    protected column, correcting for the KNOWN randomized-response flip
    probability p(epsilon) = 1 / (1 + e^epsilon) (pipeline.ldp's exact
    mechanism) via inversion of the RR transition matrix.

    Derivation. RR maps X_true -> X_obs via the symmetric transition matrix
        T = [[1-p, p], [p, 1-p]],   T[i, j] = P(X_obs=j | X_true=i)
    Since Y (each dependent column) depends only on X_true -- the RR flip
    bit is independent of Y given X_true -- for j in {0, 1}:
        E[Y * 1(X_obs=j)] = sum_i  m_true(i) * pi_true(i) * T[i, j]
                     v_j   =        sum_i        w_i        * T[i, j]
    i.e. v = T @ w (T symmetric, so T^T = T), where v is directly estimable
    from the observed (X_obs, Y) sample. Solve w = T^{-1} v, then recover
    m_true(i) = w_i / pi_true(i) using the standard RR-debiased estimate of
    the true group-1 proportion pi_true(1) = (P(X_obs=1) - p) / (1 - 2p).

    Returns an SCMParams with the same shape as pipeline.cf_generation
    .fit_scm()'s output (coef[col] = [intercept, slope], resid[col] computed
    against each row's OWN observed (LDP) protected value, max_val[col]),
    so it is a drop-in replacement anywhere an SCMParams is expected:
    generate_training_race_cfs, generate_test_race_cfs, _apply_intervention.

    Falls back to plain fit_scm() (the uncorrected fit) at epsilon ~ 0,
    where the RR transition matrix is (numerically) singular and the
    correction is not identified -- dividing by 1-2p~0 would just blow up
    rather than mean anything.
    """
    dependent_cols = dependent_cols if dependent_cols is not None else SCM_DEPENDENT_COLS
    if not dependent_cols:
        raise NotImplementedError(
            "fit_scm_corrected_ldp: no SCMSpec registered for the active dataset "
            "(pipeline.config.SCM_DEPENDENT_COLS is empty)."
        )
    X = np.asarray(X_train_ldp, dtype=float)
    N = len(X)
    protected = X[:, _PROTECTED_IDX].astype(int)
    p = 1.0 / (1.0 + np.exp(epsilon))  # matches pipeline.ldp.randomised_response's flip_prob exactly

    if abs(1.0 - 2.0 * p) < 1e-9:
        if verbose:
            print(f"[SCM-corrected] epsilon={epsilon:g} -> p={p:.4f} is numerically "
                  f"uninformative (1-2p ~ 0); falling back to the uncorrected fit.")
        return fit_scm(X, dependent_cols=dependent_cols, verbose=verbose)

    p_obs_1 = float(protected.mean())
    pi1 = (p_obs_1 - p) / (1.0 - 2.0 * p)
    pi1 = min(max(pi1, 1e-3), 1 - 1e-3)
    pi0 = 1.0 - pi1
    T = np.array([[1 - p, p], [p, 1 - p]])

    coef, resid, max_val = {}, {}, {}
    for col in dependent_cols:
        idx = FEATURE_COLS.index(col)
        y = X[:, idx]
        v = np.array([np.mean(y * (protected == 0)), np.mean(y * (protected == 1))])
        w = np.linalg.solve(T, v)
        m0, m1 = float(w[0] / pi0), float(w[1] / pi1)
        b0, b1 = m0, m1 - m0
        coef[col] = np.array([b0, b1])
        own_mean = np.where(protected == 1, m1, m0)
        resid[col] = y - own_mean
        max_val[col] = float(y.max())
        if verbose:
            print(f"[SCM-corrected] {col:<20} = {b0:.4f} + {b1:.4f}*protected   "
                  f"(eps={epsilon:g}, p_flip={p:.4f})")
    return SCMParams(dependent_cols=list(dependent_cols), coef=coef, resid=resid,
                     max_val=max_val, n_train=N)


def augment_with_ldp_correction(X_train_ldp: np.ndarray, y_train: np.ndarray, method: str,
                                epsilon: float, dataset_tag: str = None, verbose: bool = True):
    """
    Build (X_aug, y_aug, sample_weight) for any method in LDP_CORRECTED_METHODS
    ('SCM_ldp_corrected' / 'update_labels_ldp_corrected' /
    'add_comparators_ldp_corrected' / 'add_comparators_bidir_ldp_corrected'),
    given the already-LDP-perturbed training matrix X_train_ldp (protected
    column already replaced by its RR-perturbed value -- same input
    full_ablation_pipeline.py's Category 5a already builds for the
    uncorrected 'SCM'/'update_labels'/'add_comparators'/'add_comparators_bidir'
    methods). Both fitting AND the per-row flip-direction decision use only
    this privatized column -- one single draw, matching how every other
    Category 5a method is evaluated (this is NOT the repeated-draws
    statistical study; see module docstring / run_causal_graph_comparison).

    'SCM_ldp_corrected' corrects fit_scm's single-covariate regression
    (fit_scm_corrected_ldp, the original group-mean-based correction).
    'update_labels_ldp_corrected' / 'add_comparators_ldp_corrected' /
    'add_comparators_bidir_ldp_corrected' correct main_MM's multi-covariate
    do_white_*/do_male_*.csv regression instead (cf_generation.py::main_MM's
    `epsilon` parameter, which routes through
    pipeline.ldp_correction.fit_ols_corrected_ldp) -- the SAME correction
    algorithm, just applied to whichever regression that augmentation method
    actually depends on. The bidir/non-bidir pair correct the EXACT SAME
    regression (the AUG_COMPARATORS_BIDIRECTIONAL flag only changes which CF
    rows main_MM appends afterwards, toggled here exactly like
    full_ablation_pipeline.py::_resolve_aug_method does for the uncorrected
    pair) -- not two different corrections.
    """
    if method not in LDP_CORRECTED_METHODS:
        raise ValueError(f"augment_with_ldp_correction: unknown method {method!r}, "
                         f"expected one of {LDP_CORRECTED_METHODS}")
    dataset_tag = dataset_tag if dataset_tag is not None else DATASET_NAME.lower()

    if method == "SCM_ldp_corrected":
        scm_fit = fit_scm_corrected_ldp(X_train_ldp, epsilon, verbose=verbose)
        X_cf, y_cf, src_idx = generate_training_race_cfs(X_train_ldp, y_train, scm_fit, verbose=verbose,
                                                         return_src_idx=True)
        return finalize_cf_augmentation(X_train_ldp, y_train, X_cf, y_cf, src_idx=src_idx, verbose=verbose)

    base_method = _CORRECTED_TO_BASE[method]
    if base_method == "update_labels":
        # Relabels in place -- no synthetic rows, no sample_weight, same as
        # augment_data()'s plain 'update_labels' branch.
        X_aug, y_aug = main_MM(X_train_ldp, y_train, "update_labels", dataset_tag, epsilon=epsilon)
        return X_aug, y_aug, None

    bidir = (method == "add_comparators_bidir_ldp_corrected")
    saved = _cfg.AUG_COMPARATORS_BIDIRECTIONAL
    try:
        _cfg.AUG_COMPARATORS_BIDIRECTIONAL = bidir
        cf_gen_mod.AUG_COMPARATORS_BIDIRECTIONAL = bidir
        X_cf, y_cf, src_idx = main_MM(X_train_ldp, y_train, "add_comparators", dataset_tag,
                                      epsilon=epsilon, return_src_idx=True)
    finally:
        _cfg.AUG_COMPARATORS_BIDIRECTIONAL = saved
        cf_gen_mod.AUG_COMPARATORS_BIDIRECTIONAL = saved
    return finalize_cf_augmentation(X_train_ldp, y_train, X_cf, y_cf, src_idx=src_idx, verbose=verbose)


def compare_scm_to_true(scm_estimated: SCMParams, scm_true: SCMParams) -> Dict[str, float]:
    """Whole-graph comparison between one fitted SCM and the TRUE
    (clean-data) SCM: treats the vector of per-dependent-column slopes as
    the causal graph's edge-weight parameterization (the graph's topology
    -- protected attribute -> each dependent column -- never changes; only
    the edge weights can be distorted by augmentation or privacy).

      l2_distance         Euclidean distance between edge-weight vectors.
                          0 = identical graph.
      cosine_similarity   cosine of the angle between the two vectors --
                          whether the RELATIVE pattern of which columns are
                          most/least affected is preserved independent of
                          any uniform rescaling. 1 = same pattern.
      sign_flip_rate      fraction of edges whose direction is reversed --
                          worse than being merely biased in magnitude,
                          since a do-intervention along a sign-flipped edge
                          moves outcomes the WRONG way.
      mean_relative_bias  average |estimate - true| / |true| across edges.
      mean_coef_error     average |estimate - true| across edges, in the
                          dependent columns' own units (unlike
                          mean_relative_bias this is not normalised by
                          |true|, so it is comparable across epsilon even
                          for edges whose true slope is near zero).

    Used by both run_causal_graph_comparison() below (privacy context) and
    pipeline.cf_generation's no-privacy augmentation-method comparison.
    """
    dep_cols = scm_true.dependent_cols
    true_vec = np.array([scm_true.coef[c][1] for c in dep_cols])
    est_vec = np.array([scm_estimated.coef[c][1] for c in dep_cols])

    l2 = float(np.linalg.norm(est_vec - true_vec))
    denom = np.linalg.norm(est_vec) * np.linalg.norm(true_vec)
    cosine = float(np.dot(est_vec, true_vec) / denom) if denom > 1e-12 else float("nan")
    sign_flip = float(np.mean(np.sign(est_vec) != np.sign(true_vec)))
    rel_bias = float(np.mean(np.abs(est_vec - true_vec) / np.abs(true_vec)))
    coef_error = float(np.mean(np.abs(est_vec - true_vec)))
    return dict(l2_distance=l2, cosine_similarity=cosine, sign_flip_rate=sign_flip,
                mean_relative_bias=rel_bias, mean_coef_error=coef_error)


def _counterfactual_predictions(scm: SCMParams, protected_true: np.ndarray) -> Dict[str, np.ndarray]:
    """do(flip) counterfactual prediction of every dependent column, for
    every row, under a GIVEN scm's own coefficients and its own residuals
    (abduction already baked into scm.resid at fit time -- see fit_scm /
    fit_scm_corrected_ldp, both of which preserve the input array's row
    order, so scm.resid[col][i] always corresponds to row i of whatever
    array the scm was fit on).

    The intervention direction is always "flip the row's TRUE protected
    value" (protected_true), regardless of which protected value (true or
    LDP-observed) the scm itself was fit against -- this holds the actual
    fairness intervention fixed so two different SCMs' predictions for the
    SAME individual are directly comparable (see counterfactual_prediction_mae).
    """
    flipped = 1.0 - np.asarray(protected_true, dtype=float)
    out = {}
    for col in scm.dependent_cols:
        b0, b1 = scm.coef[col]
        pred = b0 + b1 * flipped + scm.resid[col]
        out[col] = np.clip(np.round(pred), 0.0, scm.max_val[col])
    return out


def counterfactual_prediction_mae(scm_estimated: SCMParams, scm_true: SCMParams,
                                  protected_true: np.ndarray) -> Dict[str, float]:
    """Downstream-effect metric: mean absolute difference between the
    do(flip) counterfactual dependent-column values scm_estimated would
    generate vs. what the TRUE SCM would generate, for the SAME rows and
    the SAME intervention (see _counterfactual_predictions). This is the
    concrete "so what" behind l2_distance/cosine_similarity/sign_flip_rate:
    it measures how much a fairness pipeline's actual generated
    counterfactuals would differ from the ones it would have generated on
    non-private data, purely because of SCM parameter error -- independent
    of whether any individual's own observed protected bit was flipped by
    LDP.

    Requires scm_estimated.resid[col] and scm_true.resid[col] to be indexed
    by the SAME row order as protected_true (true for any SCMParams
    produced by fit_scm/fit_scm_corrected_ldp on an array that is itself a
    row-for-row copy of the array protected_true came from -- e.g.
    X_train_ldp = data.X_train.copy() with only the protected column
    perturbed, exactly how run_causal_graph_comparison builds it).

    Returns {dep_col: mae, ..., "overall": mean-over-dep_cols}.
    """
    cf_true = _counterfactual_predictions(scm_true, protected_true)
    cf_est = _counterfactual_predictions(scm_estimated, protected_true)
    out = {}
    for col in scm_true.dependent_cols:
        out[col] = float(np.mean(np.abs(cf_true[col] - cf_est[col])))
    out["overall"] = float(np.mean([out[c] for c in scm_true.dependent_cols]))
    return out


def _main_mm_design(X: np.ndarray) -> Tuple[np.ndarray, int]:
    """Build main_MM's do_white_*/do_male_*.csv design -- [female, nonwhite
    (+ age)] -- from a raw feature matrix X in FEATURE_COLS order. Returns
    (X_design, protected_idx): protected_idx is whichever design column
    actually corresponds to config.PROTECTED_COL (the only one that ever
    gets LDP-perturbed; the other sensitive column and age are always
    treated as measured without error, matching the rest of the pipeline).

    This is the ONLY place 'update_labels'/'add_comparators' causal-graph
    analysis differs from 'SCM' below -- everything downstream (the fit,
    the correction, the counterfactual step) is the SAME shared code for
    all three methods; see pipeline/ldp_correction.py.
    """
    dataset_tag = DATASET_NAME.lower()
    sex_col, race_col = sensitive_columns[dataset_tag]
    sex_idx, race_idx = FEATURE_COLS.index(sex_col), FEATURE_COLS.index(race_col)
    female = (X[:, sex_idx] == female_values[dataset_tag]).astype(float)
    nonwhite = (X[:, race_idx] != white_values[dataset_tag]).astype(float)
    cols, names = [female, nonwhite], ["female", "nonwhite"]
    if "age" in FEATURE_COLS:
        cols.append(X[:, FEATURE_COLS.index("age")].astype(float))
        names.append("age")
    X_design = np.column_stack(cols)
    protected_name = "female" if PROTECTED_COL == sex_col else "nonwhite"
    return X_design, names.index(protected_name)


def run_causal_graph_comparison(data, epsilons: Optional[List[float]] = None,
                                n_reps: int = 30, verbose: bool = True,
                                aug_methods: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classifier-free, repeated-sampling causal-graph comparison of the
    uncorrected vs. corrected fit against the TRUE (clean-data) fit, across
    an epsilon sweep, for EVERY augmentation method in `aug_methods`
    (default: AUG_METHOD_CAUSAL_GRAPH = ['SCM', 'update_labels',
    'add_comparators'] -- 'SCM' dropped automatically if the active dataset
    has no SCMSpec). 'update_labels' and 'add_comparators' share exactly the
    same regression (main_MM's do_white_*/do_male_*.csv step) and so are
    computed independently but come back numerically identical to each
    other -- that is a correct property of the underlying method, not a bug
    (see experiments/unified_ldp_correction.py's unification check).

    IMPORTANT: for each (aug_method, epsilon) this draws n_reps INDEPENDENT
    randomized-response perturbations -- a fresh draw every repetition --
    purely to characterize each estimator's Monte-Carlo sampling
    distribution under repeated local-DP releases of the same underlying
    population. This is a statistical study, not a record of what any
    specific ablation scenario trained on: the "uncorrected" series here
    describes "what the naive fit on a freshly LDP-perturbed protected
    column looks like on average, and how variable it is, across many
    independent releases at this epsilon" -- NOT the one particular
    X_train_ldp that, say, scenario "ldp_SCM_eps1" in ablation_summary.csv
    happened to train its classifier on (that scenario used exactly one
    draw, seeded from config.RANDOM_STATE, like every other Category 5a
    method).

    Returns (edge_df, graph_summary_df):
      edge_df           one row per (aug_method, fit, epsilon, rep, dep_col):
                        true_slope, est_slope, coef_error (=|est-true|, RQ2/3),
                        cf_mae (downstream do-intervention counterfactual
                        error vs. the true fit, RQ4's "downstream effect").
                        fit is "uncorrected" or "corrected".
      graph_summary_df  one row per (aug_method, fit, epsilon), aggregated
                        across reps: l2_distance, cosine_similarity,
                        sign_flip_rate, mean_relative_bias, mean_coef_error
                        (means of the per-rep whole-graph comparison,
                        answers RQ2 for "uncorrected" and RQ3's recovery
                        quality for "corrected"), counterfactual_mae (mean of
                        the per-rep, per-column cf_mae -- RQ4), plus
                        detection_rate -- the fraction of dependent columns
                        whose estimated effect remains statistically
                        distinguishable from zero at the 95% level using
                        z = mean(est_slope across reps) / std(est_slope
                        across reps) -- the standard way to evaluate an
                        estimator's power via Monte Carlo simulation, and
                        RQ3's direct answer ("what remains recoverable").
    """
    epsilons = epsilons if epsilons is not None else DEFAULT_GRAPH_EPSILONS
    aug_methods = aug_methods if aug_methods is not None else AUG_METHOD_CAUSAL_GRAPH
    if data.scm is None:
        aug_methods = [m for m in aug_methods if m != "SCM"]
    if not aug_methods:
        raise ValueError("run_causal_graph_comparison: no augmentation methods available to "
                         "compare (no SCMSpec for this dataset, and 'SCM' was the only one requested).")

    rng = np.random.default_rng(RANDOM_STATE)
    edge_rows: List[Dict] = []

    for aug_method in aug_methods:
        if aug_method == "SCM":
            dep_cols = data.scm.dependent_cols
            protected_true = data.X_train[:, _PROTECTED_IDX].astype(int)
            true_slopes = {c: float(data.scm.coef[c][1]) for c in dep_cols}
        else:
            X_design_true, protected_idx = _main_mm_design(data.X_train)
            dep_cols = list(datasets_info)
            protected_true = X_design_true[:, protected_idx].astype(int)
            true_betas = {c: fit_ols_naive(X_design_true, data.X_train[:, FEATURE_COLS.index(c)].astype(float))
                         for c in dep_cols}
            true_slopes = {c: float(true_betas[c][1 + protected_idx]) for c in dep_cols}

        for eps in epsilons:
            for rep in range(n_reps):
                seed_i = int(rng.integers(0, 2**31 - 1))
                protected_ldp, _ = randomised_response(protected_true, epsilon=float(eps), random_state=seed_i)

                if aug_method == "SCM":
                    X_train_ldp = data.X_train.copy()
                    X_train_ldp[:, _PROTECTED_IDX] = protected_ldp
                    fit_uncorrected = fit_scm(X_train_ldp, dependent_cols=dep_cols, verbose=False)
                    fit_corrected = fit_scm_corrected_ldp(X_train_ldp, float(eps), dependent_cols=dep_cols, verbose=False)
                    for fit_name, scm_fit in [("uncorrected", fit_uncorrected), ("corrected", fit_corrected)]:
                        cf_mae = counterfactual_prediction_mae(scm_fit, data.scm, protected_true)
                        for col in dep_cols:
                            est_slope = float(scm_fit.coef[col][1])
                            edge_rows.append(dict(
                                aug_method=aug_method, fit=fit_name, epsilon=float(eps), rep=rep, dep_col=col,
                                true_slope=true_slopes[col], est_slope=est_slope,
                                coef_error=abs(est_slope - true_slopes[col]), cf_mae=cf_mae[col],
                            ))
                else:
                    X_obs = X_design_true.copy()
                    X_obs[:, protected_idx] = protected_ldp
                    for col in dep_cols:
                        y = data.X_train[:, FEATURE_COLS.index(col)].astype(float)
                        beta_true = true_betas[col]
                        beta_naive = fit_ols_naive(X_obs, y)
                        beta_corr = fit_ols_corrected_ldp(X_obs, y, protected_idx, float(eps))
                        # target=0.0: main_MM's do(protected := privileged
                        # value)'s fixed-value convention (do_white/do_male
                        # always zero the respective column) -- same
                        # convention regardless of privacy, so the true and
                        # LDP fits are compared under the SAME intervention.
                        cf_true = counterfactual_predictions(X_design_true, y, beta_true, protected_idx, 0.0)
                        for fit_name, beta in [("uncorrected", beta_naive), ("corrected", beta_corr)]:
                            cf_est = counterfactual_predictions(X_obs, y, beta, protected_idx, 0.0)
                            cf_mae_val = float(np.mean(np.abs(cf_est - cf_true)))
                            est_slope = float(beta[1 + protected_idx])
                            edge_rows.append(dict(
                                aug_method=aug_method, fit=fit_name, epsilon=float(eps), rep=rep, dep_col=col,
                                true_slope=true_slopes[col], est_slope=est_slope,
                                coef_error=abs(est_slope - true_slopes[col]), cf_mae=cf_mae_val,
                            ))
            if verbose:
                print(f"[dp_causal_analysis] causal-graph comparison: aug_method={aug_method} "
                     f"epsilon={eps:g} done ({n_reps} reps)")

    edge_df = pd.DataFrame(edge_rows)

    summary_rows: List[Dict] = []
    for (aug_method, fit_name, eps), sub_me in edge_df.groupby(["aug_method", "fit", "epsilon"]):
        dep_cols_here = sub_me["dep_col"].unique()
        true_vec, est_means, est_stds = [], [], []
        for col in dep_cols_here:
            sub = sub_me[sub_me.dep_col == col]
            true_vec.append(float(sub["true_slope"].iloc[0]))
            est_means.append(sub["est_slope"].mean())
            est_stds.append(sub["est_slope"].std(ddof=1))
        true_vec = np.array(true_vec)
        est_means = np.array(est_means)
        est_stds = np.array(est_stds)

        l2 = float(np.linalg.norm(est_means - true_vec))
        denom = np.linalg.norm(est_means) * np.linalg.norm(true_vec)
        cosine = float(np.dot(est_means, true_vec) / denom) if denom > 1e-12 else float("nan")
        sign_flip = float(np.mean(np.sign(est_means) != np.sign(true_vec)))
        rel_bias = float(np.mean(np.abs(est_means - true_vec) / np.abs(true_vec)))
        coef_error = float(np.mean(np.abs(est_means - true_vec)))
        # Average over dep_cols WITHIN each rep first, then take mean/std
        # ACROSS reps -- keeps "how much does this vary across independent
        # LDP releases" (the std) from being contaminated by cross-feature
        # variation, which averaging over the flat (rep, dep_col) pool would do.
        per_rep_mae = sub_me.groupby("rep")["cf_mae"].mean()
        cf_mae = float(per_rep_mae.mean())
        cf_mae_std = float(per_rep_mae.std(ddof=1)) if len(per_rep_mae) > 1 else 0.0
        z = np.divide(est_means, est_stds, out=np.full_like(est_means, np.inf), where=est_stds > 1e-12)
        detected = float(np.mean(np.abs(z) > 1.96))

        summary_rows.append(dict(
            aug_method=aug_method, fit=fit_name, epsilon=eps, l2_distance=l2, cosine_similarity=cosine,
            sign_flip_rate=sign_flip, detection_rate=detected, mean_relative_bias=rel_bias,
            mean_coef_error=coef_error, counterfactual_mae=cf_mae, counterfactual_mae_std=cf_mae_std,
        ))

    graph_summary_df = pd.DataFrame(summary_rows)
    return edge_df, graph_summary_df


def run_augmentation_causal_graph_comparison(
    data, augmented_by_method: Dict[str, Tuple[np.ndarray, np.ndarray]], verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    NO-PRIVACY causal-graph comparison across the fairness-only augmentation
    methods (Category 4: 'SCM', 'update_labels', 'add_comparators',
    'add_comparators_bidir'). For each method, refit the SCM's
    `dep_col ~ 1 + protected` relationship on that method's resulting
    (X_aug, y_aug) and compare it to the TRUE SCM (data.scm, fit on the
    untouched clean X_train). No LDP, no correction, no repeated sampling
    anywhere in this function -- everything here is deterministic given the
    fixed training data, so there is exactly one point per method (no error
    bars, no detection_rate -- those only make sense across independent RR
    draws, which is what run_causal_graph_comparison() is for).

    This answers a different question than the privacy analysis: not "does
    privacy distort the causal graph," but "does this fairness-augmentation
    method itself distort it, with no privacy involved at all." Note
    'update_labels' only ever edits the target label y, never feature
    values -- so its X_aug is identical to X_train and its refit SCM is
    expected to come back IDENTICAL to the true one (l2_distance=0 exactly);
    that is a correct null result, not a bug.

    augmented_by_method: {method_name: (X_aug, y_aug)} -- pass the exact
    arrays full_ablation_pipeline.py's Category 4 loop already computed via
    _augment(data.X_train, data.y_train, method), so the augmentation is
    computed once and reused rather than duplicated here.

    Returns (edge_df, graph_summary_df):
      edge_df           one row per (method, dep_col): true_slope, est_slope,
                        coef_error (=|est-true|).
      graph_summary_df  one row per method: l2_distance, cosine_similarity,
                        sign_flip_rate, mean_relative_bias, mean_coef_error.
    """
    if data.scm is None:
        raise ValueError("run_augmentation_causal_graph_comparison requires a dataset with an SCMSpec.")
    dep_cols = data.scm.dependent_cols
    true_slopes = {c: float(data.scm.coef[c][1]) for c in dep_cols}

    edge_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    for method, (X_aug, y_aug) in augmented_by_method.items():
        scm_fit = fit_scm(X_aug, dependent_cols=dep_cols, verbose=False)
        for col in dep_cols:
            est_slope = float(scm_fit.coef[col][1])
            edge_rows.append(dict(
                method=method, dep_col=col,
                true_slope=true_slopes[col], est_slope=est_slope,
                coef_error=abs(est_slope - true_slopes[col]),
            ))
        metrics = compare_scm_to_true(scm_fit, data.scm)
        summary_rows.append(dict(method=method, **metrics))
        if verbose:
            print(f"[dp_causal_analysis] no-privacy causal-graph refit: method={method}  "
                  f"l2_distance={metrics['l2_distance']:.4f}  cosine={metrics['cosine_similarity']:.4f}")

    return pd.DataFrame(edge_rows), pd.DataFrame(summary_rows)


def save_augmentation_causal_graph_comparison(edge_df: pd.DataFrame, summary_df: pd.DataFrame,
                                              results_dir: str, verbose: bool = True) -> None:
    """Save the no-privacy augmentation-method causal-graph comparison
    (see run_augmentation_causal_graph_comparison) as CSVs under
    results_dir, alongside every other ablation-study CSV for that run."""
    os.makedirs(results_dir, exist_ok=True)
    edge_path = os.path.join(results_dir, "augmentation_causal_graph_edges.csv")
    summary_path = os.path.join(results_dir, "augmentation_causal_graph_summary.csv")
    edge_df.to_csv(edge_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    if verbose:
        print(f"  [save] augmentation_causal_graph_edges.csv -> {edge_path}  ({len(edge_df):,} rows)")
        print(f"  [save] augmentation_causal_graph_summary.csv -> {summary_path}  ({len(summary_df):,} rows)")


def save_causal_graph_comparison(edge_df: pd.DataFrame, graph_summary_df: pd.DataFrame,
                                 results_dir: str, verbose: bool = True) -> None:
    """Save the raw per-repetition edge-level values and the aggregated
    whole-graph summary as CSVs under results_dir (the caller's results
    directory -- e.g. full_ablation_pipeline.py's _RES_DIR -- so these land
    alongside every other ablation-study CSV for that run)."""
    os.makedirs(results_dir, exist_ok=True)
    edge_path = os.path.join(results_dir, "dp_causal_graph_edges.csv")
    summary_path = os.path.join(results_dir, "dp_causal_graph_summary.csv")
    edge_df.to_csv(edge_path, index=False)
    graph_summary_df.to_csv(summary_path, index=False)
    if verbose:
        print(f"  [save] dp_causal_graph_edges.csv -> {edge_path}  ({len(edge_df):,} rows)")
        print(f"  [save] dp_causal_graph_summary.csv -> {summary_path}  ({len(graph_summary_df):,} rows)")
