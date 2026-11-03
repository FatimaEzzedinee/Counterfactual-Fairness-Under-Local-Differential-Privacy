"""
Called at the end of full_ablation_pipeline.run_full_ablation(); writes to
pipeline_outputs/full_ablation_<dataset>/results/paper/:

    models.csv                       one row per trained (scenario, family): what it is
    performance.csv                  accuracy, precision, recall, f1, auc_roc
    fairness_group.csv               SPD, DI, EOD, AOD, EqOdds, PP + per-group rates
    fairness_individual.csv          consistency (kNN), Theil index
    counterfactual_fairness.csv      flip-only and SCM (mutatis-mutandis) CF metrics
    all_models.csv                   everything above joined, one row per model
    ldp_correction_params.csv        per epsilon: p_flip, lambda, base rates
    ldp_correction_coefficients.csv  per epsilon x regression x dep_col x term:
                                     true / naive / corrected coefficient
    README.md                        column dictionary

Every model table shares the same identifying columns (MODEL_KEYS), so any
file can be filtered/joined on its own in LaTeX (pgfplotstable) or pandas.
String values contain no commas and missing values are written as "nan",
both of which pgfplots reads natively.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.config import (
    FEATURE_COLS, PROTECTED_COL, PROTECTED_PRIV, PROTECTED_UNPRIV,
    GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL,
)
from pipeline.cf_generation import fit_scm, generate_test_race_cfs
from pipeline.ldp_correction import flip_probability, fit_ols_naive, fit_ols_corrected_ldp
import pipeline.dp_causal_analysis as dpc

_PROTECTED_IDX = FEATURE_COLS.index(PROTECTED_COL)
_LDP_CORRECTED_SUFFIX = "_ldp_corrected"

MODEL_KEYS = ["dataset", "seed", "scenario", "family", "category", "method",
              "aug_method", "fl_constraint", "privacy", "ldp_epsilon", "ldp_corrected"]


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

def component_flags(aug_method) -> dict:
    """Which of the three proposed methods a (possibly '+'-combined,
    possibly '_ldp_corrected') aug_method uses."""
    if aug_method is None or str(aug_method) == "none":
        parts = []
    else:
        name = str(aug_method)
        if name.endswith(_LDP_CORRECTED_SUFFIX):
            name = name[:-len(_LDP_CORRECTED_SUFFIX)]
        parts = name.split("+")
    if "add_comparators_bidir" in parts:
        s3 = "bidir"
    elif "add_comparators" in parts:
        s3 = "one_sided"
    else:
        s3 = "none"
    return {"S1_SCM": int("SCM" in parts), "S2_update_labels": int("update_labels" in parts),
            "S3_add_comparators": s3, "n_aug_components": len(parts)}


def _method_id(meta: dict) -> str:
    aug, fl = str(meta.get("aug_method", "none")), str(meta.get("fl_constraint", "none"))
    if aug != "none":
        return aug
    if fl != "none":
        return "fairlearn_" + fl
    return "baseline"


def _model_index(all_scenarios, scenario_meta, dataset, seed, n_original) -> pd.DataFrame:
    rows = []
    for sc, sm in all_scenarios.items():
        meta = scenario_meta.get(sc, {})
        eps = meta.get("ldp_epsilon")
        aug = meta.get("aug_method", "none")
        for fam, res in sm.results.items():
            row = {
                "dataset": dataset, "seed": seed, "scenario": sc, "family": fam,
                "category": meta.get("ablation_category"), "method": _method_id(meta),
                "aug_method": aug, "fl_constraint": meta.get("fl_constraint", "none"),
                "privacy": int(eps is not None), "ldp_epsilon": eps,
                "ldp_corrected": int(str(aug).endswith(_LDP_CORRECTED_SUFFIX)),
            }
            row.update(component_flags(aug))
            row["n_train"] = res.n_train
            row["n_synthetic"] = res.n_train - n_original
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Counterfactual fairness
# ---------------------------------------------------------------------------

def _scores(estimator, X):
    try:
        return np.asarray(estimator.predict_proba(X))[:, 1]
    except Exception:
        return np.asarray(estimator.predict(X), dtype=float)


def counterfactual_metrics(estimator, X, X_cf, protected, prefix: str) -> Dict[str, float]:
    """Compare a model's predictions on X with its predictions on the
    counterfactual X_cf (row i of X_cf is individual i in the other group).

      <prefix>_consistency       fraction of hard predictions unchanged (ideal 1)
      <prefix>_gap_mean / _max   mean / max |score(x) - score(x_cf)|, the
                                 paper's Delta_f on scores (ideal 0)
      <prefix>_gap_mean_priv     ... over privileged individuals (priv -> unpriv)
      <prefix>_gap_mean_unpriv   ... over unprivileged individuals (unpriv -> priv)
      <prefix>_shift_unpriv      mean signed score(x_cf) - score(x) when an
                                 unprivileged individual is moved to the
                                 privileged group (< 0: becoming privileged
                                 lowers the predicted score)
      <prefix>_shift_priv        same, privileged -> unprivileged
    """
    y, y_cf = np.asarray(estimator.predict(X)), np.asarray(estimator.predict(X_cf))
    s, s_cf = _scores(estimator, X), _scores(estimator, X_cf)
    gap = np.abs(s - s_cf)
    priv, unpriv = protected == PROTECTED_PRIV, protected == PROTECTED_UNPRIV

    def _mean(v, mask):
        return float(v[mask].mean()) if mask.any() else float("nan")

    return {
        prefix + "_consistency": float(np.mean(y == y_cf)),
        prefix + "_consistency_priv": _mean((y == y_cf).astype(float), priv),
        prefix + "_consistency_unpriv": _mean((y == y_cf).astype(float), unpriv),
        prefix + "_gap_mean": float(gap.mean()),
        prefix + "_gap_max": float(gap.max()),
        prefix + "_gap_mean_priv": _mean(gap, priv),
        prefix + "_gap_mean_unpriv": _mean(gap, unpriv),
        prefix + "_shift_priv": _mean(s_cf - s, priv),
        prefix + "_shift_unpriv": _mean(s_cf - s, unpriv),
    }


def _counterfactual_rows(all_scenarios, X_test, X_test_cf_scm) -> List[dict]:
    """Two counterfactuals per test individual, the same for every model:
      cf_flip  only the protected attribute flipped (direct path only)
      cf_scm   the SCM counterfactual: protected attribute flipped AND its
               causal descendants shifted (direct + indirect paths), from
               the SCM fit on the CLEAN training data -- so private and
               non-private models are scored against one reference.
    """
    protected = X_test[:, _PROTECTED_IDX]
    X_flip = X_test.copy()
    X_flip[:, _PROTECTED_IDX] = 1.0 - protected
    rows = []
    for sc, sm in all_scenarios.items():
        for fam, res in sm.results.items():
            row = {"scenario": sc, "family": fam}
            row.update(counterfactual_metrics(res.estimator, X_test, X_flip, protected, "cf_flip"))
            if X_test_cf_scm is not None:
                row.update(counterfactual_metrics(res.estimator, X_test, X_test_cf_scm, protected, "cf_scm"))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Group / individual fairness (re-shaped from fairness_metrics.compute_fairness)
# ---------------------------------------------------------------------------

_GROUP_RENAME = {
    "TPR_{}".format(GROUP_PRIV_LABEL): "tpr_priv", "TPR_{}".format(GROUP_UNPRIV_LABEL): "tpr_unpriv",
    "FPR_{}".format(GROUP_PRIV_LABEL): "fpr_priv", "FPR_{}".format(GROUP_UNPRIV_LABEL): "fpr_unpriv",
    "PosRate_{}".format(GROUP_PRIV_LABEL): "posrate_priv", "PosRate_{}".format(GROUP_UNPRIV_LABEL): "posrate_unpriv",
    "Precision_{}".format(GROUP_PRIV_LABEL): "precision_priv",
    "Precision_{}".format(GROUP_UNPRIV_LABEL): "precision_unpriv",
}


def _fairness_rows(fairness_results):
    group_rows, ind_rows = [], []
    for sc, fams in fairness_results.items():
        for fam, fr in fams.items():
            g = {_GROUP_RENAME.get(k, k): v for k, v in fr.group.items() if k != "Theil"}
            g.update({"scenario": sc, "family": fam,
                      "priv_group": GROUP_PRIV_LABEL, "unpriv_group": GROUP_UNPRIV_LABEL})
            group_rows.append(g)
            ind_rows.append({"scenario": sc, "family": fam,
                             "consistency_knn5": fr.individual.get("consistency", float("nan")),
                             "theil_index": fr.group.get("Theil", float("nan"))})
    return group_rows, ind_rows


# ---------------------------------------------------------------------------
# LDP correction
# ---------------------------------------------------------------------------

def _mm_design_names() -> List[str]:
    # Same column order dp_causal_analysis._main_mm_design builds.
    return ["female", "nonwhite"] + (["age"] if "age" in FEATURE_COLS else [])


def ldp_correction_tables(data, ldp_draws: Dict[float, np.ndarray], dataset, seed):
    """Every quantity the LDP correction uses, for the exact randomized-
    response draw each privacy scenario trained on (ldp_draws: epsilon ->
    the LDP-perturbed X_train of that draw).

    Two regressions are corrected (pipeline/ldp_correction.py):
      scm  dep_col ~ 1 + protected                 (SCM, S1)
      mm   dep_col ~ 1 + female + nonwhite (+ age)  (update_labels / add_comparators)
    For each: true (clean data), naive (OLS on the LDP data, what the
    uncorrected scenarios use) and corrected coefficients.
    """
    prot_true = data.X_train[:, _PROTECTED_IDX]
    params, coefs = [], []
    X_design_true, mm_pidx = dpc._main_mm_design(data.X_train)
    names = _mm_design_names()
    assert X_design_true.shape[1] == len(names), "main_MM design columns changed -- update _mm_design_names()"
    mm_true = {c: fit_ols_naive(X_design_true, data.X_train[:, FEATURE_COLS.index(c)].astype(float))
               for c in dpc.datasets_info}

    for eps in sorted(ldp_draws):
        X_ldp = ldp_draws[eps]
        s = X_ldp[:, _PROTECTED_IDX]
        p = flip_probability(eps)
        lam = 1.0 - 2.0 * p
        pi_obs = float(s.mean())
        pi_hat = (pi_obs - p) / lam if abs(lam) > 1e-9 else float("nan")
        params.append({
            "dataset": dataset, "seed": seed, "epsilon": eps,
            "p_flip": p, "lambda": lam, "n_train": len(s),
            "n_flipped": int((s != prot_true).sum()), "flip_rate_empirical": float((s != prot_true).mean()),
            "base_rate_true": float(prot_true.mean()), "base_rate_observed": pi_obs,
            "base_rate_debiased": pi_hat,
            "base_rate_debiased_clipped": min(max(pi_hat, 1e-3), 1 - 1e-3) if pi_hat == pi_hat else float("nan"),
        })

        def _add(regression, dep_col, term, true, naive, corr):
            coefs.append({
                "dataset": dataset, "seed": seed, "epsilon": eps, "p_flip": p, "lambda": lam,
                "regression": regression, "dep_col": dep_col, "term": term,
                "is_protected_term": int(term == "protected"),
                # the two regressions code the protected attribute differently,
                # so their 'protected' slopes can have opposite signs
                "protected_variable": PROTECTED_COL if regression == "scm" else names[mm_pidx],
                "true": true, "naive": naive, "corrected": corr,
                "naive_error": naive - true, "corrected_error": corr - true,
                "naive_abs_error": abs(naive - true), "corrected_abs_error": abs(corr - true),
                "naive_over_true": naive / true if abs(true) > 1e-12 else float("nan"),
                "corrected_over_true": corr / true if abs(true) > 1e-12 else float("nan"),
            })

        if data.scm is not None:
            cols = data.scm.dependent_cols
            naive = fit_scm(X_ldp, dependent_cols=cols)
            corr = dpc.fit_scm_corrected_ldp(X_ldp, eps, dependent_cols=cols)
            for c in cols:
                for i, term in enumerate(["intercept", "protected"]):
                    _add("scm", c, term, float(data.scm.coef[c][i]),
                         float(naive.coef[c][i]), float(corr.coef[c][i]))

        X_obs, _ = dpc._main_mm_design(X_ldp)
        for c in dpc.datasets_info:
            y = data.X_train[:, FEATURE_COLS.index(c)].astype(float)
            b_naive = fit_ols_naive(X_obs, y)
            b_corr = fit_ols_corrected_ldp(X_obs, y, mm_pidx, eps)
            terms = ["intercept"] + ["protected" if j == mm_pidx else n for j, n in enumerate(names)]
            for i, term in enumerate(terms):
                _add("mm", c, term, float(mm_true[c][i]), float(b_naive[i]), float(b_corr[i]))
    return pd.DataFrame(params), pd.DataFrame(coefs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _write(df: pd.DataFrame, out_dir: str, name: str, written: list, verbose: bool):
    if df is None or df.empty:
        return
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False, na_rep="nan")
    written.append(name)
    if verbose:
        print("  [paper] {:<34} {:>6} rows -> {}".format(name, len(df), path))


def export_paper_tables(out_dir, all_scenarios, scenario_meta, fairness_results, data,
                        ldp_draws: Optional[Dict[float, np.ndarray]], dataset, seed,
                        verbose: bool = True) -> List[str]:
    """Write every paper CSV (see module docstring) to out_dir; returns the
    file names written."""
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    index = _model_index(all_scenarios, scenario_meta, dataset, seed, len(data.X_train))
    if index.empty:
        return written
    key = index[MODEL_KEYS]

    def _keyed(rows):
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return key.merge(df, on=["scenario", "family"], how="inner")

    perf_rows = []
    for sc, sm in all_scenarios.items():
        for fam, res in sm.results.items():
            m = res.metrics or {}
            perf_rows.append({"scenario": sc, "family": fam, "n_train": res.n_train,
                              **{k: m.get(k, float("nan"))
                                 for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]}})
    group_rows, ind_rows = _fairness_rows(fairness_results)
    X_test_cf_scm = generate_test_race_cfs(data.X_test, data.scm) if data.scm is not None else None
    cf_rows = _counterfactual_rows(all_scenarios, data.X_test, X_test_cf_scm)

    tables = {
        "performance.csv": _keyed(perf_rows),
        "fairness_group.csv": _keyed(group_rows),
        "fairness_individual.csv": _keyed(ind_rows),
        "counterfactual_fairness.csv": _keyed(cf_rows),
    }
    _write(index, out_dir, "models.csv", written, verbose)
    for name, df in tables.items():
        _write(df, out_dir, name, written, verbose)

    combined = index
    for df in tables.values():
        if not df.empty:
            extra = [c for c in df.columns if c not in combined.columns or c in ("scenario", "family")]
            combined = combined.merge(df[extra], on=["scenario", "family"], how="left")
    _write(combined, out_dir, "all_models.csv", written, verbose)

    if ldp_draws:
        params, coefs = ldp_correction_tables(data, ldp_draws, dataset, seed)
        _write(params, out_dir, "ldp_correction_params.csv", written, verbose)
        _write(coefs, out_dir, "ldp_correction_coefficients.csv", written, verbose)

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(_README.format(dataset=dataset, seed=seed, priv=GROUP_PRIV_LABEL,
                               unpriv=GROUP_UNPRIV_LABEL, prot=PROTECTED_COL, priv_val=PROTECTED_PRIV,
                               files="\n".join("- " + w for w in written)))
    return written


_README = """# Paper tables -- dataset={dataset}, seed={seed}

Protected attribute: `{prot}` (privileged = {priv}, unprivileged = {unpriv}).
All metrics are computed on the same held-out test set. Missing values are `nan`.

Files written in this run:
{files}

## Columns shared by every model table
| column | meaning |
|---|---|
| scenario, family | the trained model (family: logistic_regression / random_forest / xgboost) |
| category | no_privacy_no_fairness, fairness_causal_only, fairness_causal_combined, fairness_fairlearn_only, privacy_only, privacy_plus_fairness |
| method | baseline, an augmentation method (`SCM`, `update_labels`, `add_comparators`, `add_comparators_bidir`, `+`-joined combinations, `*_ldp_corrected`), or `fairlearn_<constraint>` |
| privacy, ldp_epsilon | 1 and the epsilon if the protected attribute went through randomized response (else 0, nan) |
| ldp_corrected | 1 if the augmentation used the LDP-corrected regression |
| S1_SCM, S2_update_labels, S3_add_comparators, n_aug_components | which proposed methods the model uses (models.csv / all_models.csv) |
| n_train, n_synthetic | training rows, and how many of them were added by augmentation |

## performance.csv
accuracy, precision, recall, f1, auc_roc (positive class = 1).

## fairness_group.csv
SPD = P(yhat=1 | priv) - P(yhat=1 | unpriv) (ideal 0); DI = P(yhat=1 | unpriv) / P(yhat=1 | priv) (ideal 1);
EOD = TPR_priv - TPR_unpriv (ideal 0); AOD = mean of FPR and TPR gaps (ideal 0);
EqOdds = max(|TPR gap|, |FPR gap|) (ideal 0); PP = precision_priv - precision_unpriv (ideal 0);
plus tpr/fpr/posrate/precision for each group (_priv / _unpriv).

## fairness_individual.csv
consistency_knn5 = mean agreement of a prediction with its 5 nearest test neighbours (ideal 1);
theil_index = Theil index of prediction correctness (ideal 0).

## counterfactual_fairness.csv
Two counterfactuals per test individual, identical for every model:
- `cf_flip_*`: only the protected attribute is flipped (direct path only).
- `cf_scm_*`: the SCM counterfactual -- protected attribute flipped AND its causal
  descendants shifted by the SCM fit on the clean training data (direct + indirect
  paths; the counterfactual of the paper). `nan` for datasets without an SCM spec.

For each: `_consistency` = fraction of predictions unchanged (ideal 1; `_priv`/`_unpriv`
by original group); `_gap_mean`/`_gap_max` = mean/max |score(x) - score(x_cf)|, i.e.
Delta_f on predicted scores (ideal 0; `_gap_mean_priv`/`_unpriv` by group);
`_shift_unpriv` = mean signed score change when an unprivileged individual becomes
privileged, `_shift_priv` the reverse (ideal 0). For Fairlearn ThresholdOptimizer
the scores come from its unconstrained base model, so only `_consistency` reflects
its group-specific thresholds; ExponentiatedGradient predictions use a fixed seed,
so x and x_cf share the same randomization.

## ldp_correction_params.csv (one row per epsilon)
p_flip = 1/(1+e^epsilon); lambda = 1 - 2 p_flip; n_flipped / flip_rate_empirical of
the draw the privacy scenarios trained on; base_rate_true / _observed of the
protected attribute and base_rate_debiased = (observed - p_flip) / lambda
(_clipped: clipped to [0.001, 0.999] as used by the correction).

## ldp_correction_coefficients.csv (one row per epsilon x regression x dep_col x term)
regression `scm`: dep_col ~ 1 + protected (method S1); regression `mm`:
dep_col ~ 1 + female + nonwhite (+ age) (methods S2 and S3).
term = intercept / protected / the other covariates. `protected_variable` names the
regressor behind term=protected: in `scm` it is the protected column as coded in the
data (privileged = {priv_val}), in `mm` the `nonwhite`/`female` indicator (1 = unprivileged
group), so the two regressions' protected slopes have opposite signs. `true` = OLS on clean data,
`naive` = OLS on the LDP data (uncorrected scenarios), `corrected` = the LDP
measurement-error correction (`*_ldp_corrected` scenarios); errors are
estimate - true, plus absolute errors and estimate/true ratios.
"""
