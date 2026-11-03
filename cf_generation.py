"""
cf_generation.py -- Level-3 structural (mutatis-mutandis) counterfactual generation.
Implements Pearl Abduction-Action-Prediction over the protected attribute.

Causal DAG: protected_attr -> SCM_DEPENDENT_COLS (config-driven per dataset --
see pipeline/datasets/<name>.py::SCMSpec). For COMPAS this is
race -> priors_count, juv_fel_count, juv_misd_count, juv_other_count.
Variables not listed in SCM_DEPENDENT_COLS are left untouched by the
intervention (only the protected attribute itself + the dependent columns
change).

augmentation_method='SCM' is only available for datasets whose DatasetSpec
defines an SCMSpec (i.e. config.SCM_DEPENDENT_COLS is non-empty) -- fit_scm()
raises a clear NotImplementedError otherwise. This is the one part of the
pipeline that genuinely cannot be made dataset-agnostic: it requires a
person to assert which features are causally downstream of the protected
attribute for a given dataset. 'update_labels' and 'add_comparators' below
(main_MM) make no such assumption and work for any registered dataset.
"""

from __future__ import annotations
import os, sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import FEATURE_COLS, RACE_ENC_COL, PROTECTED_COL, PROTECTED_PRIV, GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL, datasets_info, TARGET_COL, sensitive_columns, female_values, male_values, white_values, DATA_DIR, OUTPUT_DIR, AUG_RELABEL_DISTANCE_PERCENTILE, AUG_RELABEL_K_NEIGHBORS, AUG_RELABEL_AGREEMENT_THRESHOLD, AUG_COMPARATORS_BIDIRECTIONAL, SCM_DEPENDENT_COLS
import pipeline.ldp_correction as ldpc

_PROTECTED_IDX = FEATURE_COLS.index(PROTECTED_COL)


@dataclass
class SCMParams:
    dependent_cols: List[str]
    coef:    Dict[str, np.ndarray] = field(default_factory=dict)  # col -> [intercept, slope]
    resid:   Dict[str, np.ndarray] = field(default_factory=dict)  # col -> residuals on X_train
    max_val: Dict[str, float]      = field(default_factory=dict)  # col -> max observed value (clip bound)
    n_train: int = 0


def fit_scm(X_train: np.ndarray, dependent_cols: List[str] = None, verbose: bool = False) -> SCMParams:
    """Fit `col ~ 1 + protected_attr` for each dependent column via OLS.

    dependent_cols defaults to config.SCM_DEPENDENT_COLS (derived from the
    active dataset's SCMSpec). Raises NotImplementedError if empty -- i.e.
    the active dataset has not defined a causal spec for the 'SCM' method.
    """
    dependent_cols = dependent_cols if dependent_cols is not None else SCM_DEPENDENT_COLS
    if not dependent_cols:
        raise NotImplementedError(
            "augmentation_method='SCM' is unavailable for the active dataset: "
            "no SCMSpec was registered (pipeline.config.SCM_DEPENDENT_COLS is empty). "
            "Add one in pipeline/datasets/<name>.py (SCMSpec(dependent_cols=[...])) "
            "naming the features that are causally downstream of the protected "
            "attribute, or use augmentation_method='update_labels' / "
            "'add_comparators' instead, which need no causal spec."
        )
    X = np.asarray(X_train, dtype=float)
    N = len(X)
    protected = X[:, _PROTECTED_IDX]
    A = np.column_stack([np.ones(N), protected])
    coef, resid, max_val = {}, {}, {}
    for col in dependent_cols:
        idx = FEATURE_COLS.index(col)
        y = X[:, idx]
        c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        coef[col] = c
        resid[col] = y - (A @ c)
        max_val[col] = float(y.max())
        if verbose:
            b0, b1 = c
            print(f"[SCM] {col:<20} = {b0:.4f} + {b1:.4f}*protected   "
                  f"residual range=[{resid[col].min():.3f}, {resid[col].max():.3f}]")
    return SCMParams(dependent_cols=list(dependent_cols), coef=coef, resid=resid,
                     max_val=max_val, n_train=N)


def _apply_intervention(X, resid_map, scm, protected_value):
    N = len(X)
    A_cf = np.column_stack([np.ones(N), np.full(N, protected_value)])
    X_new = X.copy()
    for col in scm.dependent_cols:
        idx = FEATURE_COLS.index(col)
        pred = A_cf @ scm.coef[col] + resid_map[col]
        X_new[:, idx] = np.clip(np.round(pred), 0.0, scm.max_val[col])
    X_new[:, _PROTECTED_IDX] = protected_value
    return X_new


def generate_training_race_cfs(X_train, y_train, scm, verbose=True, return_src_idx=False):
    """Generate bidirectional protected-attribute CFs for the training set
    (flip 0->1 rows to 1, and 1->0 rows to 0 -- binary protected attribute).

    CF rows are grouped by protected value (all protected==1 rows first), so
    they are NOT in X_train's row order. return_src_idx=True also returns
    src_idx, where X_cf[k] is the twin of X_train[src_idx[k]]."""
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    mask1 = X[:, _PROTECTED_IDX] == 1
    mask0 = ~mask1
    n1 = int(mask1.sum())
    n0 = int(mask0.sum())
    parts_X, parts_y = [], []
    if n1 > 0:
        resid1 = {col: scm.resid[col][mask1] for col in scm.dependent_cols}
        parts_X.append(_apply_intervention(X[mask1], resid1, scm, protected_value=0.0))
        parts_y.append(y[mask1])
    if n0 > 0:
        resid0 = {col: scm.resid[col][mask0] for col in scm.dependent_cols}
        parts_X.append(_apply_intervention(X[mask0], resid0, scm, protected_value=1.0))
        parts_y.append(y[mask0])
    X_cf = np.vstack(parts_X)
    y_cf = np.concatenate(parts_y)
    if verbose:
        print(f"[CF] Protected-attr CFs: {n1} do(0) (1->0) + {n0} do(1) (0->1) = {len(X_cf):,} total")
        X_orig = np.vstack([X[mask1], X[mask0]])
        for col in scm.dependent_cols:
            idx = FEATURE_COLS.index(col)
            d = np.abs(X_cf[:, idx] - X_orig[:, idx])
            print(f"       {col} delta:  mean={d.mean():.3f}  max={d.max():.3f}")
    if return_src_idx:
        src_idx = np.concatenate([np.flatnonzero(mask1), np.flatnonzero(mask0)])
        return X_cf, y_cf, src_idx
    return X_cf, y_cf


def generate_training_flip_cfs(X_train, y_train, verbose=True, return_src_idx=False):
    """Non-causal baseline: one twin per training row with ONLY the
    protected attribute flipped (binary, 1 - value); every other feature,
    descendants included, and the label are copied unchanged. No SCM is fit.

    Twins are in X_train's row order, so src_idx is simply arange(n)."""
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    X_cf = X.copy()
    X_cf[:, _PROTECTED_IDX] = 1.0 - X[:, _PROTECTED_IDX]
    y_cf = y.copy()
    if verbose:
        n1 = int((X[:, _PROTECTED_IDX] == 1).sum())
        print(f"[CF] Flip-only CFs: {n1} (1->0) + {len(X) - n1} (0->1) = {len(X_cf):,} total "
              f"(protected attribute only, no SCM)")
    if return_src_idx:
        return X_cf, y_cf, np.arange(len(X))
    return X_cf, y_cf


def generate_test_race_cfs(X_test, scm):
    """Generate protected-attribute CFs for the test set (for counterfactual
    fairness evaluation)."""
    X = np.asarray(X_test, dtype=float)
    N = len(X)
    protected = X[:, _PROTECTED_IDX]
    A_test = np.column_stack([np.ones(N), protected])
    resid = {}
    for col in scm.dependent_cols:
        idx = FEATURE_COLS.index(col)
        resid[col] = X[:, idx] - (A_test @ scm.coef[col])
    protected_cf = 1.0 - protected
    # protected_cf is per-row (not a scalar) -- _apply_intervention broadcasts
    # np.full(N, protected_cf) to itself, so this handles the per-row flip.
    return _apply_intervention(X, resid, scm, protected_value=protected_cf)


# Process each dataset
def _protected_params(dataset_name):
    """Resolve which protected attribute main_MM should intervene on.

    Returns a dict with:
      prot_col      column name of the protected attribute
      priv_val      numeric value that identifies the privileged group
      do_prefix     filename prefix of the corresponding do-CSVs written by
                    the OLS SCM loop  ('do_white' | 'do_male' | ...)
      priv_label    human label of the privileged group (used in filenames)
      unpriv_label  human label of the unprivileged group

    Dispatches on config.PROTECTED_COL -- the SAME setting LDP, the fairness
    metrics, AIA and the SCM all use -- rather than the separate legacy
    config.PROTECTED flag ('RACE' | 'SEX') this used to read. That flag could
    silently drift out of sync with PROTECTED_COL (e.g. PROTECTED_COL switched
    to sex_enc while PROTECTED was still 'RACE'), which made update_labels/
    add_comparators relabel against a completely different column than the
    one every fairness metric is actually measured on -- there is now only
    one setting to keep in sync.

    sensitive_columns[dataset_name] is always [sex-like col, race-like col]
    by convention (matches the OLS loop above, which always derives "female"/
    "male" from index 0 and "nonwhite" from index 1). do_prefix must be one of
    'do_male' / 'do_white' since those are the only two intervention CSVs that
    loop ever writes -- so PROTECTED_COL has to resolve to one of this
    dataset's two sensitive_columns, not an arbitrary feature.
    """
    sens = sensitive_columns[dataset_name]
    if PROTECTED_COL == sens[0]:
        do_prefix = "do_male"
    elif len(sens) > 1 and PROTECTED_COL == sens[1]:
        do_prefix = "do_white"
    else:
        raise ValueError(
            f"main_MM: config.PROTECTED_COL={PROTECTED_COL!r} is not one of "
            f"this dataset's sensitive_columns {sens!r} -- extend main_MM's "
            "OLS loop and _protected_params() to support it.")
    return {
        "prot_col":     PROTECTED_COL,
        "priv_val":     float(PROTECTED_PRIV),
        "do_prefix":    do_prefix,
        "priv_label":   GROUP_PRIV_LABEL,
        "unpriv_label": GROUP_UNPRIV_LABEL,
    }


def _protected_local_index(train_columns: List[str], do_prefix: str) -> int:
    """Which index within train_columns (['female','nonwhite'(,'age')]) is
    the actually-privatized protected attribute, given do_prefix
    ('do_male' -> 'female' is protected; 'do_white' -> 'nonwhite' is) --
    matches _protected_params(dataset_name)['do_prefix']."""
    protected_name = "female" if do_prefix == "do_male" else "nonwhite"
    return train_columns.index(protected_name)


def _fit_do_intervention_predictions(df: pd.DataFrame, dep_col: str, train_columns: List[str],
                                     protected_local_idx: int, epsilon: Optional[float]):
    """Fit `dep_col ~ 1 + train_columns` once, then generate BOTH
    do(female:=0) and do(nonwhite:=0) counterfactual predictions from it
    (Pearl abduction-action-prediction) -- matches main_MM's pre-existing
    behaviour of always writing both do_male_*.csv and do_white_*.csv
    regardless of which one is actually config.PROTECTED_COL.

    epsilon=None uses plain OLS (statsmodels) -- BYTE-IDENTICAL to this
    function's original behaviour: model.predict() and `A @ model.params`
    are the same linear-algebra operation on the same fitted coefficients,
    so routing both actions through the shared numpy prediction step below
    changes nothing numerically when there is no correction to apply.

    epsilon=<float> corrects the coefficient on train_columns[protected_local_idx]
    for the known RR flip probability (and, through the covariance-matrix
    correction, every OTHER coefficient correlated with it) -- see
    pipeline/ldp_correction.py::fit_ols_corrected_ldp.

    Returns {"do_male": np.ndarray, "do_white": np.ndarray} of rounded,
    int-cast scf_dep_col predictions, one per row of df.
    """
    X = df[train_columns].to_numpy(dtype=float)
    y = df[dep_col].to_numpy(dtype=float)

    if epsilon is None:
        design = sm.add_constant(df[train_columns], has_constant='add')
        model = sm.OLS(df[dep_col], design).fit()
        beta = model.params.reindex(["const"] + train_columns).to_numpy()
    else:
        beta = ldpc.fit_ols_corrected_ldp(X, y, protected_local_idx, epsilon)

    predictions = {}
    for action, col in [("do_male", "female"), ("do_white", "nonwhite")]:
        if col not in train_columns:
            continue
        col_idx = train_columns.index(col)
        cf = ldpc.counterfactual_predictions(X, y, beta, col_idx, 0.0)
        predictions[action] = np.round(cf).astype(int)
    return predictions


def main_MM(X_train, y_train, augmentation_method, dataset_name, epsilon: Optional[float] = None,
            return_src_idx: bool = False):
    # return_src_idx=True (add_comparators only) also returns src_idx, where
    # X_aug[k] is the twin of X_train[src_idx[k]] -- comparator rows are
    # grouped unprivileged-first, not in X_train's row order.
    # Paths
    path_data    = os.path.join(DATA_DIR, "")          # e.g. .../New/data/
    path_results = os.path.join(OUTPUT_DIR, "results", "mm_cf_csvs", "")
    # Ensure both directories exist
    os.makedirs(path_data,    exist_ok=True)
    os.makedirs(path_results, exist_ok=True)

    # Initialize lists to store data for white and male comparators
    white_comparator_data = []
    male_comparator_data = []
    dependent_columns =  datasets_info
    sen_columns = [RACE_ENC_COL]
    # for dependent_columns in dependent_columns.items():
    #print(f"Processing dataset: {dataset_name}")

    dataset = pd.DataFrame(X_train, columns = FEATURE_COLS) #data[FEATURE_COLS].astype(float)
    label = pd.Series(y_train, name=TARGET_COL) #data[[TARGET_COL]].astype(float)

    # Add an ID column
    dataset['id'] = range(1, len(dataset) + 1)
    
    dataset['target'] = label  # Add target variable to the dataset
    dataset.to_csv(f"{path_data}{dataset_name}_with_ids.csv", index=False)
    dataset = dataset.drop(columns=["target"])  # Drop target variable for modeling
    dataset.reset_index(drop=True, inplace=True)  # Reset index after adding ID column

    white_comparator_data = dataset.copy()  # Initialize with original dataset for white comparators
    male_comparator_data = dataset.copy()  # Initialize with original dataset for male comparators

    # Resolved once up front (needed inside the per-dep_col loop below when
    # epsilon is given, to know which of female/nonwhite is the actually
    # LDP-perturbed column) and reused again later for priv_comparator_data --
    # a single call, not one per dep_col or a second one further down.
    prot = _protected_params(dataset_name)

    # Process each dependent column
    # dep columns are the cf columns we can do changes to
    for dep_col in dependent_columns:
            #print(f"Processing dependent column: {dep_col}")

            # Ensure required columns exist
            
            if not all(col in dataset.columns for col in sen_columns + ["age", dep_col]):
            # if not all(col in dataset.columns for col in [sen_columns, "age", dep_col]):
                print(f"Skipping {dep_col} as required columns are missing.")
                continue

            # Check if 'age' exists in the dataset columns
            if 'age' in dataset.columns:
                # Prepare data with age
                df = dataset[[sensitive_columns[dataset_name][0], sensitive_columns[dataset_name][1], "age", dep_col, "id"]].copy()
                df["female"] = (df[sensitive_columns[dataset_name][0]] == female_values[dataset_name]).astype(int)
                df["male"] = (df[sensitive_columns[dataset_name][0]] == male_values[dataset_name]).astype(int)
                df["nonwhite"] = (df[sensitive_columns[dataset_name][1]] != white_values[dataset_name]).astype(int)

                # Fit dep_col ~ 1 + female + nonwhite + age once, then derive
                # BOTH do(female:=0)/do(nonwhite:=0) counterfactual predictions
                # from it (Pearl abduction-action-prediction). epsilon=None
                # here reproduces the original statsmodels-only behaviour
                # exactly; epsilon=<float> applies the LDP correction to
                # whichever column is actually config.PROTECTED_COL -- see
                # _fit_do_intervention_predictions's docstring.
                train_columns = ["female", "nonwhite", "age"]
                protected_local_idx = _protected_local_index(train_columns, prot["do_prefix"])
                predictions = _fit_do_intervention_predictions(
                    df, dep_col, train_columns, protected_local_idx, epsilon)

                # Generate counterfactuals
                for action, action_values in {"do_male": {"female": 0}, "do_white": {"nonwhite": 0}}.items():
                    cf_df = df.copy()
                    for col, value in action_values.items():
                        cf_df[col] = value

                    cf_df[f"scf_{dep_col}"] = predictions[action]

                    # Save results
                    output_file = f"{path_results}{action}_{dataset_name}_{dep_col}.csv"
                    cf_df.to_csv(output_file, index=False)
                    #print(f"Saved {action} results to {output_file}")
            else:
                # Prepare data without age
                df = dataset[[sensitive_columns[dataset_name][0], sensitive_columns[dataset_name][1], dep_col, "id"]].copy()
                df["female"] = (df[sensitive_columns[dataset_name][0]] == female_values[dataset_name]).astype(int)
                df["male"] = (df[sensitive_columns[dataset_name][0]] == male_values[dataset_name]).astype(int)
                df["nonwhite"] = (df[sensitive_columns[dataset_name][1]] != white_values[dataset_name]).astype(int)

                # Same fit-once / both-actions approach as the with-age
                # branch above, just with a shorter covariate list.
                train_columns = ["female", "nonwhite"]
                protected_local_idx = _protected_local_index(train_columns, prot["do_prefix"])
                predictions = _fit_do_intervention_predictions(
                    df, dep_col, train_columns, protected_local_idx, epsilon)

                # Generate counterfactuals
                for action, action_values in {"do_male": {"female": 0}, "do_white": {"nonwhite": 0}}.items():
                    cf_df = df.copy()
                    for col, value in action_values.items():
                        cf_df[col] = value

                    cf_df[f"scf_{dep_col}"] = predictions[action]

                    # Save results
                    output_file = f"{path_results}{action}_{dataset_name}_{dep_col}.csv"
                    cf_df.to_csv(output_file, index=False)
                    #print(f"Saved {action} results to {output_file}")

        # After processing all dependent columns, merge and save comparator files
        # if white_comparator_data:
            # white_merged = pd.concat(white_comparator_data).drop_duplicates(subset="id")
    # -----------------------------------------------------------------
    # Attribute-agnostic dispatch.
    #
    # The old code raised NotImplementedError unless config.PROTECTED == 'RACE'
    # and then hard-coded do_white / white_comparator_data. It is now driven
    # by _protected_params() (resolved once, above, before the per-dep_col
    # loop), which reads config.PROTECTED_COL (the same setting everything
    # else in the pipeline uses) -- so the same code path handles either of
    # this dataset's sensitive_columns without a separate flag to keep in
    # sync.
    # -----------------------------------------------------------------
    prot_col     = prot["prot_col"]
    priv_val     = prot["priv_val"]
    do_prefix    = prot["do_prefix"]
    priv_lbl     = prot["priv_label"]
    unpriv_lbl   = prot["unpriv_label"]

    # Build privileged-comparator data from the corresponding do_*.csv files.
    # The OLS SCM loop above always writes both do_white_* and do_male_*
    # CSVs, so we simply pick the one matching PROTECTED_COL.
    priv_comparator_data = dataset.copy()
    augmented_set = pd.DataFrame()
    for dep_col in dependent_columns:
        do_file = f"{path_results}{do_prefix}_{dataset_name}_{dep_col}.csv"
        if os.path.exists(do_file):
            do_df = pd.read_csv(do_file)
            priv_comparator_data = priv_comparator_data.merge(
                do_df[["id", f"scf_{dep_col}"]],
                on="id",
                how="left",
            )
            priv_comparator_data.loc[:, dep_col] = priv_comparator_data.loc[:, f"scf_{dep_col}"]
            priv_comparator_data = priv_comparator_data.drop(columns=[f"scf_{dep_col}"])
    priv_comparator_data.to_csv(
        f"{path_results}{priv_lbl}_comparators_{dataset_name}.csv",
        index=False,
    )

    if augmentation_method == "update_labels":
        # Build original data with IDs + targets.
        original_df = dataset[FEATURE_COLS].astype(float).copy()
        original_df["id"] = range(1, len(original_df) + 1)
        original_df["target"] = label.values if hasattr(label, "values") else label
        common_cols = [c for c in priv_comparator_data.columns
                       if c in original_df.columns and c not in {"id", "target"}]

        id_to_target = original_df.set_index("id")["target"]
        desired_targets = priv_comparator_data["id"].map(id_to_target)

        # Privileged rows keep their original target.
        priv_mask = priv_comparator_data[prot_col].astype(float) == priv_val

        # Unprivileged rows: label them from their nearest privileged
        # neighbour in the ORIGINAL (un-intervened) dataset.
        original_priv = original_df[
            original_df[prot_col].astype(float) == priv_val
        ].copy()
        unpriv_idx = priv_comparator_data.index[~priv_mask]

        if len(unpriv_idx) > 0 and not original_priv.empty:
            # Feature space for the nearest-neighbour lookup excludes BOTH
            # sensitive columns (sex + race). Comparing on non-sensitive
            # attributes prevents the NN from trivially matching on race or
            # sex and mirrors the previous RACE-only behaviour.
            nn_features = [c for c in common_cols
                           if c not in set(sensitive_columns[dataset_name])]

            X_priv = pd.get_dummies(original_priv[nn_features], drop_first=False)
            X_unpriv = pd.get_dummies(
                priv_comparator_data.loc[unpriv_idx, nn_features],
                drop_first=False,
            ).reindex(columns=X_priv.columns, fill_value=0)

            if AUG_RELABEL_K_NEIGHBORS is None:
                k_neighbors = 1
            else:
                k_neighbors = max(1, int(AUG_RELABEL_K_NEIGHBORS))
            k_neighbors = min(k_neighbors, len(X_priv))
            nn_priv = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
            nn_priv.fit(X_priv)
            distances, idx_priv = nn_priv.kneighbors(X_unpriv)

            priv_targets = original_priv["target"].to_numpy()
            neighbor_labels = priv_targets[idx_priv]              # (n_unpriv, K)
            if k_neighbors == 1 or AUG_RELABEL_AGREEMENT_THRESHOLD is None:
                candidate_labels = neighbor_labels[:, 0]
                agreement_mask   = np.ones(len(neighbor_labels), dtype=bool)
            else:
                pos_frac = neighbor_labels.mean(axis=1)
                candidate_labels = (pos_frac >= 0.5).astype(neighbor_labels.dtype)
                majority_frac    = np.where(candidate_labels == 1, pos_frac, 1.0 - pos_frac)
                agreement_mask   = majority_frac >= float(AUG_RELABEL_AGREEMENT_THRESHOLD)

            nearest_dist = distances[:, 0]
            if AUG_RELABEL_DISTANCE_PERCENTILE is not None:
                threshold = np.percentile(nearest_dist, AUG_RELABEL_DISTANCE_PERCENTILE)
                distance_mask = nearest_dist <= threshold
            else:
                distance_mask = np.ones(len(nearest_dist), dtype=bool)

            keep_mask = agreement_mask & distance_mask
            relabel_idx    = unpriv_idx[keep_mask]
            relabel_labels = candidate_labels[keep_mask]

            desired_targets.loc[relabel_idx] = relabel_labels

            n_total         = len(unpriv_idx)
            n_kept          = int(keep_mask.sum())
            n_dropped_agree = int((~agreement_mask).sum())
            n_dropped_dist  = int(agreement_mask.sum() - keep_mask.sum())
            print(f"[update_labels] prot_col={prot_col}  "
                  f"K={k_neighbors}  "
                  f"agreement_thr={AUG_RELABEL_AGREEMENT_THRESHOLD}  "
                  f"dist_pct={AUG_RELABEL_DISTANCE_PERCENTILE}  "
                  f"-> relabelled {n_kept}/{n_total} "
                  f"(skipped {n_dropped_agree} low-agreement, "
                  f"{n_dropped_dist} far-distance)")

        original_df["target"] = desired_targets.reset_index(drop=True).copy()

        augmented_set = original_df.copy()
        X_aug = augmented_set[FEATURE_COLS].values
        y_aug = augmented_set["target"].values

        print("number of positive labels in augmented set:", augmented_set["target"].sum())
        print("number of positive labels in original set:", label.sum())
        augmented_set.to_csv(
            f"{path_results}augmented_set_{dataset_name}_{augmentation_method}.csv",
            index=False,
        )

    elif augmentation_method == "add_comparators":
        # Add comparator rows to the original dataset.
        original_df = dataset[FEATURE_COLS].astype(float).copy()
        original_df["id"] = dataset["id"].values
        original_df["target"] = label.values if hasattr(label, "values") else label

        id_to_target = original_df.set_index("id")["target"]

        # ---------- direction 1: unprivileged -> privileged ----------
        # Take rows originally in the unprivileged group, use the
        # do(protected=privileged) intervened features (already baked into
        # priv_comparator_data), and explicitly set the protected column to
        # the privileged value. Keep the original target.
        unpriv_mask = priv_comparator_data[prot_col].astype(float) != priv_val
        unpriv_rows = priv_comparator_data[unpriv_mask].copy()
        unpriv_rows[prot_col] = priv_val
        unpriv_rows["target"] = unpriv_rows["id"].map(id_to_target)
        unpriv_rows = unpriv_rows.reindex(columns=original_df.columns, fill_value=np.nan)

        # ---------- direction 2 (optional): privileged -> unprivileged ----------
        # Symmetric flip: take originally-privileged rows, keep their
        # features, set the protected column to unprivileged. We don't have
        # do(protected=unprivileged) intervened features, so this flip
        # preserves features (protected-attribute-only intervention).
        # Binary assumption: unpriv_val = 1 - priv_val.
        if AUG_COMPARATORS_BIDIRECTIONAL:
            unpriv_val_num = 1.0 - priv_val
            priv_mask   = priv_comparator_data[prot_col].astype(float) == priv_val
            priv_src    = dataset.loc[priv_mask.values].copy()
            priv_src[prot_col] = unpriv_val_num
            priv_src["target"] = priv_src["id"].map(id_to_target)
            priv_src = priv_src.reindex(columns=original_df.columns, fill_value=np.nan)
            comp_rows = pd.concat([unpriv_rows, priv_src], ignore_index=True)
            print(f"[add_comparators] BIDIRECTIONAL ({prot_col}): "
                  f"{len(unpriv_rows)} {unpriv_lbl}->{priv_lbl} + "
                  f"{len(priv_src)} {priv_lbl}->{unpriv_lbl} = {len(comp_rows)} total")
        else:
            comp_rows = unpriv_rows
            print(f"[add_comparators] one-sided ({prot_col}): "
                  f"{len(comp_rows)} {unpriv_lbl}->{priv_lbl}")

        X_aug = comp_rows[FEATURE_COLS].values
        y_aug = comp_rows["target"].values
        src_idx = comp_rows["id"].to_numpy(dtype=int) - 1  # ids are 1-based row positions

        augmented_set = pd.concat([original_df, comp_rows], ignore_index=True)
        print("number of positive labels in augmented set:", augmented_set["target"].sum())
        print("number of positive labels in original set:", label.sum())
        augmented_set.to_csv(
            f"{path_results}augmented_set_{priv_lbl}_{dataset_name}_{augmentation_method}.csv",
            index=False,
        )

    else:
        raise ValueError(
            f"main_MM: unknown augmentation_method={augmentation_method!r}. "
            "Expected 'update_labels' or 'add_comparators'.")

    if return_src_idx and augmentation_method == "add_comparators":
        return X_aug, y_aug, src_idx
    return X_aug, y_aug
