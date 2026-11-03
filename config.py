import os

from pipeline.datasets import get_dataset_spec

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

# Absolute path to the "New" parent directory (two levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = os.path.join(BASE_DIR, "data")
MIA_DIR    = os.path.join(BASE_DIR, "ensemble_mia-main")
OUTPUT_DIR = os.path.join(BASE_DIR, "pipeline_outputs")

# ---------------------------------------------------------------------------
# Active dataset — everything below this line is derived from _SPEC
# ---------------------------------------------------------------------------

ACTIVE_DATASET = os.environ.get("ABLATION_DATASET", "compas")
_SPEC = get_dataset_spec(ACTIVE_DATASET)
ACTIVE_DATASET_SPEC = _SPEC

DATASET_NAME = _SPEC.name.upper()

# Absolute path to the active dataset's raw CSV.
DATASET_CSV = os.path.join(BASE_DIR, _SPEC.csv_relpath)
# Back-compat alias: existing code (data_preparation.py) historically imported
# COMPAS_CSV by name. Points at whichever dataset is active.
COMPAS_CSV = DATASET_CSV

AUGMENTATION_METHOD = "add_comparators" # default method 'update_labels' # "update_labels", "add_comparators", "SCM"

# ---------------------------------------------------------------------------
# Dataset feature schema (all derived from the active DatasetSpec)
# ---------------------------------------------------------------------------

FEATURE_COLS = _SPEC.feature_cols
TARGET_COL   = _SPEC.target_col

# Protected attribute (binary — see pipeline/datasets/base.py docstring).
# Defaults to the active dataset's own DatasetSpec.protected_col. Override
# it here (or via the ABLATION_PROTECTED_COL env var, same resolution-order
# rule as ABLATION_DATASET above: set it BEFORE this module is imported) to
# switch to any of the dataset's other sensitive_cols WITHOUT editing that
# dataset's file, e.g. PROTECTED_COL_OVERRIDE = "race_enc" for COMPAS.
# This is the ONLY switch for which attribute is protected: every module
# that needs it (LDP, fairness metrics, AIA, the SCM, and cf_generation.py's
# update_labels/add_comparators) derives everything below from
# PROTECTED_COL/PROTECTED_PRIV/PROTECTED_UNPRIV/GROUP_PRIV_LABEL/
# GROUP_UNPRIV_LABEL, so there is nothing else to keep in sync.
PROTECTED_COL_OVERRIDE = os.environ.get("ABLATION_PROTECTED_COL") or None  # e.g. "race_enc" | "sex_enc" | None
PROTECTED_COL = PROTECTED_COL_OVERRIDE or _SPEC.protected_col
if PROTECTED_COL not in _SPEC.col_value_labels:
    raise ValueError(
        "config.PROTECTED_COL={!r} (from PROTECTED_COL_OVERRIDE={!r}) has no "
        "col_value_labels entry in dataset {!r}'s DatasetSpec -- it must be "
        "one of {!r}.".format(PROTECTED_COL, PROTECTED_COL_OVERRIDE,
                              _SPEC.name, list(_SPEC.col_value_labels)))

PROTECTED_PRIV     = _SPEC.priv_value(PROTECTED_COL)
PROTECTED_UNPRIV   = _SPEC.unpriv_value(PROTECTED_COL)
GROUP_PRIV_LABEL   = _SPEC.col_value_labels[PROTECTED_COL][PROTECTED_PRIV]
GROUP_UNPRIV_LABEL = _SPEC.col_value_labels[PROTECTED_COL][PROTECTED_UNPRIV]

# Back-compat alias: cf_generation.py historically imported RACE_ENC_COL as
# "the protected column name" (not necessarily literally about race).
RACE_ENC_COL = PROTECTED_COL

# Dependent columns the update_labels/add_comparators augmentation methods are
# allowed to intervene on (cf_generation.py::main_MM).
datasets_info = _SPEC.dependent_cols

# Feature indices nice_cf.py treats as categorical (vs. continuous) when
# generating NiCE counterfactual explanations.
CATEGORICAL_COLS = set(_SPEC.categorical_cols)

# Dependent columns for the 'SCM' (structural-causal) augmentation method.
# Empty => augmentation_method='SCM' is unavailable for the active dataset
# (cf_generation.py::fit_scm raises a clear NotImplementedError).
SCM_DEPENDENT_COLS = list(_SPEC.scm.dependent_cols) if _SPEC.scm is not None else []

# main_MM (cf_generation.py) indexes these three dicts by dataset name
# (DATASET_NAME.lower()) — only the active dataset needs an entry.
sensitive_columns = {_SPEC.name: list(_SPEC.sensitive_cols)}
male_values       = {_SPEC.name: _SPEC.male_value}
female_values     = {_SPEC.name: _SPEC.female_value}
white_values      = {_SPEC.name: _SPEC.white_value}

# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

RANDOM_STATE = int(os.environ.get("ABLATION_SEED", 3458))    # seed

TEST_SIZE    = 0.1  #10% test
VAL_SIZE     = 0  # 20% validation -- disjoint from X_train and X_test; used as
                     # the attribute-inference attack's attacker-training pool
                     # (data.X_val), never for fitting any target/classifier model.
# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------

MODEL_FAMILIES = ["logistic_regression", "random_forest", "xgboost"]

LR_PARAMS = {
    "random_state": RANDOM_STATE,
    "max_iter": 200,
    "C": 1.0,
    "solver": "lbfgs",
}
RF_PARAMS = {
    "random_state": RANDOM_STATE,
    "n_estimators": 200,
    "max_depth": 12,
}
XGB_PARAMS = {
    "random_state": RANDOM_STATE,
    "n_estimators": 200,
    "max_depth": 7,
    "learning_rate": 0.1,
    #"objective": 'multi:softprob',
    "eval_metric": "logloss",
}

# ---------------------------------------------------------------------------
# Local Differential Privacy (LDP)
# ---------------------------------------------------------------------------

LDP_EPSILON  = 10.0              # default legacy single-epsilon default (used by ldp.py)
LDP_EPSILONS = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # sweep values used by all pipelines (edit here)

# ---------------------------------------------------------------------------
# Augmentation quality controls
# ---------------------------------------------------------------------------
# These flags trade-off fairness improvement against accuracy loss.
# All default to False / None (= original behaviour, no change).

# update_labels: only relabel a non-white individual when their nearest white
# neighbour is within this percentile of all pairwise distances.
# None  → relabel everyone (original behaviour).
# 50    → only relabel the closest 50% of non-white individuals.
# Lower value = more conservative = less label noise = less accuracy loss.
AUG_RELABEL_DISTANCE_PERCENTILE = 25     # None | int 1-99

# update_labels: number of nearest white neighbours to consult when assigning a
# label to a non-white row. K=1 reproduces the original (single-NN) behaviour;
# K>1 averages over more neighbours so a single outlier cannot drive the label.
# Used together with AUG_RELABEL_AGREEMENT_THRESHOLD to gate relabelling on
# neighbour consensus.
AUG_RELABEL_K_NEIGHBORS = 11  #11              # int >= 1
# update_labels: only relabel a non-white row when at least this fraction of its
# K nearest white neighbours share the SAME label. Rows that fail the test keep
# their original label (no relabelling), which removes noisy / ambiguous matches
# without dropping the row entirely.
# None       → no agreement requirement (original single-NN behaviour).
# 0.6 .. 1.0 → require this fraction of K neighbours to agree on a label.
# Only meaningful when AUG_RELABEL_K_NEIGHBORS > 1 (single neighbour always
# agrees with itself).
AUG_RELABEL_AGREEMENT_THRESHOLD = 0.8   # None | float in (0.5, 1.0]

# add_comparators / SCM: drop synthetic CF rows that are identical to their
# source row (no feature changed). These add nothing but noise.
# False → keep all CFs (original behaviour).
AUG_DROP_IDENTICAL_CFS = True         # True | False

# add_comparators: when True, also generate the reverse direction
# (white -> non-white) so each side has matched counterparts. Original
# behaviour only added non-white -> white comparators, which leaves trees
# with one-sided evidence and limits CF-fairness gain.
# False → original (one-sided) behaviour.
# True  → symmetric: append both non-white→white AND white→non-white flips.
AUG_COMPARATORS_BIDIRECTIONAL = True   # True | False

# All augmented scenarios: down-weight synthetic rows relative to original rows.
# None  → equal weights (original behaviour, sklearn default).
# 0.5   → synthetic rows count half as much as original rows during training.
AUG_SYNTHETIC_WEIGHT = None           # None | float 0 < w < 1

# ---------------------------------------------------------------------------
# NiCE counterfactual generation
# ---------------------------------------------------------------------------

NICE_MAX_SAMPLES = 50    # max test-set points used as NiCE CF queries
                         # set to 50 for fast testing, 2000 for full runs

# ---------------------------------------------------------------------------
# Visualisation output
# ---------------------------------------------------------------------------

FIGURE_DPI  = 360
FIGURE_EXT  = "png"


# ---------------------------------------------------------------------------
# attack configuration
# ---------------------------------------------------------------------------
ATTACKS = ["aia"]  # "mia", "aia"
