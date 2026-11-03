import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import data_preparation as dp
from pipeline import models as mdl
from pipeline import fairness_metrics as fm
from pipeline import config as cfg
from pipeline import cf_generation as cfg_mod
from pipeline.config import FEATURE_COLS, PROTECTED_COL
from pipeline.unified_analysis import ALL_AUG_METHODS

print("Config: K=%s agree>=%s dist_pct=%s drop_id=%s bidir(default)=%s syn_w=%s" % (
    cfg.AUG_RELABEL_K_NEIGHBORS, cfg.AUG_RELABEL_AGREEMENT_THRESHOLD,
    cfg.AUG_RELABEL_DISTANCE_PERCENTILE, cfg.AUG_DROP_IDENTICAL_CFS,
    cfg.AUG_COMPARATORS_BIDIRECTIONAL, cfg.AUG_SYNTHETIC_WEIGHT))
print("Methods in loop:", ALL_AUG_METHODS)

data = dp.load_and_split(augmentation_method="add_comparators", verbose=False)
prot_idx  = FEATURE_COLS.index(PROTECTED_COL)
prot_test = data.X_test[:, prot_idx]
families  = ["logistic_regression", "random_forest", "xgboost"]
saved_bidir = cfg.AUG_COMPARATORS_BIDIRECTIONAL


def _augment(method):
    """Mirror the unified loop's virtual-method handling."""
    if method == "add_comparators_bidir":
        cfg.AUG_COMPARATORS_BIDIRECTIONAL = True
        cfg_mod.AUG_COMPARATORS_BIDIRECTIONAL = True
        real = "add_comparators"
    elif method == "add_comparators":
        cfg.AUG_COMPARATORS_BIDIRECTIONAL = False
        cfg_mod.AUG_COMPARATORS_BIDIRECTIONAL = False
        real = "add_comparators"
    else:
        real = method
    return dp.augment_data(data.X_train, data.y_train, real, verbose=False)


def _eval(label, X, y, sw):
    out = []
    for fam in families:
        sm = mdl.train_scenario(label, X, y, data.X_test, data.y_test,
                                 families=[fam], sample_weight=sw, verbose=False)
        mr = sm.results[fam]
        fr = fm.compute_fairness(model_result=mr, X_test=data.X_test,
                                  y_test=data.y_test, protected_test=prot_test,
                                  verbose=False)
        out.append((fam, label, mr.metrics.get("accuracy"), mr.metrics.get("auc_roc"),
                    fr.individual.get("cf_fairness"), fr.group.get("SPD")))
    return out


# Step 1 — baseline
rows = _eval("baseline", data.X_train, data.y_train, None)

# Step 2 — augmented loop (mirrors unified_analysis Step 2)
for method in ALL_AUG_METHODS:
    try:
        X_aug, y_aug, sw = _augment(method)
        rows += _eval(f"augmented_{method}", X_aug, y_aug, sw)
    except Exception as e:
        print(f"  [aug] {method} failed: {e}")

# restore flag
cfg.AUG_COMPARATORS_BIDIRECTIONAL = saved_bidir
cfg_mod.AUG_COMPARATORS_BIDIRECTIONAL = saved_bidir

# Step 3 — print summary table BEFORE Step 4 (fairlearn would go here)
print("\n" + "=" * 60)
print("  STEP 3 — Fairness Analysis: Baseline vs Augmented")
print("=" * 60)
print()
print("  {:<22}  {:<32}  {:>6}  {:>6}  {:>6}  {:>8}".format(
    "family", "scenario", "acc", "AUC", "CF", "SPD"))
print("  " + "-" * 96)
for r in rows:
    print("  {:<22}  {:<32}  {:>6.3f}  {:>6.3f}  {:>6.3f}  {:>+8.3f}".format(*r))

print("\n" + "=" * 60)
print("  STEP 4 — Fairlearn Baselines (skipped in this quick test)")
print("=" * 60)
