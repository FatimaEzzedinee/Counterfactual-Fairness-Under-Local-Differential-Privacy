"""
cf_summary.py -- print the counterfactual-fairness metrics (mean (std) over
seeds) of the proposed methods and the Kusner et al. baselines, no privacy.

Reads the 'paper_all_models' sheet of every
<run_root>/multi_seed_<dataset>_*/final_results_mean_std.xlsx written by
run_multi_seed.py.

Usage:
    python3 pipeline/cf_summary.py <run_root>
"""

import glob
import os
import sys

import pandas as pd

# (aug_method / method id, display name, group)
METHODS = [
    ("baseline",              "Baseline (full)",         "baseline"),
    ("SCM",                   "SCM aug",                 "ours"),
    ("update_labels",         "Update labels",           "ours"),
    ("add_comparators",       "Add comparators",         "ours"),
    ("add_comparators_bidir", "Add comparators (bidir)", "ours"),
    ("unaware",               "Unaware",                 "Kusner"),
    ("fair_add",              "Fair Add",                "Kusner"),
]
METRICS = [
    ("cf_scm_consistency",  "CF cons.^"),
    ("cf_scm_gap_mean",     "Delta_f v"),
    ("cf_flip_consistency", "Flip cons.^"),
    ("cf_flip_gap_mean",    "Delta_flip v"),
]
NO_PRIVACY = {"no_privacy_no_fairness", "fairness_causal_only"}


def summarize(xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name="paper_all_models")
    df = df[df["category"].isin(NO_PRIVACY)]
    rows = []
    for fam in sorted(df["family"].unique()):
        for mid, name, group in METHODS:
            r = df[(df["family"] == fam) & (df["method"] == mid)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"model": fam, "method": name, "group": group,
                         **{label: r.get(col, "nan") for col, label in METRICS}})
    return pd.DataFrame(rows)


def main(run_root: str) -> None:
    files = sorted(glob.glob(os.path.join(run_root, "multi_seed_*", "final_results_mean_std.xlsx")))
    if not files:
        sys.exit("No final_results_mean_std.xlsx found under {}".format(run_root))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    for f in files:
        dataset = os.path.basename(os.path.dirname(f)).split("multi_seed_", 1)[1].rsplit("_", 2)[0]
        print("\n" + "=" * 100)
        print("  {}   (mean (std) over seeds, no privacy; ^ higher is better, v lower is better)".format(
            dataset.upper()))
        print("=" * 100)
        print(summarize(f).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
