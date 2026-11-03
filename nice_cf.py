from __future__ import annotations
import os, sys, warnings, json
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import FEATURE_COLS, NICE_MAX_SAMPLES, RANDOM_STATE, CATEGORICAL_COLS

try:
    from nice import NICE as _NICELibrary
    _NICE_AVAILABLE = True
except ImportError:
    _NICE_AVAILABLE = False
    warnings.warn("NICEx not installed. pip install NICEx", ImportWarning)

# Categorical vs. continuous feature split, driven by the active dataset's
# DatasetSpec.categorical_cols (pipeline/datasets/<name>.py) -- for COMPAS
# this is {c_charge_degree_enc, sex_enc, race_enc}, matching the historical
# hardcoded set.
_CAT_FEAT = [i for i, c in enumerate(FEATURE_COLS) if c in CATEGORICAL_COLS]
_NUM_FEAT = [i for i in range(len(FEATURE_COLS)) if i not in set(_CAT_FEAT)]


@dataclass
class NiCEResult:
    X_query:    np.ndarray
    X_cf:       np.ndarray
    y_query:    np.ndarray
    y_cf:       np.ndarray
    metrics:    Dict[str,float] = field(default_factory=dict)
    model_name: str = ""


def _make_predict_fn(estimator):
    """Build a predict_proba wrapper. Models in this pipeline are trained
    directly on raw features, so no scaling is applied here."""
    def predict_fn(X):
        X = np.asarray(X, dtype=float)
        return estimator.predict_proba(X)
    return predict_fn


def _compute_quality(X_query, X_cf, X_train, random_state=RANDOM_STATE):
    rng = np.ptp(X_train, axis=0); rng[rng==0] = 1.0
    diff = np.abs(X_cf - X_query)
    l1 = float((diff / rng).sum(axis=1).mean())
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=random_state)
    iso.fit(X_train)
    pla = float((iso.predict(X_cf)==1).mean())
    spa = float((diff < 1e-6).astype(float).mean(axis=1).mean())
    return {"proximity": l1, "plausibility": pla, "sparsity": spa}


def _cf_dir(out_dir: str, scenario: str, family: str) -> str:
    """Return the directory for one (scenario, family) pair."""
    safe = scenario.replace("/", "_").replace("\\", "_")
    return os.path.join(out_dir, "{}_{}".format(safe, family))


def save_nice_cf_results(results: dict, out_dir: str) -> None:
    """Persist all NiCEResult arrays to disk.

    Layout::

        out_dir/
          <scenario>_<family>/
            X_query.npy
            X_cf.npy
            y_query.npy
            y_cf.npy
            meta.json   # scenario, family, model_name, quality metrics
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for sc, families in results.items():
        for fam, nr in families.items():
            d = _cf_dir(out_dir, sc, fam)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "X_query.npy"), nr.X_query)
            np.save(os.path.join(d, "X_cf.npy"),    nr.X_cf)
            np.save(os.path.join(d, "y_query.npy"), nr.y_query)
            np.save(os.path.join(d, "y_cf.npy"),    nr.y_cf)
            # Store scenario + family explicitly so load doesn't need to parse dir name
            meta = {
                "scenario":   sc,
                "family":     fam,
                "model_name": nr.model_name,
                "metrics":    nr.metrics,
            }
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            saved += 1
    print("  [nice_cf] Saved {} (scenario, family) CF sets -> {}".format(saved, out_dir))


def load_nice_cf_results(out_dir: str) -> dict:
    """Reload NiCEResult objects saved by save_nice_cf_results.

    Returns a dict with the same structure as generate_nice_cfs_for_all_models:
      {scenario: {family: NiCEResult}}
    Returns an empty dict if out_dir does not exist or is empty.

    Reads scenario and family from meta.json (written by save_nice_cf_results),
    so directory names do not need to be parsed — works for all pipelines
    regardless of whether they use short ('lr') or long ('logistic_regression')
    family keys.
    """
    if not os.path.isdir(out_dir):
        return {}
    results: dict = {}
    for entry in sorted(os.listdir(out_dir)):
        d = os.path.join(out_dir, entry)
        if not os.path.isdir(d):
            continue
        meta_path = os.path.join(d, "meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            X_query = np.load(os.path.join(d, "X_query.npy"))
            X_cf    = np.load(os.path.join(d, "X_cf.npy"))
            y_query = np.load(os.path.join(d, "y_query.npy"))
            y_cf    = np.load(os.path.join(d, "y_cf.npy"))
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            warnings.warn("  [nice_cf] Failed to load {}: {}".format(d, e))
            continue

        # Prefer explicit scenario/family stored in meta.json (new format).
        # Fall back to parsing the directory name for legacy saves.
        sc  = meta.get("scenario")
        fam = meta.get("family")
        if sc is None or fam is None:
            # Legacy: try to parse directory name as <scenario>_<family>
            parsed = False
            for known in ("logistic_regression", "random_forest", "xgboost",
                          "lr", "rf", "xgb"):
                if entry.endswith("_" + known):
                    fam = known
                    sc  = entry[: -(len(known) + 1)]
                    parsed = True
                    break
            if not parsed:
                warnings.warn(
                    "  [nice_cf] Could not parse scenario/family from dir: {}".format(entry))
                continue

        results.setdefault(sc, {})[fam] = NiCEResult(
            X_query=X_query, X_cf=X_cf,
            y_query=y_query, y_cf=y_cf,
            metrics=meta.get("metrics", {}),
            model_name=meta.get("model_name", ""),
        )
    n = sum(len(v) for v in results.values())
    if n:
        print("  [nice_cf] Loaded {} (scenario, family) CF sets from {}".format(n, out_dir))
    return results


def generate_nice_cfs_for_all_models(scenarios, data, max_samples=NICE_MAX_SAMPLES, verbose=True):
    if not _NICE_AVAILABLE:
        warnings.warn("NICEx not installed"); return {}
    if verbose:
        print("\n" + "="*60)
        print("  NiCE CF GENERATION  (NICEx library, COMPAS)")
        print(f"  cat_feat={_CAT_FEAT}  num_feat={_NUM_FEAT}")
        print("="*60)

    # X_train is kept as the NiCE background set (the set the algorithm searches
    # over to find counterfactuals) and for isolation-forest plausibility scoring.
    # Query points come from X_test so that we explain predictions on held-out
    # data — this gives a fair, scenario-independent evaluation since the test
    # set is identical across all trained scenarios.
    X_train = data.X_train; y_train = data.y_train
    X_test  = data.X_test

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = (rng.choice(len(X_test), size=max_samples, replace=False)
                  if (max_samples and len(X_test) > max_samples)
                  else np.arange(len(X_test)))
    X_query = X_test[sample_idx]
    n_query = len(X_query)

    all_results = {}
    for sc_name, sm in scenarios.items():
        all_results[sc_name] = {}
        for family, model_res in sm.results.items():
            if verbose: print(f"[NiCE] {sc_name}/{family}  ({n_query:,} queries from test set)")
            pf = _make_predict_fn(model_res.estimator)
            try:
                # NiCE uses X_train as background — the CF search space.
                nm = _NICELibrary(predict_fn=pf, X_train=X_train.copy().astype(float),
                                  cat_feat=_CAT_FEAT, num_feat=_NUM_FEAT,
                                  y_train=y_train.copy(), optimization="proximity",
                                  justified_cf=True, distance_metric="HEOM",
                                  num_normalization="minmax")
            except Exception as e:
                warnings.warn(f"NICE init failed for {sc_name}/{family}: {e}")
                all_results[sc_name][family] = NiCEResult(
                    X_query=X_query, X_cf=np.empty_like(X_query),
                    y_query=np.array([]), y_cf=np.array([]),
                    metrics={}, model_name=model_res.name)
                continue
            cfs = []; valid_idx = []
            if verbose: print("  Generating ... ", end="", flush=True)
            for i, x in enumerate(X_query):
                try:
                    cf = nm.explain(x.reshape(1,-1).astype(float))
                    if cf is not None and cf.shape==(1, len(FEATURE_COLS)):
                        cfs.append(cf[0]); valid_idx.append(i)
                except Exception: pass
                if verbose and (i+1) % max(1, n_query//10)==0:
                    print(f"{(i+1)*100//n_query}%.. ", end="", flush=True)
            if verbose: print("done.")
            if not cfs:
                warnings.warn(f"No CFs for {sc_name}/{family}")
                all_results[sc_name][family] = NiCEResult(
                    X_query=X_query, X_cf=np.empty_like(X_query),
                    y_query=np.array([]), y_cf=np.array([]),
                    metrics={}, model_name=model_res.name)
                continue
            Xq_valid = X_query[valid_idx]; X_cf = np.vstack(cfs)
            yq = model_res.estimator.predict(Xq_valid)
            yc = model_res.estimator.predict(X_cf)
            flip_rate = float((yq != yc).mean())
            # Plausibility scored against X_train distribution (isolation forest)
            metrics = _compute_quality(Xq_valid, X_cf, X_train)
            if verbose:
                print("  {:,}/{:,} CFs  flip={:.1f}%  prox={:.4f}  pla={:.3f}  spa={:.3f}".format(
                    len(cfs), n_query, flip_rate*100,
                    metrics["proximity"], metrics["plausibility"], metrics["sparsity"]))
            all_results[sc_name][family] = NiCEResult(
                X_query=Xq_valid, X_cf=X_cf,
                y_query=yq, y_cf=yc,
                metrics=metrics, model_name=model_res.name)
    return all_results
