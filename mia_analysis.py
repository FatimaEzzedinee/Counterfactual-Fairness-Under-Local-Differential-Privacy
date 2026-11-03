from __future__ import annotations
import importlib.util, os, sys, time, warnings
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np, pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

_NEW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MIA_DIR = os.path.join(_NEW_DIR, "ensemble_mia-main")
for _p in [_MIA_DIR, _NEW_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from pipeline.config import RANDOM_STATE, FEATURE_COLS

_MIAEvaluator = None
_EVALUATOR_AVAILABLE = False
try:
    _spec = importlib.util.spec_from_file_location(
        "utils_evaluator_mia",
        os.path.join(_MIA_DIR, "utils", "utils_evaluator.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _MIAEvaluator = _mod.MIAEvaluator
    _EVALUATOR_AVAILABLE = True
except Exception as _e:
    warnings.warn(f"Could not load MIAEvaluator: {_e}", ImportWarning)
try:
    from synth_mia.evaluation import AttackEvaluator
    _ATTACK_EVALUATOR_AVAILABLE = True
except Exception:
    _ATTACK_EVALUATOR_AVAILABLE = False


@dataclass
class AttackResult:
    attacker_name: str; model_name: str; input_type: str
    metrics:   Dict[str, float] = field(default_factory=dict)
    fpr_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr_curve: np.ndarray = field(default_factory=lambda: np.array([]))

@dataclass
class MIAResult:
    model_name: str; input_type: str
    attacks:    Dict[str, AttackResult] = field(default_factory=dict)
    score_df:   Optional[pd.DataFrame] = None
    true_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    runtime_sec: float = 0.0


def _compute_metrics(true_labels, scores):
    scores = np.asarray(scores, dtype=float)
    if np.isnan(scores).all():
        return {}, np.array([]), np.array([])
    if _ATTACK_EVALUATOR_AVAILABLE:
        ev = AttackEvaluator(true_labels, scores)
        m  = ev.roc_metrics(); m.update(ev.classification_metrics()); m.update(ev.privacy_metrics())
        m["pr_auc"] = float(average_precision_score(true_labels, scores))
    else:
        th = np.median(scores); yp = (scores >= th).astype(int)
        tp=int(((true_labels==1)&(yp==1)).sum()); fp=int(((true_labels==0)&(yp==1)).sum())
        tn=int(((true_labels==0)&(yp==0)).sum()); fn=int(((true_labels==1)&(yp==0)).sum())
        tpr=tp/(tp+fn) if (tp+fn)>0 else 0.0; fv=fp/(fp+tn) if (fp+tn)>0 else 0.0
        fc, tc, _ = roc_curve(true_labels, scores)
        tpr_at = {f"tpr_at_fpr_{t}": float(np.interp(t,fc,tc)) for t in [0.0,0.001,0.01,0.1]}
        m = {"auc_roc":float(roc_auc_score(true_labels,scores)),
             "pr_auc":float(average_precision_score(true_labels,scores)),
             "accuracy":float(accuracy_score(true_labels,yp)),
             "precision":float(precision_score(true_labels,yp,zero_division=0)),
             "recall":float(recall_score(true_labels,yp,zero_division=0)),
             "f1":float(f1_score(true_labels,yp,zero_division=0)),
             "true_positive_rate":float(tpr),"false_positive_rate":float(fv),
             "mia_advantage":float(tpr-fv),"privacy_gain":float(1.0-(tpr-fv)),**tpr_at}
    fc, tc, _ = roc_curve(true_labels, scores)
    return m, fc, tc


def run_mia_single(model_name, input_type, mem, non_mem, synth, ref, verbose=True):
    if not _EVALUATOR_AVAILABLE:
        warnings.warn("MIAEvaluator not available"); return MIAResult(model_name=model_name, input_type=input_type)
    if verbose:
        print(f"  [MIA] {model_name} | {input_type} | mem={len(mem):,} non_mem={len(non_mem):,} synth={len(synth):,}")
    ev = _MIAEvaluator(); t0 = time.perf_counter()
    try:
        score_df, true_labels = ev._run_individual_attacks(mem, non_mem, synth, ref)
    except Exception as e:
        warnings.warn(f"[MIA] _run_individual_attacks failed: {e}")
        return MIAResult(model_name=model_name, input_type=input_type)
    try:
        score_df = ev._create_ensemble_methods(score_df)
    except Exception as e:
        warnings.warn(f"[MIA] _create_ensemble_methods failed: {e}")

    result = MIAResult(model_name=model_name, input_type=input_type,
                       score_df=score_df, true_labels=true_labels, runtime_sec=time.perf_counter()-t0)
    
    for col in score_df.columns:
        cs = pd.to_numeric(score_df[col], errors="coerce").values
        if np.isnan(cs).all(): continue
        try: mets, fc, tc = _compute_metrics(true_labels, cs)
        except Exception as e: warnings.warn(f"[MIA] metrics for {col} failed: {e}"); continue
        result.attacks[col] = AttackResult(
            attacker_name=col, model_name=model_name, input_type=input_type,
            metrics=mets, fpr_curve=fc, tpr_curve=tc)
        
    if verbose and result.attacks:
        best = max(result.attacks.values(), key=lambda r: r.metrics.get("auc_roc",0), default=None)
        if best:
            auc=best.metrics.get("auc_roc",0); adv=best.metrics.get("mia_advantage",float("nan"))
            print(f"  [MIA] Best: {best.attacker_name}  AUC={auc:.3f}  Adv={adv:+.3f}  ({result.runtime_sec:.1f}s)")
    return result


def run_mia_analysis(scenarios, data, nice_cf_results, verbose=True):
    """Membership-Inference Attack driven by NiCE counterfactuals only.

    Setup
    -----
    * mem        = a sample of training rows  (members of the training set)
    * non_mem    = a sample of test rows      (non-members)
    * synth      = NiCE-generated counterfactuals for each (scenario, family);
                   queries are drawn from X_test in nice_cf.py, so synth is
                   derived from non-member data and probes what the trained
                   model leaks about its members.
    * ref        = validation set if available, otherwise X_test (background
                   distribution for the attack evaluator).

    Only NiCE CFs are used as the synth input; the older mm_cf path
    (race-flipped SCM CFs from data preparation) is intentionally disabled.
    """
    if verbose:
        print("\n" + "="*60)
        print("  MEMBERSHIP INFERENCE ATTACK  (COMPAS — NiCE CFs only)")
        print("="*60)
    rng = np.random.default_rng(42)
    _full_mem     = data.X_train
    _full_non_mem = data.X_test
    mem_eval     = _full_mem[rng.choice(_full_mem.shape[0],
                                        min(500, _full_mem.shape[0]), replace=False)]
    non_mem_eval = _full_non_mem[rng.choice(_full_non_mem.shape[0],
                                            min(500, _full_non_mem.shape[0]), replace=False)]
    ref = data.X_val if data.X_val.size > 0 else data.X_test

    if verbose:
        print(f"  members={len(mem_eval):,}  non-members={len(non_mem_eval):,}  "
              f"ref={len(ref):,}")
    all_results = {}
    for sc_name, sm in scenarios.items():
        all_results[sc_name] = {}
        for family, model_res in sm.results.items():
            all_results[sc_name][family] = {}
            if verbose:
                print(f"\n[MIA] Scenario={sc_name}  Family={family}")

            nice_res = nice_cf_results.get(sc_name, {}).get(family)
            if nice_res is not None and len(nice_res.X_cf) > 0:
                all_results[sc_name][family]["nice_cf"] = run_mia_single(
                    model_res.name, "nice_cf",
                    mem_eval, non_mem_eval, nice_res.X_cf, ref, verbose)
            else:
                warnings.warn(f"No NiCE CFs for {sc_name}/{family} -- skipping MIA")
                all_results[sc_name][family]["nice_cf"] = MIAResult(
                    model_res.name, "nice_cf")
    return all_results


def mia_results_to_dataframe(all_results):
    rows=[]
    for sc,families in all_results.items():
        for fam,inputs in families.items():
            for itype,mia_res in inputs.items():
                for att_name,att_res in mia_res.attacks.items():
                    row={"scenario":sc,"family":fam,"input_type":itype,"attacker":att_name}
                    row.update(att_res.metrics); rows.append(row)
    return pd.DataFrame(rows)