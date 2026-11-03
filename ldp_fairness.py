"""
ldp_fairness.py -- Train fairlearn-constrained models on LDP-perturbed data.
"""
from __future__ import annotations
import os, sys, time, warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_PIPELINE_DIR)
for _p in [_ROOT_DIR, _PIPELINE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.config import (RANDOM_STATE, FEATURE_COLS, PROTECTED_COL,
    LR_PARAMS, RF_PARAMS, XGB_PARAMS, GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL,
    PROTECTED_PRIV, PROTECTED_UNPRIV, NICE_MAX_SAMPLES, LDP_EPSILONS)
from pipeline.models import ModelResult, ScenarioModels
from pipeline.fairness_metrics import compute_fairness
from pipeline.nice_cf import (generate_nice_cfs_for_all_models, NiCEResult,
                               save_nice_cf_results, load_nice_cf_results)
from pipeline.mia_analysis import run_mia_single
from pipeline.fairlearn_pipeline import (FairlearnWrapper, _build_base,
                                          _FAIRLEARN_AVAILABLE)

try:
    from fairlearn.reductions import (ExponentiatedGradient, GridSearch,
                                       DemographicParity, EqualizedOdds)
    from fairlearn.postprocessing import ThresholdOptimizer
    try:
        from fairlearn.reductions import TruePositiveRateParity as _TPR
    except ImportError:
        try:
            from fairlearn.reductions import EqualOpportunity as _TPR
        except ImportError:
            _TPR = None
except ImportError:
    pass

from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RACE_IDX = FEATURE_COLS.index(PROTECTED_COL)

# Constraints that reliably improve fairness on COMPAS
_CONSTRAINT_CONFIGS = {
    'eg_dp': ('EG-DemoParity', 'EG', lambda: DemographicParity()),
    'eg_eo': ('EG-EqOdds',     'EG', lambda: EqualizedOdds()),
    'to_dp': ('TO-DemoParity', 'TO', 'demographic_parity'),
    'to_eo': ('TO-EqOdds',     'TO', 'equalized_odds'),
}

_OUTDIR  = os.path.join(_ROOT_DIR, 'pipeline_outputs', 'ldp_fair')
_RES_DIR = os.path.join(_OUTDIR, 'results')


def scenario_key(eps: float, constraint: str, family: str) -> str:
    return 'ldp_fair_eps{}_{}'.format(eps, constraint) + '_{}'.format(family)


def scenario_label(eps: float, constraint_display: str, family: str) -> str:
    fam_label = {'lr': 'LR', 'rf': 'RF', 'xgb': 'XGB'}.get(family, family.upper())
    return 'LDP(ε={}) + {} ({})'.format(eps, constraint_display, fam_label)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ldp_fair_models(
    data,
    ldp_results: Dict,
    epsilons:    List[float],
    constraints: List[str] = ('eg_dp', 'to_dp'),
    families:    List[str] = ('lr', 'rf', 'xgb'),
    verbose:     bool = True,
) -> Dict[str, ModelResult]:
    """Train fairlearn-constrained models on LDP-perturbed training data.

    Parameters
    ----------
    data        : DataBundle from load_and_split().
    ldp_results : Dict mapping 'ldp_eps{e}' -> LDPResult (from apply_ldp_multi_eps).
    epsilons    : Which epsilon values to use (must be in ldp_results).
    constraints : Fairlearn constraint keys (subset of _CONSTRAINT_CONFIGS).
    families    : Base estimator families ('lr', 'rf', 'xgb').

    Returns
    -------
    Dict[scenario_key -> ModelResult]
    """
    if not _FAIRLEARN_AVAILABLE:
        raise RuntimeError('fairlearn is not installed.')

    if verbose:
        print('\n' + '='*60)
        print('  LDP + FAIRNESS COMBINED TRAINING  (COMPAS)')
        print('  epsilons={} | constraints={} | families={}'.format(
            epsilons, list(constraints), list(families)))
        print('='*60)

    X_te = data.X_test
    y_te = data.y_test
    results: Dict[str, ModelResult] = {}

    for eps in epsilons:
        ldp_key = 'ldp_eps{}'.format(eps)
        if ldp_key not in ldp_results:
            warnings.warn('[ldp_fair] No LDP result for epsilon={}'.format(eps))
            continue

        ldp = ldp_results[ldp_key]
        X_tr = ldp.X_train_ldp   # LDP-perturbed original data
        y_tr = ldp.y_train_ldp

        # Sensitive feature: race from LDP-perturbed original data
        race_tr = np.round(ldp.X_train_ldp[:, _RACE_IDX]).astype(int)

        if verbose:
            print('\n  === epsilon={} ({:,} training rows) ==='.format(
                eps, len(X_tr)))

        for fam in families:
            for ctag in constraints:
                if ctag not in _CONSTRAINT_CONFIGS:
                    warnings.warn('[ldp_fair] Unknown constraint: {}'.format(ctag))
                    continue

                cdisp, method, cobj_or_str = _CONSTRAINT_CONFIGS[ctag]
                key   = scenario_key(eps, ctag, fam)
                label = scenario_label(eps, cdisp, fam)

                if verbose:
                    print('  [{}+{}] {} ...'.format(method, fam.upper(), label),
                          end='', flush=True)
                t0 = time.perf_counter()

                try:
                    if method == 'EG':
                        cobj = cobj_or_str()
                        fm   = ExponentiatedGradient(
                            estimator=_build_base(fam),
                            constraints=cobj, eps=0.05, max_iter=50)
                        fm.fit(X_tr, y_tr, sensitive_features=race_tr)
                        wrapper = FairlearnWrapper(fm, key, _RACE_IDX)

                    elif method == 'TO':
                        base = _build_base(fam)
                        base.fit(X_tr, y_tr)
                        fm = ThresholdOptimizer(
                            estimator=base,
                            constraints=cobj_or_str,
                            predict_method='predict_proba',
                            objective='balanced_accuracy_score')
                        fm.fit(X_tr, y_tr, sensitive_features=race_tr)
                        wrapper = FairlearnWrapper(fm, key, _RACE_IDX,
                                                    base_estimator=base)
                    else:
                        raise ValueError('Unknown method: {}'.format(method))

                    yp   = wrapper.predict(X_te)
                    ypr  = wrapper.predict_proba(X_te)[:, 1]
                    mets = {
                        'accuracy':  accuracy_score(y_te, yp),
                        'auc_roc':   roc_auc_score(y_te, ypr),
                        'f1':        f1_score(y_te, yp, zero_division=0),
                        'precision': precision_score(y_te, yp, zero_division=0),
                        'recall':    recall_score(y_te, yp, zero_division=0),
                    }

                    mr = ModelResult(
                        name=key, family=fam, scenario='ldp_fair',
                        estimator=wrapper, y_pred=yp, y_proba=ypr,
                        metrics=mets, n_train=len(X_tr), feature_cols=FEATURE_COLS)
                    mr.label          = label
                    mr.constraint_tag = ctag
                    mr.epsilon        = eps
                    results[key] = mr

                    if verbose:
                        print(' AUC={:.3f}  Acc={:.3f}  ({:.1f}s)'.format(
                            mets['auc_roc'], mets['accuracy'],
                            time.perf_counter() - t0))

                except Exception as exc:
                    if verbose:
                        print(' FAILED: {}'.format(exc))
                    warnings.warn('[ldp_fair] {} failed: {}'.format(label, exc))

    return results


# ---------------------------------------------------------------------------
# Fairness evaluation
# ---------------------------------------------------------------------------

def run_fairness(ldp_fair_models: Dict, data, verbose=True) -> Dict:
    """Compute fairness metrics for all LDP+fairness combined models."""
    protected_test = data.X_test[:, _RACE_IDX]
    results = {}
    for key, mr in ldp_fair_models.items():
        fr = compute_fairness(
            model_result=mr,
            X_test=data.X_test,
            y_test=data.y_test,
            protected_test=protected_test,
            verbose=verbose)
        results[key] = fr
    return results


# ---------------------------------------------------------------------------
# NiCE + MIA
# ---------------------------------------------------------------------------

def run_nice(ldp_fair_models: Dict, data, max_samples: int = NICE_MAX_SAMPLES,
             verbose: bool = True) -> Dict:
    """Generate NiCE CFs for LDP+fairness models."""
    if not ldp_fair_models:
        return {}
    scenario_obj = ScenarioModels(scenario='ldp_fair', results=ldp_fair_models)
    return generate_nice_cfs_for_all_models(
        scenarios={'ldp_fair': scenario_obj},
        data=data, max_samples=max_samples, verbose=verbose)


def run_mia(ldp_fair_models: Dict, data, nice_results: Dict,
            verbose: bool = True) -> List[Dict]:
    """Run MIA using NiCE CFs as adversary proxy."""
    rng = np.random.default_rng(42)
    _fm  = data.X_train
    _fnm = data.X_test
    mem_eval = _fm[rng.choice(_fm.shape[0], min(500, _fm.shape[0]), replace=False)]
    non_mem  = _fnm[rng.choice(_fnm.shape[0], min(500, _fnm.shape[0]), replace=False)]
    ref = data.X_val if data.X_val.size > 0 else data.X_test
    all_rows   = []

    for key, mr in ldp_fair_models.items():
        label    = getattr(mr, 'label', key)
        nice_res = nice_results.get('ldp_fair', {}).get(key)
        if nice_res is None or len(getattr(nice_res, 'X_cf', [])) == 0:
            warnings.warn('[ldp_fair MIA] No NiCE CFs for {} -- skipping.'.format(label))
            continue

        mia_res  = run_mia_single(
            model_name=key, input_type='nice_cf',
            mem=mem_eval, non_mem=non_mem,
            synth=nice_res.X_cf, ref=ref, verbose=verbose)

        for att_name, att_res in mia_res.attacks.items():
            row = {
                'model':        key,
                'label':        label,
                'family':       getattr(mr, 'family', ''),
                'constraint':   getattr(mr, 'constraint_tag', ''),
                'epsilon':      getattr(mr, 'epsilon', float('nan')),
                'attacker':     att_name,
            }
            row.update(att_res.metrics)
            all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(ldp_fair_models: Dict, fairness_res: Dict,
                 nice_results: Dict, mia_rows: List[Dict],
                 verbose: bool = True) -> None:
    os.makedirs(_RES_DIR, exist_ok=True)

    # Performance
    pd.DataFrame([{
        'model':      k, 'label': getattr(v, 'label', k),
        'family':     getattr(v, 'family', ''),
        'constraint': getattr(v, 'constraint_tag', ''),
        'epsilon':    getattr(v, 'epsilon', float('nan')),
        **v.metrics,
    } for k, v in ldp_fair_models.items()]).to_csv(
        os.path.join(_RES_DIR, 'ldp_fair_model_metrics.csv'), index=False)

    # Fairness
    fair_rows = []
    for k, fr in fairness_res.items():
        mr = ldp_fair_models.get(k)
        row = {
            'model':      k,
            'label':      getattr(mr, 'label', k) if mr else k,
            'family':     getattr(mr, 'family', '') if mr else '',
            'constraint': getattr(mr, 'constraint_tag', '') if mr else '',
            'epsilon':    getattr(mr, 'epsilon', float('nan')) if mr else float('nan'),
        }
        row.update(fr.group)
        row.update(fr.individual)
        fair_rows.append(row)
    pd.DataFrame(fair_rows).to_csv(
        os.path.join(_RES_DIR, 'ldp_fair_fairness.csv'), index=False)

    # NiCE quality
    nice_rows = []
    for sc_n, sc_d in nice_results.items():
        for mn, nr in sc_d.items():
            if getattr(nr, 'metrics', {}):
                mr  = ldp_fair_models.get(mn)
                yq  = getattr(nr, 'y_query', np.array([]))
                yc  = getattr(nr, 'y_cf',    np.array([]))
                nice_rows.append({
                    'model':      mn,
                    'label':      getattr(mr, 'label', mn) if mr else mn,
                    'family':     getattr(mr, 'family', '') if mr else '',
                    'constraint': getattr(mr, 'constraint_tag', '') if mr else '',
                    'epsilon':    getattr(mr, 'epsilon', float('nan')) if mr else float('nan'),
                    'flip_rate':  float((yq != yc).mean())
                                  if len(yq) > 0 and len(yc) > 0 else float('nan'),
                    **nr.metrics,
                })
    pd.DataFrame(nice_rows).to_csv(
        os.path.join(_RES_DIR, 'ldp_fair_nice_quality.csv'), index=False)

    # NiCE CF arrays — persist X_cf/X_query/y_cf/y_query for reuse
    if nice_results:
        _cf_cache = os.path.join(_RES_DIR, 'nice_cf_arrays')
        save_nice_cf_results(nice_results, _cf_cache)

    # MIA
    if mia_rows:
        pd.DataFrame(mia_rows).to_csv(
            os.path.join(_RES_DIR, 'ldp_fair_mia.csv'), index=False)

    if verbose:
        print('  [ldp_fair] Results saved to {}'.format(_RES_DIR))


# ---------------------------------------------------------------------------
# Full runner (callable from unified_analysis.py)
# ---------------------------------------------------------------------------

def run_ldp_fairness_pipeline(
    data,
    ldp_results:  Dict,
    epsilons:     List[float] = LDP_EPSILONS,
    constraints:  List[str]   = ('eg_dp', 'to_dp'),
    families:     List[str]   = ('lr', 'rf', 'xgb'),
    nice_samples: int         = NICE_MAX_SAMPLES,
    skip_nice:    bool        = False,
    skip_mia:     bool        = False,
    verbose:      bool        = True,
) -> Tuple[Dict, Dict, Dict, List]:
    """End-to-end: train LDP+fair models, evaluate fairness, NiCE, MIA."""
    t0 = time.perf_counter()

    print('\n' + '='*60)
    print('  LDP + FAIRNESS PIPELINE')
    print('  epsilons={}  constraints={}  families={}'.format(
        list(epsilons), list(constraints), list(families)))
    print('  nice_samples={}  skip_nice={}  skip_mia={}'.format(
        nice_samples, skip_nice, skip_mia))
    print('='*60)

    ldp_fair_models = train_ldp_fair_models(
        data, ldp_results, list(epsilons),
        list(constraints), list(families), verbose=verbose)

    if not ldp_fair_models:
        print('  [ldp_fair] No models trained -- check epsilons/constraints/families.')
        return {}, {}, {}, []

    if verbose:
        print('\n  Performance summary:')
        print('  {:<45} {:>6} {:>6}'.format('Model', 'AUC', 'Acc'))
        print('  ' + '-'*60)
        for k, mr in ldp_fair_models.items():
            print('  {:<45} {:.3f} {:.3f}'.format(
                getattr(mr, 'label', k)[:44], mr.metrics['auc_roc'],
                mr.metrics['accuracy']))

    fairness_res = run_fairness(ldp_fair_models, data, verbose=verbose)

    _cf_cache = os.path.join(_RES_DIR, 'nice_cf_arrays')
    if skip_nice:
        nice_results = load_nice_cf_results(_cf_cache)
        if nice_results:
            print('\n  [ldp_fair] Loaded cached NiCE CFs ({} model sets).'.format(
                sum(len(v) for v in nice_results.values())))
        else:
            print('\n  [ldp_fair] skip_nice=True and no cached CFs found — skipping NiCE.')
    else:
        print('\n  [ldp_fair] NiCE CF generation (max_samples={})'.format(nice_samples))
        nice_results = run_nice(ldp_fair_models, data,
                                max_samples=nice_samples, verbose=verbose)

    if skip_mia or not nice_results:
        mia_rows = []
    else:
        print('\n  [ldp_fair] MIA analysis')
        mia_rows = run_mia(ldp_fair_models, data, nice_results, verbose=verbose)

    save_results(ldp_fair_models, fairness_res, nice_results, mia_rows,
                 verbose=verbose)

    print('\n  [ldp_fair] Pipeline complete ({:.1f}s)  {} models'.format(
        time.perf_counter() - t0, len(ldp_fair_models)))

    return ldp_fair_models, fairness_res, nice_results, mia_rows
