'''
fairlearn_pipeline.py -- Fair model training baselines (fairlearn) on COMPAS.
'''

from __future__ import annotations
import os, sys, time, warnings
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score)

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_PIPELINE_DIR)
for _p in [_ROOT_DIR, _PIPELINE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.config import (RANDOM_STATE, FEATURE_COLS, TARGET_COL,
    PROTECTED_COL, PROTECTED_PRIV, PROTECTED_UNPRIV,
    GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL, NICE_MAX_SAMPLES,
    LR_PARAMS, RF_PARAMS, XGB_PARAMS)
from pipeline.data_preparation import load_and_split
from pipeline.models import ModelResult, ScenarioModels
from pipeline.fairness_metrics import compute_fairness, FairnessResult
from pipeline.nice_cf import (generate_nice_cfs_for_all_models,
                               save_nice_cf_results, load_nice_cf_results)
from pipeline.mia_analysis import run_mia_single

# ---------------------------------------------------------------------------
# XGBoost -- optional
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier as _XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn('xgboost not installed -- XGB fairlearn models will be skipped.',
                  ImportWarning)

# ---------------------------------------------------------------------------
# fairlearn imports
# ---------------------------------------------------------------------------
_FAIRLEARN_AVAILABLE = False
try:
    from fairlearn.reductions import (ExponentiatedGradient, GridSearch,
                                       DemographicParity, EqualizedOdds)
    try:
        from fairlearn.reductions import TruePositiveRateParity as _TPR
    except ImportError:
        try:
            from fairlearn.reductions import EqualOpportunity as _TPR
        except ImportError:
            _TPR = None
    from fairlearn.postprocessing import ThresholdOptimizer
    _FAIRLEARN_AVAILABLE = True
except ImportError:
    warnings.warn('fairlearn not installed. Run: pip install fairlearn', ImportWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RACE_IDX = FEATURE_COLS.index(PROTECTED_COL)   # index of race_enc

_OUTDIR  = os.path.join(_ROOT_DIR, 'pipeline_outputs_fairlearn')
_RES_DIR = os.path.join(_OUTDIR, 'results')
_FIG_DIR = os.path.join(_OUTDIR, 'figures')

# Base-estimator families -- keyed by short tag
_FAMILY_TAGS = ['lr', 'rf', 'xgb']
_FAMILY_LABELS = {'lr': 'LR', 'rf': 'RF', 'xgb': 'XGB'}

# Constraint configurations -- (short_name, display_name, method_tag, constraint_or_str)
# constraint_or_str is filled in at runtime once fairlearn is imported.
_CONSTRAINT_DEFS = [
    # name_suffix,  display_suffix,   method_tag
    ('baseline',   'Baseline',        'NONE'),
    ('eg_dp',      'EG-DemoParity',   'EG'),
    ('eg_eo',      'EG-EqOdds',       'EG'),
    ('eg_tpr',     'EG-TPRParity',    'EG'),
    #('gs_dp',      'GS-DemoParity',   'GS'),
    #('gs_eo',      'GS-EqOdds',       'GS'),
    ('to_dp',      'TO-DemoParity',   'TO'),
    ('to_eo',      'TO-EqOdds',       'TO'),
]


def _model_key(constraint_tag: str, family_tag: str) -> str:
    """Canonical model key: e.g. 'eg_dp_lr', 'baseline_rf'."""
    return '{}_{}'.format(constraint_tag, family_tag)


def _model_label(constraint_display: str, family_tag: str) -> str:
    """Human-readable label: e.g. 'EG-DemoParity (LR)'."""
    return '{} ({})'.format(constraint_display, _FAMILY_LABELS[family_tag])


# ---------------------------------------------------------------------------
# Base estimator builder
# ---------------------------------------------------------------------------

def _build_base(family_tag: str) -> Any:
    """Return a fresh, unfitted base estimator for the given family."""
    if family_tag == 'lr':
        return LogisticRegression(**LR_PARAMS)
    elif family_tag == 'rf':
        return RandomForestClassifier(**RF_PARAMS)
    elif family_tag == 'xgb':
        if not _XGB_AVAILABLE:
            raise RuntimeError('xgboost not installed')
        params = {k: v for k, v in XGB_PARAMS.items()}
        return _XGBClassifier(**params)
    else:
        raise ValueError('Unknown family: {}'.format(family_tag))


# ---------------------------------------------------------------------------
# FairlearnWrapper
# ---------------------------------------------------------------------------

class FairlearnWrapper:
    '''sklearn-compatible wrapper for fairlearn estimators.

    ThresholdOptimizer: predict() uses fair thresholds via sensitive_features;
    predict_proba() delegates to the pre-fitted base estimator.

    ExponentiatedGradient / GridSearch: both delegate directly to the fairlearn
    model's predict / predict_proba (with weighted-ensemble fallback).
    '''

    def __init__(self, fair_model, name, race_idx, base_estimator=None):
        self._fair  = fair_model
        self.name   = name
        self._ridx  = race_idx
        self._base  = base_estimator
        self._is_to = 'ThresholdOptimizer' in type(fair_model).__name__

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self._is_to:
            sf  = np.round(X[:, self._ridx]).astype(int)
            out = self._fair.predict(X, sensitive_features=sf)
        else:
            # ExponentiatedGradient.predict is randomized; a fixed seed makes it
            # deterministic, so comparing predictions on x and a counterfactual
            # x' (CF fairness) reflects the change of x, not a fresh coin flip.
            try:
                out = self._fair.predict(X, random_state=RANDOM_STATE)
            except TypeError:
                out = self._fair.predict(X)
        return np.asarray(out.values if hasattr(out, 'values') else out)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if self._is_to:
            if self._base is not None:
                return self._base.predict_proba(X)
            preds = self.predict(X).astype(float)
            return np.column_stack([1.0 - preds, preds])
        # Try direct predict_proba first
        if hasattr(self._fair, 'predict_proba'):
            out = self._fair.predict_proba(X)
            return np.asarray(out.values if hasattr(out, 'values') else out)
        # Weighted-mixture fallback for EG (fairlearn 0.10)
        if hasattr(self._fair, 'predictors_') and hasattr(self._fair, 'weights_'):
            proba = sum(w * est.predict_proba(X)
                        for w, est in zip(self._fair.weights_, self._fair.predictors_))
            return np.asarray(proba)
        # Last resort
        preds = self.predict(X).astype(float)
        return np.column_stack([1.0 - preds, preds])

    def __repr__(self):
        return 'FairlearnWrapper({})'.format(self.name)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_fair_models(data, families=None, verbose=True):
    '''Train fair models for each (base-family x constraint) combination.

    Parameters
    ----------
    data     : DataBundle from load_and_split().
    families : List of family tags to run, e.g. ['lr', 'rf', 'xgb'].
               Defaults to all available families.
    verbose  : Print progress.

    Returns
    -------
    Dict[model_key -> ModelResult]
    '''
    if not _FAIRLEARN_AVAILABLE:
        raise RuntimeError('fairlearn is not installed.')

    if families is None:
        families = _FAMILY_TAGS if _XGB_AVAILABLE else ['lr', 'rf']

    if verbose:
        print('\n' + '='*60)
        print('  FAIR MODEL TRAINING  (fairlearn, COMPAS)')
        print('  Base families: {}  Constraints: {}'.format(
            families, [c[0] for c in _CONSTRAINT_DEFS]))
        print('='*60)

    X_tr = data.X_train;  y_tr = data.y_train
    X_te = data.X_test;   y_te = data.y_test
    race_tr = X_tr[:, _RACE_IDX].astype(int)

    results: Dict[str, ModelResult] = {}

    def _eval(estimator):
        yp  = estimator.predict(X_te)
        ypr = estimator.predict_proba(X_te)[:, 1]
        return yp, ypr, dict(
            accuracy =accuracy_score(y_te, yp),
            auc_roc  =roc_auc_score(y_te, ypr),
            f1       =f1_score(y_te, yp, zero_division=0),
            precision=precision_score(y_te, yp, zero_division=0),
            recall   =recall_score(y_te, yp, zero_division=0),
        )

    def _store(key, label, family_tag, constraint_tag, estimator):
        yp, ypr, m = _eval(estimator)
        results[key] = ModelResult(
            name=key, family=family_tag, scenario='fairlearn',
            estimator=estimator, y_pred=yp, y_proba=ypr,
            metrics=m, n_train=len(X_tr), feature_cols=FEATURE_COLS)
        results[key].label          = label           # extra attr for display
        results[key].constraint_tag = constraint_tag  # extra attr for grouping
        return m

    # Build runtime constraint objects
    def _make_constraints():
        constraints = {
            'baseline': None,
            'eg_dp':    DemographicParity(),
            'eg_eo':    EqualizedOdds(),
            'eg_tpr':   _TPR() if _TPR is not None else None,
            'gs_dp':    DemographicParity(difference_bound=0.05),
            'gs_eo':    EqualizedOdds(difference_bound=0.30),
            'to_dp':    'demographic_parity',
            'to_eo':    'equalized_odds',
        }
        return constraints

    for fam in families:
        if verbose:
            print('\n--- Base family: {} ---'.format(_FAMILY_LABELS[fam]))

        constraints = _make_constraints()   # fresh objects per family

        for (ctag, cdisp, method) in [(d[0], d[1], d[2]) for d in _CONSTRAINT_DEFS]:
            key   = _model_key(ctag, fam)
            label = _model_label(cdisp, fam)
            cobj  = constraints[ctag]

            if cobj is None and ctag != 'baseline':
                if verbose:
                    print('  [skip] {} (constraint unavailable)'.format(label))
                continue

            if verbose:
                print('  [{}] {} ...'.format(method if method != 'NONE' else 'BASE', label),
                      end='', flush=True)
            t0 = time.perf_counter()

            try:
                if method == 'NONE':
                    # Pure unfair baseline
                    base = _build_base(fam)
                    base.fit(X_tr, y_tr)
                    m = _store(key, label, fam, ctag, base)

                elif method == 'EG':
                    fm = ExponentiatedGradient(
                        estimator=_build_base(fam),
                        constraints=cobj, eps=0.05, max_iter=50)
                    fm.fit(X_tr, y_tr, sensitive_features=race_tr)
                    wrapper = FairlearnWrapper(fm, key, _RACE_IDX)
                    m = _store(key, label, fam, ctag, wrapper)

                elif method == 'GS':
                    fm = GridSearch(
                        estimator=_build_base(fam),
                        constraints=cobj, grid_size=30)
                    fm.fit(X_tr, y_tr, sensitive_features=race_tr)
                    wrapper = FairlearnWrapper(fm, key, _RACE_IDX)
                    m = _store(key, label, fam, ctag, wrapper)

                else:  # TO -- needs pre-fitted base
                    base = _build_base(fam)
                    base.fit(X_tr, y_tr)
                    fm = ThresholdOptimizer(
                        estimator=base, constraints=cobj,
                        predict_method='predict_proba',
                        objective='balanced_accuracy_score')
                    fm.fit(X_tr, y_tr, sensitive_features=race_tr)
                    wrapper = FairlearnWrapper(fm, key, _RACE_IDX,
                                               base_estimator=base)
                    m = _store(key, label, fam, ctag, wrapper)

                if verbose:
                    print(' AUC={:.3f}  Acc={:.3f}  ({:.1f}s)'.format(
                        m['auc_roc'], m['accuracy'], time.perf_counter() - t0))

            except Exception as exc:
                if verbose:
                    print(' FAILED: {}'.format(exc))
                warnings.warn('[fairlearn] {} failed: {}'.format(label, exc))

    return results


# ---------------------------------------------------------------------------
# Fairness analysis
# ---------------------------------------------------------------------------

def run_fairness(fair_models, data, verbose=True):
    '''Compute group + individual fairness metrics for all fair models.'''
    if verbose:
        print('\n' + '='*60)
        print('  FAIRNESS ANALYSIS  (protected: {} -- {} vs {})'.format(
            PROTECTED_COL, GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL))
        print('='*60)

    protected_test = data.X_test[:, _RACE_IDX]
    results = {}

    for name, model_res in fair_models.items():
        fr = compute_fairness(
            model_result=model_res,
            X_test=data.X_test,
            y_test=data.y_test,
            protected_test=protected_test,
            verbose=verbose)
        results[name] = fr
    return results


# ---------------------------------------------------------------------------
# NiCE CF generation
# ---------------------------------------------------------------------------

def run_nice(fair_models, data, max_samples=None, verbose=True):
    '''Generate NiCE CFs for all fair models (packaged as single scenario).'''
    if max_samples is None:
        max_samples = NICE_MAX_SAMPLES
    scenario_obj = ScenarioModels(scenario='fairlearn', results=fair_models)
    scenarios    = {'fairlearn': scenario_obj}
    return generate_nice_cfs_for_all_models(
        scenarios=scenarios, data=data,
        max_samples=max_samples, verbose=verbose)


# ---------------------------------------------------------------------------
# MIA (NiCE CFs as proxy, no MM-CFs)
# ---------------------------------------------------------------------------

def run_mia(fair_models, data, nice_results, verbose=True):
    '''Run MIA for each fair model; NiCE CFs serve as adversary proxy.'''
    if verbose:
        print('\n' + '='*60)
        print('  MEMBERSHIP INFERENCE ATTACK  (NiCE CFs as proxy)')
        print('='*60)
    rng = np.random.default_rng(42)
    _fm  = data.X_train
    _fnm = data.X_test
    mem_eval = _fm[rng.choice(_fm.shape[0], min(500, _fm.shape[0]), replace=False)]
    non_mem  = _fnm[rng.choice(_fnm.shape[0], min(500, _fnm.shape[0]), replace=False)]
    ref = data.X_val if data.X_val.size > 0 else data.X_test
    if verbose:
        print('  members={:,}  non-members={:,}  ref={:,}'.format(
            len(mem_eval), len(non_mem), len(ref)))
    all_rows = []
    for name, model_res in fair_models.items():
        label = getattr(model_res, 'label', name)
        if verbose:
            print('\n[MIA] {}'.format(label))
        nice_res = nice_results.get('fairlearn', {}).get(name)
        if nice_res is None or len(getattr(nice_res, 'X_cf', [])) == 0:
            warnings.warn('[MIA] No NiCE CFs for {} -- skipping.'.format(label))
            continue
        mia_res  = run_mia_single(
            model_name=name, input_type='nice_cf',
            mem=mem_eval, non_mem=non_mem,
            synth=nice_res.X_cf, ref=ref, verbose=verbose)
        for att_name, att_res in mia_res.attacks.items():
            row = {
                'model':        name,
                'model_label':  label,
                'family':       getattr(model_res, 'family', ''),
                'constraint':   getattr(model_res, 'constraint_tag', ''),
                'attacker':     att_name,
            }
            row.update(att_res.metrics)
            all_rows.append(row)
    return all_rows


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(fair_models, fairness_res, nice_results, mia_rows,
                 verbose=True):
    os.makedirs(_RES_DIR, exist_ok=True)

    # ---- Model performance ----
    perf_rows = []
    for n, mr in fair_models.items():
        row = {
            'model':      n,
            'label':      getattr(mr, 'label', n),
            'family':     getattr(mr, 'family', ''),
            'constraint': getattr(mr, 'constraint_tag', ''),
        }
        row.update(mr.metrics)
        perf_rows.append(row)
    pd.DataFrame(perf_rows).to_csv(
        os.path.join(_RES_DIR, 'fairlearn_model_metrics.csv'), index=False)

    # ---- Fairness metrics ----
    fair_rows = []
    for n, fr in fairness_res.items():
        mr = fair_models.get(n)
        row = {
            'model':      n,
            'label':      getattr(mr, 'label', n) if mr else n,
            'family':     getattr(mr, 'family', '') if mr else '',
            'constraint': getattr(mr, 'constraint_tag', '') if mr else '',
        }
        row.update(fr.group)
        row.update(fr.individual)
        fair_rows.append(row)
    pd.DataFrame(fair_rows).to_csv(
        os.path.join(_RES_DIR, 'fairlearn_fairness.csv'), index=False)

    # ---- NiCE quality ----
    nice_rows = []
    for sc_n, sc_d in nice_results.items():
        for mn, nr in sc_d.items():
            if getattr(nr, 'metrics', {}):
                mr  = fair_models.get(mn)
                yq  = getattr(nr, 'y_query', np.array([]))
                yc  = getattr(nr, 'y_cf',    np.array([]))
                r = {
                    'model':      mn,
                    'label':      getattr(mr, 'label', mn) if mr else mn,
                    'family':     getattr(mr, 'family', '') if mr else '',
                    'constraint': getattr(mr, 'constraint_tag', '') if mr else '',
                    'flip_rate':  float((yq != yc).mean())
                                  if len(yq) > 0 and len(yc) > 0 else float('nan'),
                }
                r.update(nr.metrics)
                nice_rows.append(r)
    pd.DataFrame(nice_rows).to_csv(
        os.path.join(_RES_DIR, 'fairlearn_nice_quality.csv'), index=False)

    # NiCE CF arrays — persist for reuse
    if nice_results:
        save_nice_cf_results(nice_results, os.path.join(_RES_DIR, 'nice_cf_arrays'))

    # ---- MIA ----
    if mia_rows:
        pd.DataFrame(mia_rows).to_csv(
            os.path.join(_RES_DIR, 'fairlearn_mia.csv'), index=False)

    if verbose:
        print('\n  [results] CSVs saved to {}'.format(_RES_DIR))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(fair_models, fairness_res, nice_results, mia_rows,
                     verbose=True):
    os.makedirs(_FIG_DIR, exist_ok=True)

    model_names  = list(fair_models.keys())
    labels       = [getattr(fair_models[n], 'label', n) for n in model_names]
    n_m          = len(labels)
    x            = np.arange(n_m)
    pl, ul       = GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL

    # Use a colormap that distinguishes families clearly
    # Assign color by family: LR=blue, RF=orange, XGB=green
    fam_colors = {'lr': '#4878CF', 'rf': '#F0A500', 'xgb': '#3DAA77'}
    colors = [fam_colors.get(getattr(fair_models[n], 'family', 'lr'), '#888888')
              for n in model_names]

    # ---- F1: Fairness metrics grid ----
    fair_keys = ['SPD', 'EOD', 'DI', 'EqOdds', 'AOD', 'PP']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        'Fairness Metrics -- fairlearn Models (COMPAS, {}: {} vs {})\n'
        'Blue=LR  Orange=RF  Green=XGB'.format(PROTECTED_COL, pl, ul), fontsize=12)
    for ax, key in zip(axes.flat[:6], fair_keys):
        vals = [fairness_res[n].group.get(key, float('nan')) for n in model_names]
        ax.bar(x, vals, color=colors)
        ax.axhline(0, color='k', lw=0.8, ls='--')
        if key == 'DI':
            ax.axhline(1.0, color='g', lw=1, ls=':', label='ideal')
            ax.axhline(0.8, color='orange', lw=1, ls=':', label='0.8-rule')
            ax.legend(fontsize=6)
        ax.set_title(key, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax7 = axes.flat[6]
    cf_fair = [fairness_res[n].individual.get('cf_fairness', float('nan'))
               for n in model_names]
    ax7.bar(x, cf_fair, color=colors)
    ax7.axhline(1, color='g', lw=1, ls=':', label='ideal=1')
    ax7.set_title('CF-Fairness', fontsize=10)
    ax7.set_xticks(x)
    ax7.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax7.set_ylim(0, 1.05)
    ax8 = axes.flat[7]
    consist = [fairness_res[n].individual.get('consistency', float('nan'))
               for n in model_names]
    ax8.bar(x, consist, color=colors)
    ax8.set_title('Consistency', fontsize=10)
    ax8.set_xticks(x)
    ax8.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax8.set_ylim(0, 1.05)
    # Family legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f.upper())
                       for f, c in fam_colors.items()]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=9,
               title='Base model')
    plt.tight_layout()
    p = os.path.join(_FIG_DIR, 'F1_fairness_comparison.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    if verbose: print('  [fig] Saved -> {}'.format(p))

    # ---- F2: Accuracy vs |SPD| trade-off (per family) ----
    aucs = [fair_models[n].metrics['auc_roc'] for n in model_names]
    spds = [abs(fairness_res[n].group['SPD'])  for n in model_names]
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, (lbl, auc, spd) in enumerate(zip(labels, aucs, spds)):
        c = colors[i]
        ax.scatter(spd, auc, color=c, s=120, zorder=3)
        ax.annotate(lbl, (spd, auc), textcoords='offset points',
                    xytext=(5, 3), fontsize=7)
    ax.set_xlabel('|SPD|  (lower = fairer)', fontsize=11)
    ax.set_ylabel('AUC-ROC', fontsize=11)
    ax.set_title('Accuracy vs Fairness Trade-off -- fairlearn Models\n'
                 'Blue=LR  Orange=RF  Green=XGB')
    ax.grid(True, alpha=0.3)
    fig.legend(handles=legend_elements, loc='lower left', fontsize=9,
               title='Base model')
    plt.tight_layout()
    p = os.path.join(_FIG_DIR, 'F2_accuracy_fairness_tradeoff.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    if verbose: print('  [fig] Saved -> {}'.format(p))

    # ---- F3: Positive rates by group ----
    pr_priv   = [fairness_res[n].group.get('PosRate_{}'.format(pl), float('nan'))
                 for n in model_names]
    pr_unpriv = [fairness_res[n].group.get('PosRate_{}'.format(ul), float('nan'))
                 for n in model_names]
    tpr_priv  = [fairness_res[n].group.get('TPR_{}'.format(pl), float('nan'))
                 for n in model_names]
    tpr_unpriv= [fairness_res[n].group.get('TPR_{}'.format(ul), float('nan'))
                 for n in model_names]
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(max(16, n_m * 0.7), 6))
    axes[0].bar(x - w/2, pr_priv,   w, label=pl,  color='#4878CF', alpha=0.85)
    axes[0].bar(x + w/2, pr_unpriv, w, label=ul,  color='#D65F5F', alpha=0.85)
    axes[0].set_ylabel('Positive prediction rate')
    axes[0].set_title('Positive Rates by {}'.format(PROTECTED_COL))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    axes[0].legend(); axes[0].set_ylim(0, 1.0)
    axes[1].bar(x - w/2, tpr_priv,   w, label=pl,  color='#4878CF', alpha=0.85)
    axes[1].bar(x + w/2, tpr_unpriv, w, label=ul,  color='#D65F5F', alpha=0.85)
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('TPR by {}'.format(PROTECTED_COL))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    axes[1].legend(); axes[1].set_ylim(0, 1.0)
    fig.suptitle('Group Rates -- fairlearn Models', fontsize=12)
    plt.tight_layout()
    p = os.path.join(_FIG_DIR, 'F3_group_rates.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    if verbose: print('  [fig] Saved -> {}'.format(p))

    # ---- F4: NiCE CF quality ----
    nice_sc  = nice_results.get('fairlearn', {})
    q_names  = [n for n in model_names
                if n in nice_sc and getattr(nice_sc.get(n), 'metrics', {})]
    if q_names:
        q_labels = [getattr(fair_models[n], 'label', n) for n in q_names]
        q_colors = [fam_colors.get(getattr(fair_models[n], 'family', 'lr'), '#888')
                    for n in q_names]

        def _flip(nr):
            yq = getattr(nr, 'y_query', np.array([]))
            yc = getattr(nr, 'y_cf',    np.array([]))
            return float((yq != yc).mean()) if len(yq) > 0 and len(yc) > 0 else float('nan')

        q_flip = [_flip(nice_sc[n]) for n in q_names]
        q_prox = [nice_sc[n].metrics.get('proximity',    float('nan')) for n in q_names]
        q_pla  = [nice_sc[n].metrics.get('plausibility', float('nan')) for n in q_names]
        q_spa  = [nice_sc[n].metrics.get('sparsity',     float('nan')) for n in q_names]
        xq     = np.arange(len(q_names))

        fig, axes = plt.subplots(1, 4, figsize=(max(20, len(q_names) * 0.9), 6))
        for ax, vals, title in zip(
                axes,
                [q_flip, q_prox, q_pla, q_spa],
                ['Flip Rate (↑ effective)',
                 'Proximity (↓ better)',
                 'Plausibility (↑ better)',
                 'Sparsity (↑ better)']):
            ax.bar(xq, vals, color=q_colors)
            ax.set_title(title, fontsize=10)
            ax.set_xticks(xq)
            ax.set_xticklabels(q_labels, rotation=45, ha='right', fontsize=6)
        fig.suptitle('NiCE CF Quality -- fairlearn Models\n'
                     'Blue=LR  Orange=RF  Green=XGB', fontsize=12)
        fig.legend(handles=legend_elements, loc='lower right', fontsize=9,
                   title='Base model')
        plt.tight_layout()
        p = os.path.join(_FIG_DIR, 'F4_nice_cf_quality.png')
        fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
        if verbose: print('  [fig] Saved -> {}'.format(p))

    # ---- F5: MIA privacy ----
    if mia_rows:
        mdf  = pd.DataFrame(mia_rows)
        best = mdf.loc[mdf.groupby('model')['auc_roc'].idxmax()]
        best = best.set_index('model').reindex(model_names).dropna(subset=['auc_roc'])
        bx   = np.arange(len(best))
        blab = [getattr(fair_models[n], 'label', n) for n in best.index]
        bc   = [fam_colors.get(getattr(fair_models.get(n, ModelResult.__new__(ModelResult)),
                                        'family', 'lr'), '#888') for n in best.index]
        fig, axes = plt.subplots(1, 2, figsize=(max(14, len(best) * 0.7), 6))
        axes[0].bar(bx, best['auc_roc'].values, color=bc)
        axes[0].axhline(0.5, color='red', ls='--', lw=1.2, label='random (0.5)')
        axes[0].set_title('MIA AUC-ROC (best attacker)')
        axes[0].set_ylim(0.44, 0.62)
        axes[0].legend(fontsize=8)
        axes[0].set_xticks(bx)
        axes[0].set_xticklabels(blab, rotation=45, ha='right', fontsize=7)
        axes[1].bar(bx, best['mia_advantage'].values, color=bc)
        axes[1].axhline(0, color='red', ls='--', lw=1.2, label='no advantage')
        axes[1].set_title('MIA Advantage (best attacker)')
        axes[1].legend(fontsize=8)
        axes[1].set_xticks(bx)
        axes[1].set_xticklabels(blab, rotation=45, ha='right', fontsize=7)
        fig.suptitle('MIA Privacy -- fairlearn Models (NiCE CF proxy)\n'
                     'Blue=LR  Orange=RF  Green=XGB', fontsize=12)
        fig.legend(handles=legend_elements, loc='lower right', fontsize=9,
                   title='Base model')
        plt.tight_layout()
        p = os.path.join(_FIG_DIR, 'F5_mia_privacy.png')
        fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
        if verbose: print('  [fig] Saved -> {}'.format(p))

    # ---- F6: Full fairness heatmap ----
    heat_keys = ['SPD', 'DI', 'EOD', 'AOD', 'EqOdds', 'PP', 'Theil',
                 'cf_fairness', 'consistency']
    heat_data = []
    for n in model_names:
        row = []
        for k in heat_keys:
            v = fairness_res[n].group.get(
                k, fairness_res[n].individual.get(k, float('nan')))
            row.append(v)
        heat_data.append(row)
    heat_arr = np.array(heat_data, dtype=float)
    fig, ax  = plt.subplots(figsize=(14, max(6, n_m * 0.45)))
    im = ax.imshow(heat_arr.T, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(n_m))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(heat_keys)))
    ax.set_yticklabels(heat_keys, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.02)
    ax.set_title('Fairness Metrics Heatmap -- fairlearn Models', fontsize=12)
    for i in range(n_m):
        for j in range(len(heat_keys)):
            v = heat_arr[i, j]
            if v == v:  # not nan
                ax.text(i, j, '{:.2f}'.format(v), ha='center', va='center',
                        fontsize=5.5, color='black')
    plt.tight_layout()
    p = os.path.join(_FIG_DIR, 'F6_fairness_heatmap.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    if verbose: print('  [fig] Saved -> {}'.format(p))

    # ---- F7: Per-family grouped bar (SPD + EOD) ----
    _families_present = []
    for fam in _FAMILY_TAGS:
        if any(getattr(fair_models[n], 'family', '') == fam for n in model_names):
            _families_present.append(fam)

    if len(_families_present) > 1:
        constraint_order = [d[0] for d in _CONSTRAINT_DEFS]
        for metric in ['SPD', 'EOD']:
            fig, axes = plt.subplots(1, len(_families_present),
                                      figsize=(6 * len(_families_present), 5),
                                      sharey=True)
            if len(_families_present) == 1:
                axes = [axes]
            for ax, fam in zip(axes, _families_present):
                fam_models = [n for n in model_names
                              if getattr(fair_models[n], 'family', '') == fam]
                fam_labels = [getattr(fair_models[n], 'label', n) for n in fam_models]
                vals = [fairness_res[n].group.get(metric, float('nan'))
                        for n in fam_models]
                xf = np.arange(len(fam_models))
                ax.bar(xf, vals,
                        color=fam_colors.get(fam, '#888'),
                        alpha=0.85, edgecolor='white', linewidth=0.5)
                ax.axhline(0, color='k', lw=0.8, ls='--')
                ax.set_title('{} ({})'.format(metric, _FAMILY_LABELS[fam]),
                              fontsize=10)
                ax.set_xticks(xf)
                ax.set_xticklabels(
                    [lb.replace(' ({})'.format(_FAMILY_LABELS[fam]), '')
                     for lb in fam_labels],
                    rotation=40, ha='right', fontsize=8)
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            fig.suptitle('{} by Method and Base Model'.format(metric), fontsize=12)
            plt.tight_layout()
            p = os.path.join(_FIG_DIR,
                              'F7_{}_by_family.png'.format(metric.lower()))
            fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
            if verbose: print('  [fig] Saved -> {}'.format(p))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_fairlearn_pipeline(skip_nice=False, skip_mia=False, families=None,
                            nice_samples=None, verbose=True, data=None):
    '''Full fairlearn pipeline: data -> fair models -> fairness -> NiCE -> MIA.

    Parameters
    ----------
    nice_samples : int or None
        Max samples for NiCE CF generation. None uses NICE_MAX_SAMPLES from config.
    data : DataBundle or None
        Pre-loaded DataBundle from load_and_split(). If None, data is loaded
        fresh. Pass this from unified_analysis to avoid reloading.
    '''
    t_start = time.perf_counter()

    if data is None:
        print('\n' + '='*60)
        print('  STEP 0 -- Data Loading & Splitting')
        print('='*60)
        data = load_and_split(verbose=verbose)
    else:
        print('\n  [fairlearn] Using pre-loaded DataBundle.')

    fair_models = train_fair_models(data, families=families, verbose=verbose)

    if verbose:
        print('\n  Performance summary:')
        print('  {:<35} {:>6} {:>6} {:>6}'.format('Model', 'AUC', 'Acc', 'F1'))
        print('  ' + '-'*55)
        for name, mr in fair_models.items():
            m = mr.metrics
            print('  {:<35} {:.3f} {:.3f} {:.3f}'.format(
                getattr(mr, 'label', name), m['auc_roc'],
                m['accuracy'], m['f1']))

    fairness_res = run_fairness(fair_models, data, verbose=verbose)

    _cf_cache = os.path.join(_RES_DIR, 'nice_cf_arrays')
    if skip_nice:
        nice_results = load_nice_cf_results(_cf_cache)
        if nice_results:
            print('\n[STEP 3] Loaded cached NiCE CFs ({} model sets).'.format(
                sum(len(v) for v in nice_results.values())))
        else:
            print('\n[STEP 3] Skipped NiCE CF generation (--skip-nice). No cached CFs found.')
    else:
        n_samples = nice_samples if nice_samples is not None else NICE_MAX_SAMPLES
        print('\n' + '='*60)
        print('  STEP 3 -- NiCE CF Generation  (max_samples={:,})'.format(n_samples))
        print('='*60)
        nice_results = run_nice(fair_models, data, max_samples=n_samples,
                                verbose=verbose)

    if skip_mia or not nice_results:
        print('\n[STEP 4] Skipped MIA.')
        mia_rows = []
    else:
        mia_rows = run_mia(fair_models, data, nice_results, verbose=verbose)

    save_results(fair_models, fairness_res, nice_results, mia_rows, verbose=verbose)

    print('\n' + '='*60)
    print('  STEP 6 -- Generating Figures')
    print('='*60)
    generate_figures(fair_models, fairness_res, nice_results, mia_rows,
                     verbose=verbose)

    elapsed = time.perf_counter() - t_start
    print('\n' + '='*60)
    print('  FAIRLEARN PIPELINE COMPLETE  ({:.1f}s)'.format(elapsed))
    print('='*60)
    print('  Output directory : {}'.format(_OUTDIR))
    print('  Results CSVs     : {}'.format(_RES_DIR))
    print('  Figures          : {}'.format(_FIG_DIR))
    print('  Models trained   : {}'.format(len(fair_models)))
    return fair_models, fairness_res, nice_results, mia_rows


# ---------------------------------------------------------------------------
# unified_analysis.py integration helpers
# ---------------------------------------------------------------------------

def load_fairlearn_outputs(base_dir=None):
    '''Load all saved fairlearn CSVs into a dict for unified_analysis.py.'''
    res_dir = base_dir or _RES_DIR
    out = {}
    for fname, key in [
        ('fairlearn_fairness.csv',      'fairness'),
        ('fairlearn_model_metrics.csv', 'model_metrics'),
        ('fairlearn_nice_quality.csv',  'nice_quality'),
        ('fairlearn_mia.csv',           'mia'),
    ]:
        path = os.path.join(res_dir, fname)
        if os.path.exists(path):
            out[key] = pd.read_csv(path)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
'''
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description='fairlearn fair-model pipeline for COMPAS (multi-family)')
    ap.add_argument('--skip-nice',  action='store_true',
                    help='Skip NiCE CF generation')
    ap.add_argument('--skip-mia',   action='store_true',
                    help='Skip MIA analysis')
    ap.add_argument('--families',     nargs='+', default=None,
                    choices=['lr', 'rf', 'xgb'],
                    help='Base estimator families to test (default: all available)')
    ap.add_argument('--nice-samples', type=int, default=None,
                    help='Max NiCE query samples per model (default: NICE_MAX_SAMPLES '
                         'from config). Use e.g. 300 for a fast test run.')
    args = ap.parse_args()
    run_fairlearn_pipeline(
        skip_nice=args.skip_nice,
        skip_mia=args.skip_mia,
        families=args.families,
        nice_samples=args.nice_samples,
    )
'''
