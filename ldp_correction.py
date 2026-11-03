from __future__ import annotations

from typing import Union

import numpy as np

Target = Union[float, np.ndarray]  # do-intervention target: fixed value, or one per row


def flip_probability(epsilon: float) -> float:
    """LDP flip probability for binary randomized response at budget epsilon
    -- must match pipeline.ldp.randomised_response's mechanism exactly."""
    return 1.0 / (1.0 + np.exp(epsilon))


def fit_ols_naive(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain OLS of y on [1, X], whatever X is (noisy or clean). Returns
    [intercept, *slopes] in X's column order."""
    A = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def fit_ols_corrected_ldp(X_obs: np.ndarray, y: np.ndarray,
                          protected_idx: int, epsilon: float) -> np.ndarray:
    """
    LDP measurement-error correction for one binary regressor observed
    through known-flip-probability randomized response, alongside zero or
    more error-free covariates.

    Input
        X_obs         : (n, m) design matrix, m >= 1 columns, one of which
                        (index `protected_idx`) is the LDP-perturbed binary
                        protected attribute; the other m-1 columns (if any)
                        are measured without error.
        y             : (n,) dependent variable.
        protected_idx : which column of X_obs is the noisy one.
        epsilon       : the (known) LDP privacy budget used to perturb it.

    Output
        beta : (m+1,) array [intercept, slope_1, ..., slope_m], corrected
               for the attenuation bias the noisy column introduces in ALL
               coefficients it is correlated with (not just its own).

    Steps
        1. p <- flip_probability(epsilon); lambda <- 1 - 2p (reliability ratio).
        2. If |lambda| ~ 0 (epsilon ~ 0): correction is unidentified -- fall
           back to fit_ols_naive (step 0).
        3. pi_hat <- debiased base rate of the protected column:
              pi_hat = (mean(X_obs[:, protected_idx]) - p) / lambda
        4. Sigma_obs <- population covariance matrix of [X_obs, y] (m+1 x m+1).
        5. Build Sigma_true from Sigma_obs by undoing the attenuation:
             - every entry touching row/col `protected_idx` (except its own
               diagonal) <- divide by lambda   (Cov(s,Z) = lambda*Cov(x,Z))
             - the protected column's own diagonal entry <- pi_hat*(1-pi_hat)
               (exact Bernoulli variance of the recovered true proportion --
               NOT Var_obs/lambda^2, which would still carry the flip noise)
        6. Solve the OLS normal equations with Sigma_true for the slopes;
           recover the intercept from the (corrected) means.

    Falls back to fit_ols_naive at epsilon ~ 0, where the correction is not
    identified (dividing by lambda ~ 0 would blow up rather than mean
    anything) -- same convergence behaviour as the original single-covariate
    fit_scm_corrected_ldp.
    """
    p = flip_probability(epsilon)
    lam = 1.0 - 2.0 * p
    n, m = X_obs.shape

    if abs(lam) < 1e-9:                                                    # step 2
        return fit_ols_naive(X_obs, y)

    s = X_obs[:, protected_idx]
    pi_hat = (float(s.mean()) - p) / lam                                   # step 3
    pi_hat = min(max(pi_hat, 1e-3), 1 - 1e-3)

    combined = np.column_stack([X_obs, y])
    cov_obs = np.cov(combined, rowvar=False, ddof=0)                       # step 4

    cov_true = cov_obs.copy()                                              # step 5
    cov_true[protected_idx, :] /= lam
    cov_true[:, protected_idx] /= lam
    cov_true[protected_idx, protected_idx] = pi_hat * (1.0 - pi_hat)

    C_XX = cov_true[:m, :m]                                                # step 6
    C_Xy = cov_true[:m, m]
    slopes = np.linalg.solve(C_XX, C_Xy)

    mean_true = X_obs.mean(axis=0).copy()
    mean_true[protected_idx] = pi_hat
    intercept = float(y.mean() - mean_true @ slopes)
    return np.concatenate([[intercept], slopes])


def counterfactual_predictions(X_rows: np.ndarray, y: np.ndarray, beta: np.ndarray,
                               protected_idx: int, target: Target) -> np.ndarray:
    """Pearl abduction-action-prediction do-intervention, shared by every
    augmentation method: predict with the protected column set to `target`,
    add back each row's own fitted residual.

    `target` may be a scalar (every row intervened to the SAME fixed value
    -- main_MM's do_white/do_male convention) or a (n,) array (each row
    intervened to its OWN target, e.g. 1 - its current value -- the SCM's
    bidirectional convention, cf_generation.py::_apply_intervention). Both
    are the same operation; numpy broadcasting handles either transparently.
    """
    A = np.column_stack([np.ones(len(X_rows)), X_rows])
    resid = y - A @ beta
    X_cf = X_rows.copy()
    X_cf[:, protected_idx] = target
    A_cf = np.column_stack([np.ones(len(X_cf)), X_cf])
    return A_cf @ beta + resid
