from typing import Dict

import numpy as np
from scipy.optimize import minimize

from .pmf import (
    ensure_consistent_component_ordering,
    fit_mixture_direct,
    log_beta_binomial_pmf,
)
from .utils import chi_square_goodness_of_fit


# -------------------------------------------------------------------
# 2) EM Optimization for Beta-Binomial mixture
# -------------------------------------------------------------------
def em_mixture_beta_binomial(
    counts: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int = 42,
    n_restarts: int = 2,
) -> Dict[str, float]:
    """
    Fit a two-component mixture of Beta-Binomial distributions with
    multiple restarts and optimized computation.

    Args:
        counts: Array of observed counts
        k: Number of trials
        max_iter: Maximum number of EM iterations
        tol: Convergence tolerance for log-likelihood
        random_state: Random seed for initialization
        n_restarts: Number of random restarts

    Returns:
        Dict: Fitted model parameters including weights and alpha/beta values
    """
    rng = np.random.default_rng(random_state)

    # Filter out invalid counts
    valid_mask = (counts >= 0) & (counts <= k)
    if not np.all(valid_mask):
        print(
            f"Warning: Filtered out {np.sum(~valid_mask)} invalid counts outside [0, {k}]"
        )
        counts = counts[valid_mask]

    if len(counts) == 0:
        raise ValueError("No valid counts found in input data")

    # For efficiency, work with unique counts and their frequencies
    unique_counts, count_freq = np.unique(counts, return_counts=True)

    best_result = None
    best_ll = float("-inf")

    # Try multiple random initializations
    for restart in range(n_restarts):
        # 1) Initialization
        w = 0.5
        alpha1, beta1 = 1.0 + 2 * rng.random(), 1.0 + 2 * rng.random()
        alpha2, beta2 = 1.0 + 2 * rng.random(), 1.0 + 2 * rng.random()

        # Define log-likelihood for the entire dataset
        def log_likelihood(params):
            w_, a1, b1, a2, b2 = params
            w_ = np.clip(w_, 1e-9, 1 - 1e-9)  # keep w in (0,1)
            ll = 0.0
            for s, freq in zip(unique_counts, count_freq):
                logp1 = log_beta_binomial_pmf(s, k, a1, b1)
                logp2 = log_beta_binomial_pmf(s, k, a2, b2)
                # log p = log( w * e^(logp1) + (1-w) * e^(logp2) )
                # do log-sum-exp for numerical stability
                c1 = np.log(w_) + logp1
                c2 = np.log(1 - w_) + logp2
                cmax = max(c1, c2)
                ll += freq * (cmax + np.log(np.exp(c1 - cmax) + np.exp(c2 - cmax)))
            return ll

        def neg_log_likelihood(params):
            return -log_likelihood(params)

        old_ll = -np.inf

        for iteration in range(max_iter):
            # E-step: compute responsibilities
            # Using unique counts for efficiency
            logp1 = np.array(
                [log_beta_binomial_pmf(s, k, alpha1, beta1) for s in unique_counts]
            )
            logp2 = np.array(
                [log_beta_binomial_pmf(s, k, alpha2, beta2) for s in unique_counts]
            )

            logw1 = np.log(np.clip(w, 1e-9, 1 - 1e-9)) + logp1
            logw2 = np.log(np.clip(1 - w, 1e-9, 1 - 1e-9)) + logp2

            # denominator = log( e^(logw1) + e^(logw2) )
            max_ = np.maximum(logw1, logw2)
            denom = max_ + np.log(np.exp(logw1 - max_) + np.exp(logw2 - max_))
            gamma = np.exp(logw1 - denom)  # shape = (n_unique,)

            # M-step: update w accounting for frequencies
            w = np.sum(gamma * count_freq) / np.sum(count_freq)

            # joint numeric optimization to refine [w, alpha1, beta1, alpha2, beta2]
            x0 = [w, alpha1, beta1, alpha2, beta2]
            bnds = [
                (1e-9, 1 - 1e-9),  # w in (0,1)
                (1e-9, None),  # alpha1 > 0
                (1e-9, None),  # beta1 > 0
                (1e-9, None),  # alpha2 > 0
                (1e-9, None),  # beta2 > 0
            ]
            res = minimize(
                neg_log_likelihood,
                x0,
                method="L-BFGS-B",
                bounds=bnds,
                options={"maxiter": 20},
            )  # Fewer iterations within EM
            w, alpha1, beta1, alpha2, beta2 = res.x
            w = np.clip(w, 1e-9, 1 - 1e-9)

            new_ll = log_likelihood([w, alpha1, beta1, alpha2, beta2])
            if abs(new_ll - old_ll) < tol:
                result = {
                    "w": w,
                    "alpha1": alpha1,
                    "beta1": beta1,
                    "alpha2": alpha2,
                    "beta2": beta2,
                    "log_likelihood": new_ll,
                    "n_iter": iteration + 1,
                    "restart": restart,
                }
                if new_ll > best_ll:
                    best_ll = new_ll
                    best_result = result
                break
            old_ll = new_ll

        # If max_iter reached
        result = {
            "w": w,
            "alpha1": alpha1,
            "beta1": beta1,
            "alpha2": alpha2,
            "beta2": beta2,
            "log_likelihood": old_ll,
            "n_iter": max_iter,
            "restart": restart,
        }
        if old_ll > best_ll:
            best_ll = old_ll
            best_result = result

    # Apply consistent ordering to components before returning
    return ensure_consistent_component_ordering(best_result)


def fit_mixture_beta_binomial(
    counts: np.ndarray,
    k: int,
    fitting_method: str = "em",  # <--- CHOOSE "em" or "direct"
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int = 42,
    n_restarts: int = 2,
) -> Dict[str, float]:
    """
    Wrapper that calls either the EM-based or direct-likelihood-based approach
    to fit a 2-component Beta-Binomial mixture.

    Args:
        counts: array of observed counts in [0..k]
        k: number of trials
        fitting_method: "em" or "direct"
        max_iter: max iteration limit
        tol: convergence tolerance
        random_state: seed for random initialization
        n_restarts: number of random initializations to try

    Returns:
        Dict[str, float]: Fitted model parameters including Chi-square and p-value
    """
    if fitting_method == "em":
        result = em_mixture_beta_binomial(
            counts, k, max_iter, tol, random_state, n_restarts
        )
    elif fitting_method == "direct":
        result = fit_mixture_direct(counts, k, max_iter, tol, random_state, n_restarts)
    else:
        raise ValueError(f"Unknown fitting_method: {fitting_method}")

    # Now we calculate the Chi-square statistic and p-value
    chi_square_stat, p_value = chi_square_goodness_of_fit(
        counts,
        k,
        result["w"],
        result["alpha1"],
        result["beta1"],
        result["alpha2"],
        result["beta2"],
    )

    # Add Chi-square and p-value to the result dictionary
    result["chi_square_stat"] = chi_square_stat
    result["p_value"] = p_value

    return result
