from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize

from .pmf import (
    ensure_consistent_component_ordering,
    fit_mixture_direct,
    log_beta_binomial_pmf,
)


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
        Dict[str, float]: Fitted model parameters
    """
    if fitting_method == "em":
        return em_mixture_beta_binomial(
            counts, k, max_iter, tol, random_state, n_restarts
        )
    elif fitting_method == "direct":
        return fit_mixture_direct(counts, k, max_iter, tol, random_state, n_restarts)
    else:
        raise ValueError(f"Unknown fitting_method: {fitting_method}")


def fit_mixture_beta_binomial_with_constraints(
    counts: np.ndarray,
    k: int,
    fitting_method: str = "em",
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int = 42,
    n_restarts: int = 2,
    prev_exp_success: Optional[float] = None,
) -> Dict[str, float]:
    """
    Fit a 2-component Beta-Binomial mixture with the constraint that the
    expected success probability should not decrease from the previous rounds.

    Args:
        counts: Array of observed counts in [0..k]
        k: Number of trials
        fitting_method: "em" or "direct"
        max_iter: Maximum iteration limit
        tol: Convergence tolerance
        random_state: Random seed for initialization
        n_restarts: Number of random initializations to try
        prev_exp_success: Previous round's expected success probability

    Returns:
        Dict[str, float]: Fitted model parameters
    """
    # First, get an unconstrained fit
    result = fit_mixture_beta_binomial(
        counts, k, fitting_method, max_iter, tol, random_state, n_restarts
    )

    # If no previous success probability or no constraint needed, return the result
    if prev_exp_success is None:
        return result

    # Calculate the expected success probability of the current fit
    w = result["w"]
    alpha1 = result["alpha1"]
    beta1 = result["beta1"]
    alpha2 = result["alpha2"]
    beta2 = result["beta2"]

    exp1 = alpha1 / (alpha1 + beta1)
    exp2 = alpha2 / (alpha2 + beta2)

    # Weighted average of the two components' expected probabilities
    curr_exp_success = w * exp1 + (1 - w) * exp2

    # Check if constraint is satisfied
    if curr_exp_success >= prev_exp_success:
        return result  # Constraint already satisfied

    # If constraint not satisfied, adjust parameters to meet it
    # Approach: Scale alpha and beta to increase the expected value while
    # preserving the shape characteristics of the distribution

    # Calculate how much we need to increase the expected probability
    target_increase = (prev_exp_success - curr_exp_success) * 1.01  # Add small buffer

    # Option 1: Adjust component 1 (higher success probability)
    if exp1 < 0.95:  # Make sure we don't push it too close to 1
        # How much we need to increase exp1 to achieve target
        delta_exp1 = target_increase / w
        # New expected value for component 1
        new_exp1 = min(0.95, exp1 + delta_exp1)

        # Scale parameters to achieve the new expected value
        scale1 = (new_exp1 * (1 - exp1)) / (exp1 * (1 - new_exp1))
        new_alpha1 = alpha1 * scale1
        new_beta1 = beta1

        # Calculate the new overall expected probability
        new_curr_exp_success = w * new_exp1 + (1 - w) * exp2

        if new_curr_exp_success >= prev_exp_success:
            result["alpha1"] = new_alpha1
            result["beta1"] = new_beta1
            return result

    # Option 2: Adjust component 2 (lower success probability)
    if exp2 < 0.9:  # Make sure we don't push it too close to 1
        # How much we need to increase exp2 to achieve target
        delta_exp2 = target_increase / (1 - w)
        # New expected value for component 2
        new_exp2 = min(0.9, exp2 + delta_exp2)

        # Scale parameters to achieve the new expected value
        scale2 = (new_exp2 * (1 - exp2)) / (exp2 * (1 - new_exp2))
        new_alpha2 = alpha2 * scale2
        new_beta2 = beta2

        # Calculate the new overall expected probability
        new_curr_exp_success = w * exp1 + (1 - w) * new_exp2

        if new_curr_exp_success >= prev_exp_success:
            result["alpha2"] = new_alpha2
            result["beta2"] = new_beta2
            return result

    # Option 3: Adjust mixture weight
    # If individual components can't be adjusted enough, try adjusting the weight
    if exp1 > exp2:  # Ensure component 1 has higher success probability
        target_exp = prev_exp_success
        # Solve for w: w*exp1 + (1-w)*exp2 = target_exp
        new_w = (target_exp - exp2) / (exp1 - exp2)
        new_w = min(max(new_w, 0.1), 0.9)  # Ensure w stays in reasonable range

        result["w"] = new_w

    return result
