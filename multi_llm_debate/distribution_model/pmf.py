from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln  # More efficient than math.lgamma
from typing import Dict

# -------------------------------------------------------------------
# Beta-Binomial PMF and log-PMF with caching
# -------------------------------------------------------------------
@lru_cache(maxsize=1024)
def beta_binomial_pmf(s: int, k: int, alpha: float, beta: float) -> float:
    """
    Beta-Binomial PMF: BB(s | k, alpha, beta) = C(k, s) * B(alpha+s, beta+k-s) / B(alpha, beta).

    This implementation uses caching to avoid redundant calculations.

    Args:
        s: Number of successes, must be between 0 and k inclusive
        k: Number of trials
        alpha: First shape parameter of the beta distribution
        beta: Second shape parameter of the beta distribution

    Returns:
        float: The probability mass at s
    """
    # Validate inputs
    if not (0 <= s <= k):
        return 0.0

    # Use vectorized gammaln (scipy.special) instead of math.lgamma
    log_comb = gammaln(k + 1) - gammaln(s + 1) - gammaln(k - s + 1)
    log_num = gammaln(alpha + s) + gammaln(beta + (k - s)) - gammaln(alpha + beta + k)
    log_den = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    log_p = log_comb + log_num - log_den
    return np.exp(log_p)  # np.exp can be faster than math.exp


@lru_cache(maxsize=1024)
def log_beta_binomial_pmf(s: int, k: int, alpha: float, beta: float) -> float:
    """
    Returns the log of the Beta-Binomial PMF for s.

    This implementation uses caching to avoid redundant calculations.

    Args:
        s: Number of successes, must be between 0 and k inclusive
        k: Number of trials
        alpha: First shape parameter of the beta distribution
        beta: Second shape parameter of the beta distribution

    Returns:
        float: The log probability mass at s
    """
    # Validate inputs
    if not (0 <= s <= k):
        return float("-inf")  # log(0) = -infinity for invalid inputs

    # Use vectorized gammaln (scipy.special) instead of math.lgamma
    log_comb = gammaln(k + 1) - gammaln(s + 1) - gammaln(k - s + 1)
    log_num = gammaln(alpha + s) + gammaln(beta + (k - s)) - gammaln(alpha + beta + k)
    log_den = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    return log_comb + log_num - log_den


# -------------------------------------------------------------------
# 1) Direct Maximum Likelihood approach - Optimized
# -------------------------------------------------------------------
def direct_mixture_log_likelihood(
    params: list[float],
    counts: np.ndarray,
    k: int,
    unique_counts: np.ndarray = None,
    count_freq: np.ndarray = None,
) -> float:
    """
    Computes the log-likelihood of the dataset under a 2-component Beta-Binomial mixture
    with parameters = (w, alpha1, beta1, alpha2, beta2).

    This optimized version can work with frequency counts for better performance.

    Args:
        params: Model parameters [w, alpha1, beta1, alpha2, beta2]
        counts: Array of observed counts
        k: Number of trials
        unique_counts: Optional array of unique count values
        count_freq: Optional array of frequencies for unique_counts

    Returns:
        float: Log-likelihood value
    """
    w, alpha1, beta1, alpha2, beta2 = params
    # clip w to avoid invalid probability
    w = np.clip(w, 1e-9, 1 - 1e-9)

    # If unique counts and frequencies are provided, use them for efficiency
    if unique_counts is not None and count_freq is not None:
        ll = 0.0
        for s, freq in zip(unique_counts, count_freq):
            p1 = beta_binomial_pmf(s, k, alpha1, beta1)
            p2 = beta_binomial_pmf(s, k, alpha2, beta2)
            # mixture
            mix_val = w * p1 + (1 - w) * p2
            # add small offset to avoid log(0)
            ll += freq * np.log(mix_val + 1e-16)
        return ll

    # Otherwise, process all counts individually
    ll = 0.0
    for s in counts:
        p1 = beta_binomial_pmf(s, k, alpha1, beta1)
        p2 = beta_binomial_pmf(s, k, alpha2, beta2)
        # mixture
        mix_val = w * p1 + (1 - w) * p2
        # add small offset to avoid log(0)
        ll += np.log(mix_val + 1e-16)
    return ll


def ensure_consistent_component_ordering(params: Dict) -> Dict:
    """
    Ensure consistent ordering of mixture components to make comparison across
    rounds more reliable. This function:

    1. Orders components so that component 1 has higher expected success rate
       (alpha1/(alpha1+beta1) > alpha2/(alpha2+beta2))
    2. For component 1 (higher success): Ensures alpha1 > beta1 when possible
    3. For component 2 (lower success): Ensures alpha2 < beta2 when possible

    Args:
        params: Dictionary with fitted model parameters
                (w, alpha1, beta1, alpha2, beta2)

    Returns:
        Dict: Dictionary with consistently ordered components
    """
    # Calculate expected value of each beta component: alpha/(alpha+beta)
    expected1 = params["alpha1"] / (params["alpha1"] + params["beta1"])
    expected2 = params["alpha2"] / (params["alpha2"] + params["beta2"])

    # Step 1: Ensure components are ordered by expected success rate
    if expected1 < expected2:
        # Swap components
        result = {
            "w": 1.0 - params["w"],  # Adjust weight accordingly
            "alpha1": params["alpha2"],
            "beta1": params["beta2"],
            "alpha2": params["alpha1"],
            "beta2": params["beta1"],
            "log_likelihood": params.get("log_likelihood"),
            "n_iter": params.get("n_iter"),
            "restart": params.get("restart", 0),
        }
    else:
        # Components already in correct order
        result = params.copy()

    # Step 2: Now ensure alpha and beta are ordered within each component
    # For component 1 (higher success rate): Prefer alpha1 > beta1
    if result["alpha1"] < result["beta1"]:
        # Only swap if it doesn't change the expected value significantly
        # This ensures we don't disrupt the mixture interpretation
        exp1 = result["alpha1"] / (result["alpha1"] + result["beta1"])
        # Calculate what would happen if we scaled alpha1 and beta1
        scale = result["beta1"] / result["alpha1"]
        new_alpha1 = result["beta1"] * scale
        new_beta1 = result["alpha1"]
        new_exp1 = new_alpha1 / (new_alpha1 + new_beta1)

        # Only swap if the change in expected value is small
        if abs(new_exp1 - exp1) < 0.01:
            result["alpha1"] = new_alpha1
            result["beta1"] = new_beta1

    # For component 2 (lower success rate): Prefer alpha2 < beta2
    if result["alpha2"] > result["beta2"]:
        # Only swap if it doesn't change the expected value significantly
        exp2 = result["alpha2"] / (result["alpha2"] + result["beta2"])
        # Calculate what would happen if we scaled alpha2 and beta2
        scale = result["alpha2"] / result["beta2"]
        new_alpha2 = result["beta2"]
        new_beta2 = result["alpha2"] * scale
        new_exp2 = new_alpha2 / (new_alpha2 + new_beta2)

        # Only swap if the change in expected value is small
        if abs(new_exp2 - exp2) < 0.01:
            result["alpha2"] = new_alpha2
            result["beta2"] = new_beta2

    return result


def fit_mixture_direct(
    counts, k, max_iter=100, tol=1e-6, random_state=42, n_restarts=3
) -> Dict:
    """
    Fit a two-component Beta-Binomial mixture by directly maximizing the overall
    mixture log-likelihood with multiple restarts for better convergence.

    Args:
        counts: Array of observed counts
        k: Number of trials
        max_iter: Maximum number of optimization iterations
        tol: Convergence tolerance
        random_state: Random seed for initialization
        n_restarts: Number of random restarts to try

    Returns:
        dict: Fitted model parameters
    """
    rng = np.random.default_rng(random_state)

    # Filter out invalid counts
    valid_mask = (counts >= 0) & (counts <= k)
    counts = counts[valid_mask]
    if len(counts) == 0:
        raise ValueError("No valid counts found for direct fitting.")

    # Pre-compute unique counts and their frequencies for efficiency
    unique_counts, count_freq = np.unique(counts, return_counts=True)

    # Try multiple random initializations and pick the best
    best_result = None
    best_ll = float("-inf")

    for restart in range(n_restarts):
        # Initial guess
        w0 = 0.5
        alpha10, beta10 = 1.0 + 2 * rng.random(), 1.0 + 2 * rng.random()
        alpha20, beta20 = 1.0 + 2 * rng.random(), 1.0 + 2 * rng.random()
        x0 = [w0, alpha10, beta10, alpha20, beta20]

        # Bounds to keep alpha, beta > 0 and w in (0,1)
        bnds = [
            (1e-9, 1 - 1e-9),  # w
            (1e-9, None),  # alpha1
            (1e-9, None),  # beta1
            (1e-9, None),  # alpha2
            (1e-9, None),  # beta2
        ]

        def objective(param_vec):
            return -direct_mixture_log_likelihood(
                param_vec, counts, k, unique_counts, count_freq
            )

        # We can use L-BFGS-B or any other method
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bnds,
            options=dict(maxiter=max_iter, gtol=tol),
        )

        w, alpha1, beta1, alpha2, beta2 = res.x
        w = np.clip(w, 1e-9, 1 - 1e-9)

        final_ll = direct_mixture_log_likelihood(
            [w, alpha1, beta1, alpha2, beta2], counts, k, unique_counts, count_freq
        )

        # Keep track of the best result
        if final_ll > best_ll:
            best_ll = final_ll
            best_result = {
                "w": w,
                "alpha1": alpha1,
                "beta1": beta1,
                "alpha2": alpha2,
                "beta2": beta2,
                "log_likelihood": final_ll,
                "n_iter": res.nit,
                "restart": restart,
            }

    # Apply consistent ordering to components before returning
    return ensure_consistent_component_ordering(best_result)
