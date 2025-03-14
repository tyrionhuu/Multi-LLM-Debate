#!/usr/bin/env python
import math
from functools import lru_cache
from math import exp

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln  # More efficient than math.lgamma


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
    count_freq: np.ndarray = None
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


def fit_mixture_direct(
    counts, k, max_iter=100, tol=1e-6, random_state=42, n_restarts=3
):
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

    return best_result


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
):
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
        dict: Dictionary of the learned parameters
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

    return best_result


def fit_mixture_beta_binomial(
    counts: np.ndarray,
    k: int,
    fitting_method: str = "em",  # <--- CHOOSE "em" or "direct"
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int = 42,
    n_restarts: int = 2,
    parallel: bool = False,
):
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
        parallel: whether to use parallel processing (if available)

    Returns:
        dict: Fitted model parameters
    """
    # Try to use parallel processing if requested and available
    if parallel:
        try:
            import joblib

            print("Using parallel processing for model fitting")
            # This would need additional code to implement parallel fitting
            # Not implemented in this example
        except ImportError:
            print("joblib not available, using serial processing")
            parallel = False

    if fitting_method == "em":
        return em_mixture_beta_binomial(
            counts, k, max_iter, tol, random_state, n_restarts
        )
    elif fitting_method == "direct":
        return fit_mixture_direct(counts, k, max_iter, tol, random_state, n_restarts)
    else:
        raise ValueError(f"Unknown fitting_method: {fitting_method}")


# -------------------------------------------------------------------
# Example usage in main
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    import pandas as pd

    # Example import from your code:
    from ..analysis.calculate_correct_rate_distribution import (
        calculate_correct_rate_distribution,
    )

    # PATHS (placeholders in this example)
    ANSWERS_CSV = Path("output/bool_q/processed_data.csv")  # your "id" -> "answer" file
    DEBATES_CSV = Path(
        "data/bool_q/llama3(11)/debate_rounds.csv"
    )  # the debate rounds CSV
    MAX_ROUNDS = None  # or an int

    # Choose which method to use for fitting
    FIT_METHOD = "em"  # or "direct"

    # Load answers data
    try:
        df_answers = pd.read_csv(ANSWERS_CSV)
        # Convert id to numeric and clean
        df_answers["id"] = pd.to_numeric(df_answers["id"], errors="coerce")
        df_answers.dropna(subset=["id"], inplace=True)
        df_answers["id"] = df_answers["id"].astype(int)
        print(f"Loaded answers data from {ANSWERS_CSV}")
    except Exception as e:
        print(f"Failed to load answers data: {e}")
        sys.exit(1)

    # Load debate rounds data
    try:
        df_debates = pd.read_csv(DEBATES_CSV)
        # Convert task_id and round_number to numeric and clean
        df_debates["task_id"] = pd.to_numeric(df_debates["task_id"], errors="coerce")
        df_debates["round_number"] = pd.to_numeric(
            df_debates["round_number"], errors="coerce"
        )
        df_debates.dropna(subset=["task_id", "round_number"], inplace=True)
        df_debates["task_id"] = df_debates["task_id"].astype(int)
        df_debates["round_number"] = df_debates["round_number"].astype(int)
        print(f"Loaded debate rounds from {DEBATES_CSV}")
    except Exception as e:
        print(f"Error loading debate rounds data: {e}")
        sys.exit(1)

    # Get aggregated data for all rounds
    try:
        print("Calculating correct rate distribution...")
        aggregated_df = calculate_correct_rate_distribution(
            df_answers=df_answers, df_debates=df_debates, max_rounds=MAX_ROUNDS
        )
    except Exception as e:
        print(f"Error calculating correct rate distribution: {e}")
        sys.exit(1)

    if aggregated_df.empty:
        print("No data available for analysis.")
        sys.exit(1)

    # Print the aggregated DataFrame
    print("Aggregated DataFrame:")
    print(aggregated_df)

    prev_fit_result = None

    # Process each round in the aggregated data
    for _, row in aggregated_df.iterrows():
        round_number = int(row["round_number"])
        print(f"Processing round {round_number} using fitting method: {FIT_METHOD}")

        # Extract bin columns (representing correct counts)
        bin_columns = [col for col in aggregated_df.columns if col.isdigit()]

        # Create a dict: {count_value: frequency}
        counts_dict = {int(bin_col): row[bin_col] for bin_col in bin_columns}

        # Expand into a list of counts repeated by their frequency
        all_counts = []
        for count_val, frequency in counts_dict.items():
            all_counts.extend([count_val] * int(frequency))

        counts_array = np.array(all_counts)

        # k = max possible correct
        k = max(int(col) for col in bin_columns)

        # Fit the model (choose EM or direct)
        fit_result = fit_mixture_beta_binomial(
            counts_array, k=k, fitting_method=FIT_METHOD
        )

        # Print the fit results
        print(f"Round {round_number} fit results:")
        print(f"  Mixture weight (w): {fit_result['w']:.4f}")
        print(f"  Alpha1: {fit_result['alpha1']:.4f}")
        print(f"  Beta1:  {fit_result['beta1']:.4f}")
        print(f"  Alpha2: {fit_result['alpha2']:.4f}")
        print(f"  Beta2:  {fit_result['beta2']:.4f}")
        print(f"  Log-likelihood: {fit_result['log_likelihood']:.4f}")
        print(f"  Number of iterations: {fit_result['n_iter']}")
        print(f"  Total tasks analyzed: {row['total_tasks']}")

        # Print deltas from previous round if available
        if round_number > 0 and prev_fit_result is not None:
            print("  Deltas from previous round:")
            print(f"    Δ Mixture weight: {fit_result['w'] - prev_fit_result['w']:.4f}")
            print(
                f"    Δ Alpha1: {fit_result['alpha1'] - prev_fit_result['alpha1']:.4f}"
            )
            print(f"    Δ Beta1: {fit_result['beta1'] - prev_fit_result['beta1']:.4f}")
            print(
                f"    Δ Alpha2: {fit_result['alpha2'] - prev_fit_result['alpha2']:.4f}"
            )
            print(f"    Δ Beta2: {fit_result['beta2'] - prev_fit_result['beta2']:.4f}")

        prev_fit_result = fit_result.copy()
        print("-" * 80)
