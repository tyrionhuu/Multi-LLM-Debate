#!/usr/bin/env python
import math
from math import exp

import numpy as np
from scipy.optimize import minimize


# -------------------------------------------------------------------
# Beta-Binomial PMF and log-PMF
# -------------------------------------------------------------------
def beta_binomial_pmf(s: int, k: int, alpha: float, beta: float) -> float:
    """
    Beta-Binomial PMF: BB(s | k, alpha, beta) = C(k, s) * B(alpha+s, beta+k-s) / B(alpha, beta).

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

    log_comb = math.lgamma(k + 1) - math.lgamma(s + 1) - math.lgamma(k - s + 1)
    log_num = (
        math.lgamma(alpha + s)
        + math.lgamma(beta + (k - s))
        - math.lgamma(alpha + beta + k)
    )
    log_den = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_p = log_comb + log_num - log_den
    return exp(log_p)


def log_beta_binomial_pmf(s: int, k: int, alpha: float, beta: float) -> float:
    """
    Returns the log of the Beta-Binomial PMF for s.

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

    log_comb = math.lgamma(k + 1) - math.lgamma(s + 1) - math.lgamma(k - s + 1)
    log_num = (
        math.lgamma(alpha + s)
        + math.lgamma(beta + (k - s))
        - math.lgamma(alpha + beta + k)
    )
    log_den = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return log_comb + log_num - log_den


# -------------------------------------------------------------------
# 1) Direct Maximum Likelihood approach (no explicit EM).
# -------------------------------------------------------------------
def direct_mixture_log_likelihood(params, counts, k):
    """
    Computes the log-likelihood of the dataset under a 2-component Beta-Binomial mixture
    with parameters = (w, alpha1, beta1, alpha2, beta2).
    """
    w, alpha1, beta1, alpha2, beta2 = params
    # clip w to avoid invalid probability
    w = np.clip(w, 1e-9, 1 - 1e-9)
    ll = 0.0
    for s in counts:
        p1 = beta_binomial_pmf(s, k, alpha1, beta1)
        p2 = beta_binomial_pmf(s, k, alpha2, beta2)
        # mixture
        mix_val = w * p1 + (1 - w) * p2
        # add small offset to avoid log(0)
        ll += math.log(mix_val + 1e-16)
    return ll


def fit_mixture_direct(counts, k, max_iter=100, tol=1e-6, random_state=42):
    """
    Fit a two-component Beta-Binomial mixture by directly maximizing the overall
    mixture log-likelihood (no explicit E-step).
    """
    rng = np.random.default_rng(random_state)

    # Filter out invalid counts
    valid_mask = (counts >= 0) & (counts <= k)
    counts = counts[valid_mask]
    if len(counts) == 0:
        raise ValueError("No valid counts found for direct fitting.")

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
        return -direct_mixture_log_likelihood(param_vec, counts, k)

    # We can use L-BFGS-B or any other method
    res = minimize(
        objective, x0, method="L-BFGS-B", bounds=bnds, options=dict(maxiter=max_iter)
    )
    w, alpha1, beta1, alpha2, beta2 = res.x
    w = np.clip(w, 1e-9, 1 - 1e-9)

    final_ll = direct_mixture_log_likelihood(
        [w, alpha1, beta1, alpha2, beta2], counts, k
    )
    return {
        "w": w,
        "alpha1": alpha1,
        "beta1": beta1,
        "alpha2": alpha2,
        "beta2": beta2,
        "log_likelihood": final_ll,
        "n_iter": res.nit,
    }


# -------------------------------------------------------------------
# 2) EM for a 2-component mixture of Beta-Binomial distributions
# -------------------------------------------------------------------
def em_mixture_beta_binomial(
    counts: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int = 42,
):
    """
    Fit a two-component mixture of Beta-Binomial distributions to observed counts {s_i},
    each s_i in [0, k].  The model is:
       S ~ w * BB(k, alpha1, beta1) + (1-w) * BB(k, alpha2, beta2).

    Args:
        counts: Array of observed counts
        k: Number of trials
        max_iter: Maximum number of EM iterations
        tol: Convergence tolerance for log-likelihood
        random_state: Random seed for initialization

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

    # 1) Initialization
    w = 0.5
    alpha1, beta1 = 1.0 + 2 * rng.random(), 1.0 + 2 * rng.random()
    alpha2, beta2 = 1.0 + 2 * rng.random(), 1.0 + 2 * rng.random()

    # Define log-likelihood for the entire dataset
    def log_likelihood(params):
        w_, a1, b1, a2, b2 = params
        w_ = np.clip(w_, 1e-9, 1 - 1e-9)  # keep w in (0,1)
        ll = 0.0
        for s in counts:
            logp1 = log_beta_binomial_pmf(s, k, a1, b1)
            logp2 = log_beta_binomial_pmf(s, k, a2, b2)
            # log p = log( w * e^(logp1) + (1-w) * e^(logp2) )
            # do log-sum-exp for numerical stability
            c1 = np.log(w_) + logp1
            c2 = np.log(1 - w_) + logp2
            cmax = max(c1, c2)
            ll += cmax + math.log(math.exp(c1 - cmax) + math.exp(c2 - cmax))
        return ll

    def neg_log_likelihood(params):
        return -log_likelihood(params)

    old_ll = -np.inf

    for iteration in range(max_iter):
        # E-step: compute responsibilities gamma_i = Prob(component=1 | s_i)
        logp1 = np.array([log_beta_binomial_pmf(s, k, alpha1, beta1) for s in counts])
        logp2 = np.array([log_beta_binomial_pmf(s, k, alpha2, beta2) for s in counts])

        logw1 = math.log(np.clip(w, 1e-9, 1 - 1e-9)) + logp1
        logw2 = math.log(np.clip(1 - w, 1e-9, 1 - 1e-9)) + logp2

        # denominator = log( e^(logw1) + e^(logw2) )
        max_ = np.maximum(logw1, logw2)
        denom = max_ + np.log(np.exp(logw1 - max_) + np.exp(logw2 - max_))
        gamma = np.exp(logw1 - denom)  # shape = (n,)

        # M-step: update w, alpha1, beta1, alpha2, beta2
        w = gamma.mean()  # simple closed form for mixture weight

        # joint numeric optimization to refine [w, alpha1, beta1, alpha2, beta2]
        x0 = [w, alpha1, beta1, alpha2, beta2]
        bnds = [
            (1e-9, 1 - 1e-9),  # w in (0,1)
            (1e-9, None),  # alpha1 > 0
            (1e-9, None),  # beta1 > 0
            (1e-9, None),  # alpha2 > 0
            (1e-9, None),  # beta2 > 0
        ]
        res = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bnds)
        w, alpha1, beta1, alpha2, beta2 = res.x
        w = np.clip(w, 1e-9, 1 - 1e-9)

        new_ll = log_likelihood([w, alpha1, beta1, alpha2, beta2])
        if abs(new_ll - old_ll) < tol:
            return {
                "w": w,
                "alpha1": alpha1,
                "beta1": beta1,
                "alpha2": alpha2,
                "beta2": beta2,
                "log_likelihood": new_ll,
                "n_iter": iteration + 1,
            }
        old_ll = new_ll

    # If max_iter reached, return final
    return {
        "w": w,
        "alpha1": alpha1,
        "beta1": beta1,
        "alpha2": alpha2,
        "beta2": beta2,
        "log_likelihood": old_ll,
        "n_iter": max_iter,
    }


def fit_mixture_beta_binomial(
    counts: np.ndarray,
    k: int,
    fitting_method="em",  # <--- CHOOSE "em" or "direct"
    max_iter: int = 100,
    tol: float = 1e-6,
    random_state: int = 42,
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
    """
    if fitting_method == "em":
        return em_mixture_beta_binomial(counts, k, max_iter, tol, random_state)
    elif fitting_method == "direct":
        return fit_mixture_direct(counts, k, max_iter, tol, random_state)
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
    DEBATES_CSV = Path("data/bool_q/llama3(7)/debate_rounds.csv")  # the debate rounds CSV
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
