from scipy.stats import betabinom
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from ..analysis.calculate_correct_rate_distribution import (
    calculate_correct_rate_distribution_for_round_n,
)



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_fitting.log"),
    ],
)
logger = logging.getLogger(__name__)

# Function to compute Beta-Binomial PMF
def beta_binomial_pmf(s, k, alpha, beta):
    """Compute the PMF of Beta-Binomial distribution."""
    return betabinom.pmf(s, k, alpha, beta)

# EM algorithm for fitting the mixture of two Beta-Binomial distributions
def fit_mixture_em(observed_pmf, k, max_iter=100, tol=1e-5):
    """Fit a mixture of two Beta-Binomial distributions using EM algorithm.

    Args:
        observed_pmf: Array of observed probabilities for S = 0 to k.
        k: Number of judges.
        max_iter: Maximum number of EM iterations.
        tol: Tolerance for convergence.

    Returns:
        Tuple of (w, alpha1, beta1, alpha2, beta2)
    """
    s_values = np.arange(k + 1)

    # Initialize parameters: w, alpha1, beta1, alpha2, beta2
    w = 0.5
    alpha1, beta1 = 10, 1  # High accuracy component
    alpha2, beta2 = 1, 10  # Low accuracy component

    for _ in range(max_iter):
        # E-step: Compute responsibilities
        pmf1 = beta_binomial_pmf(s_values, k, alpha1, beta1)
        pmf2 = beta_binomial_pmf(s_values, k, alpha2, beta2)
        total_pmf = w * pmf1 + (1 - w) * pmf2

        # Avoid division by zero
        total_pmf[total_pmf == 0] = 1e-10

        resp1 = (w * pmf1) / total_pmf
        resp2 = ((1 - w) * pmf2) / total_pmf

        # M-step: Update parameters
        # Update mixture weight w
        w_new = np.sum(resp1 * observed_pmf) / np.sum(observed_pmf)

        # Update alpha1, beta1 for component 1
        def neg_log_likelihood1(params):
            alpha, beta = params
            pmf = beta_binomial_pmf(s_values, k, alpha, beta)
            return -np.sum(resp1 * observed_pmf * np.log(pmf + 1e-10))

        res1 = minimize(neg_log_likelihood1, [alpha1, beta1], bounds=[(0.1, None), (0.1, None)])
        alpha1_new, beta1_new = res1.x

        # Update alpha2, beta2 for component 2
        def neg_log_likelihood2(params):
            alpha, beta = params
            pmf = beta_binomial_pmf(s_values, k, alpha, beta)
            return -np.sum(resp2 * observed_pmf * np.log(pmf + 1e-10))

        res2 = minimize(neg_log_likelihood2, [alpha2, beta2], bounds=[(0.1, None), (0.1, None)])
        alpha2_new, beta2_new = res2.x

        # Check for convergence
        param_diff = np.abs(w_new - w) + np.abs(alpha1_new - alpha1) + np.abs(beta1_new - beta1) + \
                     np.abs(alpha2_new - alpha2) + np.abs(beta2_new - beta2)
        if param_diff < tol:
            break

        # Update parameters for next iteration
        w, alpha1, beta1, alpha2, beta2 = w_new, alpha1_new, beta1_new, alpha2_new, beta2_new

    return w, alpha1, beta1, alpha2, beta2

# Function to compute observed PMF from distribution DataFrame
def get_observed_pmf(distribution_df, k):
    """Convert the distribution DataFrame to observed PMF for S^t.

    Args:
        distribution_df: DataFrame with bins and task counts.
        k: Number of judges.

    Returns:
        observed_pmf: Array of probabilities for S = 0 to k.
    """
    bin_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    observed_counts = np.zeros(k + 1)

    for _, row in distribution_df.iterrows():
        for bin_label in bin_labels:
            if row[bin_label] == 1:
                bin_idx = bin_labels.index(bin_label)
                # Map bin to possible S values
                min_rate, max_rate = bin_idx / 10, (bin_idx + 1) / 10
                possible_s = [s for s in range(k + 1) if min_rate <= s / k <= max_rate or 
                              (max_rate == 1.0 and s / k == 1.0)]
                if possible_s:
                    # Distribute count uniformly within the bin
                    for s in possible_s:
                        observed_counts[s] += 1 / len(possible_s)
                break

    # Normalize to get PMF
    observed_pmf = observed_counts / observed_counts.sum() if observed_counts.sum() > 0 else observed_counts
    return observed_pmf

# Main function to fit the model for rounds 0 and 1
def fit_model_for_rounds(dataframe: pd.DataFrame, model_dir: Path, k: int):
    """Fit the mixture model for rounds 0 and 1.

    Args:
        dataframe: DataFrame with experiment results.
        model_dir: Directory with model outputs.
        k: Number of judges (responses per task).

    Returns:
        params_round0, params_round1: Tuples with fitted parameters (w, alpha1, beta1, alpha2, beta2).
    """
    # Fit for round 0
    dist_round0 = calculate_correct_rate_distribution_for_round_n(dataframe, model_dir, 0)
    if dist_round0.empty:
        logger.error("No data available for round 0")
        return None, None
    observed_pmf_round0 = get_observed_pmf(dist_round0, k)
    params_round0 = fit_mixture_em(observed_pmf_round0, k)
    logger.info(f"Round 0 parameters: w={params_round0[0]:.3f}, alpha1={params_round0[1]:.2f}, "
                f"beta1={params_round0[2]:.2f}, alpha2={params_round0[3]:.2f}, beta2={params_round0[4]:.2f}")

    # Fit for round 1
    dist_round1 = calculate_correct_rate_distribution_for_round_n(dataframe, model_dir, 1)
    if dist_round1.empty:
        logger.error("No data available for round 1")
        return params_round0, None
    observed_pmf_round1 = get_observed_pmf(dist_round1, k)
    params_round1 = fit_mixture_em(observed_pmf_round1, k)
    logger.info(f"Round 1 parameters: w={params_round1[0]:.3f}, alpha1={params_round1[1]:.2f}, "
                f"beta1={params_round1[2]:.2f}, alpha2={params_round1[3]:.2f}, beta2={params_round1[4]:.2f}")

    return params_round0, params_round1