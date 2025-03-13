from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import betabinom, entropy


def beta_binomial_pmf(
    s: Union[int, NDArray[np.int_]], k: int, alpha: float, beta: float
) -> Union[float, NDArray[np.float_]]:
    """Compute the PMF of Beta-Binomial distribution.

    Args:
        s: Success count or array of success counts.
        k: Number of trials.
        alpha: First shape parameter of the beta distribution.
        beta: Second shape parameter of the beta distribution.

    Returns:
        Probability mass at s or array of probabilities.
    """
    return betabinom.pmf(s, k, alpha, beta)


def compute_predicted_pmf(
    s_values: NDArray[np.int_],
    k: int,
    w: float,
    alpha1: float,
    beta1: float,
    alpha2: float,
    beta2: float,
) -> NDArray[np.float_]:
    """Compute the predicted probability mass function for a mixture model.

    Args:
        s_values: Array of success counts.
        k: Number of trials.
        w: Weight of the first component in the mixture.
        alpha1: First shape parameter of the first beta distribution.
        beta1: Second shape parameter of the first beta distribution.
        alpha2: First shape parameter of the second beta distribution.
        beta2: Second shape parameter of the second beta distribution.

    Returns:
        Array of predicted probabilities.
    """
    pmf1 = beta_binomial_pmf(s_values, k, alpha1, beta1)
    pmf2 = beta_binomial_pmf(s_values, k, alpha2, beta2)
    return w * pmf1 + (1 - w) * pmf2


def compute_tvd(
    observed_pmf: NDArray[np.float_], predicted_pmf: NDArray[np.float_]
) -> float:
    """Compute the Total Variation Distance between two distributions.

    Args:
        observed_pmf: Observed probability mass function.
        predicted_pmf: Predicted probability mass function.

    Returns:
        Total variation distance.
    """
    return 0.5 * np.sum(np.abs(observed_pmf - predicted_pmf))


def get_observed_pmf(distribution_df: pd.DataFrame, k: int) -> NDArray[np.float_]:
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
                min_rate, max_rate = bin_idx / 10, (bin_idx + 1) / 10
                possible_s = [
                    s
                    for s in range(k + 1)
                    if min_rate <= s / k <= max_rate
                    or (max_rate == 1.0 and s / k == 1.0)
                ]
                if possible_s:
                    for s in possible_s:
                        observed_counts[s] += 1 / len(possible_s)
                break

    observed_pmf = (
        observed_counts / observed_counts.sum()
        if observed_counts.sum() > 0
        else observed_counts
    )
    return observed_pmf


# Function to compute Kullback-Leibler (KL) Divergence
def compute_kl_divergence(observed_pmf: np.ndarray, predicted_pmf: np.ndarray) -> float:
    """Compute the KL Divergence from observed to predicted PMF."""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    observed_pmf = observed_pmf + epsilon
    predicted_pmf = predicted_pmf + epsilon
    # Normalize to ensure sums are 1 after adding epsilon
    observed_pmf = observed_pmf / observed_pmf.sum()
    predicted_pmf = predicted_pmf / predicted_pmf.sum()
    return entropy(observed_pmf, predicted_pmf)
