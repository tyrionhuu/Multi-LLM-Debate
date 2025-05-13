import numpy as np
import pandas as pd
from scipy.stats import chi2

from .pmf import log_beta_binomial_pmf


def extract_correct_counts(round_df: pd.DataFrame) -> pd.Series:
    """
    Given the distribution DataFrame with columns "task_id", "round_number", "0", "1", ..., "k",
    this returns a Series of integer counts: how many agents were correct for each task.
    """
    # Identify columns that are numeric bin labels
    bin_cols = [c for c in round_df.columns if c.isdigit()]
    counts = []
    for _, row in round_df.iterrows():
        for b in bin_cols:
            if row[b] == 1:
                counts.append(int(b))
                break
    return pd.Series(counts)


def chi_square_goodness_of_fit(counts, k, w, alpha1, beta1, alpha2, beta2):
    # Unique counts and their frequencies
    unique_counts, count_freq = np.unique(counts, return_counts=True)

    # Compute the expected frequencies from the fitted Beta-Binomial mixture model
    expected_freqs = np.array(
        [
            w * log_beta_binomial_pmf(s, k, alpha1, beta1)
            + (1 - w) * log_beta_binomial_pmf(s, k, alpha2, beta2)
            for s in unique_counts
        ]
    )

    # Scale the expected frequencies to match the total number of observations
    expected_freqs *= np.sum(count_freq) / np.sum(expected_freqs)

    # Compute the Chi-square statistic
    chi_square_stat = np.sum((count_freq - expected_freqs) ** 2 / expected_freqs)

    # Degrees of freedom: (number of categories - 1) - number of parameters
    df = len(unique_counts) - 1 - 5  # 5 parameters: w, alpha1, beta1, alpha2, beta2

    # Compute the p-value from the Chi-square distribution
    p_value = 1 - chi2.cdf(chi_square_stat, df)

    return chi_square_stat, p_value
