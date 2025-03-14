import numpy as np
from scipy.special import betaln, binom
import pandas as pd


def beta_binomial_pmf(s: int, k: int, alpha: float, beta: float) -> float:
    """Compute the Beta-Binomial probability mass function."""
    log_pmf = (
        np.log(binom(k, s)) + betaln(s + alpha, k - s + beta) - betaln(alpha, beta)
    )
    return np.exp(log_pmf)


def estimate_beta_mom(mu: float, sigma_sq: float) -> tuple[float, float]:
    """Estimate Beta parameters via method of moments."""
    eps = 1e-6
    sigma_sq = max(sigma_sq, eps)
    nu = (mu * (1 - mu) / sigma_sq) - 1
    if nu <= 0:
        nu = eps  # Regularization to avoid invalid parameters
    alpha = mu * nu
    beta = (1 - mu) * nu
    return max(alpha, eps), max(beta, eps)


def beta_binomial_em(
    s_list: list[int], k: int, max_iters: int = 100, tol: float = 1e-4
) -> tuple[float, float, float, float, float]:
    """EM algorithm for Beta-Binomial mixture model."""
    # Initialize parameters
    w = 0.5
    alpha1, beta1 = 1.0, 1.0
    alpha2, beta2 = 1.0, 1.0
    n = len(s_list)
    s_array = np.array(s_list)

    for _ in range(max_iters):
        # E-step: Compute responsibilities
        gamma = np.zeros(n)
        for i, s in enumerate(s_list):
            prob1 = beta_binomial_pmf(s, k, alpha1, beta1)
            prob2 = beta_binomial_pmf(s, k, alpha2, beta2)
            numerator = w * prob1
            denom = numerator + (1 - w) * prob2
            gamma[i] = numerator / denom if denom != 0 else 0.0

        # M-step: Update parameters
        w_new = np.mean(gamma)

        # Update component 1 (high-correctness)
        total_weight1 = np.sum(gamma)
        if total_weight1 > 0:
            mu1 = np.sum(gamma * s_array) / (k * total_weight1)
            sigma_sq1 = np.sum(gamma * (s_array / k - mu1) ** 2) / total_weight1
            alpha1_new, beta1_new = estimate_beta_mom(mu1, sigma_sq1)
        else:
            alpha1_new, beta1_new = alpha1, beta1  # No change

        # Update component 2 (low-correctness)
        total_weight2 = np.sum(1 - gamma)
        if total_weight2 > 0:
            mu2 = np.sum((1 - gamma) * s_array) / (k * total_weight2)
            sigma_sq2 = np.sum((1 - gamma) * (s_array / k - mu2) ** 2) / total_weight2
            alpha2_new, beta2_new = estimate_beta_mom(mu2, sigma_sq2)
        else:
            alpha2_new, beta2_new = alpha2, beta2  # No change

        # Check convergence
        delta = (
            np.abs(w_new - w)
            + np.abs(alpha1_new - alpha1)
            + np.abs(beta1_new - beta1)
            + np.abs(alpha2_new - alpha2)
            + np.abs(beta2_new - beta2)
        )
        if delta < tol:
            break

        # Update parameters
        w, alpha1, beta1, alpha2, beta2 = (
            w_new,
            alpha1_new,
            beta1_new,
            alpha2_new,
            beta2_new,
        )

    return w, alpha1, beta1, alpha2, beta2


def extract_si_from_distribution(df: pd.DataFrame) -> list[int]:
    """Extract correct counts from one-hot encoded DataFrame."""
    s_list = []
    count_cols = [c for c in df.columns if c not in ("task_id", "round_number")]

    for _, row in df.iterrows():
        for col in count_cols:
            if row[col] == 1:
                s_list.append(int(col))
                break
    return s_list


if __name__ == "__main__":
    from pathlib import Path

    from ..analysis.calculate_correct_rate_distribution import (
        calculate_correct_rate_distribution_for_round_n,
    )

    # Define paths
    DATA_PATH = Path("output/bool_q/processed_data.csv")
    MODEL_DIR_PATH = Path("data/bool_q/llama3(11)")
    OUTPUT_DIR = Path("output")
    try:
        # Load data
        dataframe = pd.read_csv(DATA_PATH)
        print(f"Loaded data from {DATA_PATH}")
    except Exception as e:
        print(f"Error loading data: {e}")

    # Process all 11 rounds
    print("\nBeta-Binomial parameters for 11 rounds:")
    print("---------------------------------------")
    for round_num in range(11):
        try:
            round_df = calculate_correct_rate_distribution_for_round_n(
                dataframe, MODEL_DIR_PATH, round_num
            )
            s_list = extract_si_from_distribution(round_df)
            if not s_list:
                print(f"Round {round_num}: No data available")
                continue
                
            k = max(s_list)  # Assumes max(s) = number of agents
            w, a1, b1, a2, b2 = beta_binomial_em(s_list, k)
            
            print(f"Round {round_num}:")
            print(f"  Component 1 (weight={w:.4f}): alpha={a1:.4f}, beta={b1:.4f}")
            print(f"  Component 2 (weight={1-w:.4f}): alpha={a2:.4f}, beta={b2:.4f}")
        except Exception as e:
            print(f"Round {round_num}: Error - {e}")
