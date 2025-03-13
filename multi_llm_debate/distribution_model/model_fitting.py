import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

from .utils import (
    beta_binomial_pmf,
    compute_predicted_pmf,
    compute_tvd,
    get_observed_pmf,
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


# EM algorithm for fitting the mixture of two Beta-Binomial distributions
def fit_mixture_em(
    observed_pmf: NDArray[np.float64], k: int, max_iter: int = 100, tol: float = 1e-5
) -> Tuple[float, float, float, float, float]:
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
    w = 0.5
    alpha1, beta1 = 10, 1  # High accuracy component
    alpha2, beta2 = 1, 10  # Low accuracy component

    for _ in range(max_iter):
        pmf1 = beta_binomial_pmf(s_values, k, alpha1, beta1)
        pmf2 = beta_binomial_pmf(s_values, k, alpha2, beta2)
        total_pmf = w * pmf1 + (1 - w) * pmf2
        total_pmf[total_pmf == 0] = 1e-10
        resp1 = (w * pmf1) / total_pmf
        resp2 = ((1 - w) * pmf2) / total_pmf

        w_new = np.sum(resp1 * observed_pmf) / np.sum(observed_pmf)

        def neg_log_likelihood1(params: NDArray[np.float64]) -> float:
            alpha, beta = params
            pmf = beta_binomial_pmf(s_values, k, alpha, beta)
            return -np.sum(resp1 * observed_pmf * np.log(pmf + 1e-10))

        res1 = minimize(
            neg_log_likelihood1, [alpha1, beta1], bounds=[(0.1, None), (0.1, None)]
        )
        alpha1_new, beta1_new = res1.x

        def neg_log_likelihood2(params: NDArray[np.float64]) -> float:
            alpha, beta = params
            pmf = beta_binomial_pmf(s_values, k, alpha, beta)
            return -np.sum(resp2 * observed_pmf * np.log(pmf + 1e-10))

        res2 = minimize(
            neg_log_likelihood2, [alpha2, beta2], bounds=[(0.1, None), (0.1, None)]
        )
        alpha2_new, beta2_new = res2.x

        param_diff = (
            np.abs(w_new - w)
            + np.abs(alpha1_new - alpha1)
            + np.abs(beta1_new - beta1)
            + np.abs(alpha2_new - alpha2)
            + np.abs(beta2_new - beta2)
        )
        if param_diff < tol:
            break

        w, alpha1, beta1, alpha2, beta2 = (
            w_new,
            alpha1_new,
            beta1_new,
            alpha2_new,
            beta2_new,
        )

    return w, alpha1, beta1, alpha2, beta2


# Modified fit_model_for_rounds to accept pre-computed distributions
def fit_model_for_rounds(
    dist_round0: pd.DataFrame, dist_round1: pd.DataFrame, k: int
) -> Tuple[
    Optional[Tuple[float, float, float, float, float]],
    Optional[Tuple[float, float, float, float, float]],
]:
    """Fit the mixture model for rounds 0 and 1 using pre-computed distributions and evaluate fit using TVD.

    Args:
        dist_round0: DataFrame with correct rate distribution for round 0.
        dist_round1: DataFrame with correct rate distribution for round 1.
        k: Number of judges (responses per task).

    Returns:
        params_round0, params_round1: Tuples with fitted parameters (w, alpha1, beta1, alpha2, beta2).
    """
    s_values = np.arange(k + 1)

    # Fit for round 0
    if dist_round0.empty:
        logger.error("No data available for round 0")
        return None, None
    observed_pmf_round0 = get_observed_pmf(dist_round0, k)
    params_round0 = fit_mixture_em(observed_pmf_round0, k)
    predicted_pmf_round0 = compute_predicted_pmf(s_values, k, *params_round0)
    tvd_round0 = compute_tvd(observed_pmf_round0, predicted_pmf_round0)
    logger.info(
        f"Round 0 parameters: w={params_round0[0]:.3f}, alpha1={params_round0[1]:.2f}, "
        f"beta1={params_round0[2]:.2f}, alpha2={params_round0[3]:.2f}, beta2={params_round0[4]:.2f}"
    )
    logger.info(f"Round 0 - TVD: {tvd_round0:.4f}")

    # Fit for round 1
    if dist_round1.empty:
        logger.error("No data available for round 1")
        return params_round0, None
    observed_pmf_round1 = get_observed_pmf(dist_round1, k)
    params_round1 = fit_mixture_em(observed_pmf_round1, k)
    predicted_pmf_round1 = compute_predicted_pmf(s_values, k, *params_round1)
    tvd_round1 = compute_tvd(observed_pmf_round1, predicted_pmf_round1)
    logger.info(
        f"Round 1 parameters: w={params_round1[0]:.3f}, alpha1={params_round1[1]:.2f}, "
        f"beta1={params_round1[2]:.2f}, alpha2={params_round1[3]:.2f}, beta2={params_round1[4]:.2f}"
    )
    logger.info(f"Round 1 - TVD: {tvd_round1:.4f}")

    return params_round0, params_round1


def main():
    """Test the model fitting process using calculate_correct_rate_distribution_for_round_n."""
    # Hardcoded configuration from your original __main__
    from ..analysis.calculate_correct_rate_distribution import (  # Adjust the import path as necessary
        calculate_correct_rate_distribution_for_round_n,
    )

    data_path = "output/bool_q/processed_data.csv"
    model_dir = "data/bool_q/llama3(7)"
    k = 10  # Assuming 10 judges; adjust based on your data

    # Load data
    try:
        dataframe = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import sys

        sys.exit(1)

    model_dir_path = Path(model_dir)
    if not model_dir_path.exists() or not model_dir_path.is_dir():
        logger.error(f"Model directory does not exist: {model_dir}")
        import sys

        sys.exit(1)

    # Calculate distributions for rounds 0 and 1
    logger.info("Calculating correct rate distribution for round 0...")
    dist_round0 = calculate_correct_rate_distribution_for_round_n(
        dataframe=dataframe, model_dir=model_dir_path, round_number=0
    )

    logger.info("Calculating correct rate distribution for round 1...")
    dist_round1 = calculate_correct_rate_distribution_for_round_n(
        dataframe=dataframe, model_dir=model_dir_path, round_number=1
    )

    # Fit the model and evaluate
    logger.info("Starting model fitting for rounds 0 and 1...")
    params_round0, params_round1 = fit_model_for_rounds(dist_round0, dist_round1, k)

    if params_round0 and params_round1:
        logger.info("Model fitting completed successfully.")
        # Optional: Extrapolate to round 2
        w0, a1_0, b1_0, a2_0, b2_0 = params_round0
        w1, a1_1, b1_1, a2_1, b2_1 = params_round1
        delta_w = w1 - w0
        delta_a1 = a1_1 - a1_0
        delta_b1 = b1_1 - b1_0
        delta_a2 = a2_1 - a2_0
        delta_b2 = b2_1 - b2_0
        t = 2
        params_round2 = (
            max(0, min(1, w0 + t * delta_w)),
            max(1, a1_0 + t * delta_a1),
            max(1, b1_0 + t * delta_b1),
            max(1, a2_0 + t * delta_a2),
            max(1, b2_0 + t * delta_b2),
        )
        logger.info(
            f"Predicted parameters for round {t}: w={params_round2[0]:.3f}, "
            f"alpha1={params_round2[1]:.2f}, beta1={params_round2[2]:.2f}, "
            f"alpha2={params_round2[3]:.2f}, beta2={params_round2[4]:.2f}"
        )
    else:
        logger.error("Model fitting failed.")


if __name__ == "__main__":
    main()
