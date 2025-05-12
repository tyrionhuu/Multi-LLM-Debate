from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..analysis.calculate_correct_rate_distribution import (
    calculate_correct_rate_distribution,
)
from .fit_beta_binomial_mixture import (
    ensure_consistent_component_ordering,
    fit_mixture_beta_binomial,
    fit_mixture_beta_binomial_with_constraints,
)


def beta_mixture_cdf(
    x: np.ndarray, w: float, alpha1: float, beta1: float, alpha2: float, beta2: float
) -> np.ndarray:
    """
    Calculate the CDF of a beta mixture distribution.

    Args:
        x: Points at which to evaluate the CDF
        w: Mixture weight for the first component
        alpha1: Alpha parameter of the first beta component
        beta1: Beta parameter of the first beta component
        alpha2: Alpha parameter of the second beta component
        beta2: Beta parameter of the second beta component

    Returns:
        Array of CDF values
    """
    cdf1 = stats.beta.cdf(x, alpha1, beta1)
    cdf2 = stats.beta.cdf(x, alpha2, beta2)
    return w * cdf1 + (1 - w) * cdf2


def ks_statistic_beta_mixtures(params1: dict, params2: dict) -> float:
    """
    Calculate the Kolmogorov-Smirnov statistic between two beta mixture
    distributions.

    Args:
        params1: Parameters of the first beta mixture
        params2: Parameters of the second beta mixture

    Returns:
        KS statistic value
    """
    # Create a grid of points in [0,1] for comparison
    x_grid = np.linspace(0, 1, 1000)

    # Calculate CDFs
    cdf1 = beta_mixture_cdf(
        x_grid,
        params1["w"],
        params1["alpha1"],
        params1["beta1"],
        params1["alpha2"],
        params1["beta2"],
    )

    cdf2 = beta_mixture_cdf(
        x_grid,
        params2["w"],
        params2["alpha1"],
        params2["beta1"],
        params2["alpha2"],
        params2["beta2"],
    )

    # KS statistic is the maximum absolute difference between CDFs
    return np.max(np.abs(cdf1 - cdf2))


def analyze_distributions_adaptive_stopping(
    answers_csv_path: Path,
    debates_csv_path: Path,
    fitting_method: str = "direct",
    max_rounds: Optional[int] = None,
    n_restarts: int = 2,
    verbose: bool = True,
    enforce_increasing_success: bool = False,
    extract_func: Callable = None,
    compare_func: Callable = None,
    adaptive_stopping: bool = False,
    ks_threshold: float = 0.05,
    stability_rounds: int = 2,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Analyze the correct rate distribution across debate rounds and fit
    Beta-Binomial mixture models to the data.

    Args:
        answers_csv_path: Path to CSV with correct answers ("id", "answer" columns)
        debates_csv_path: Path to CSV with debate rounds data
        fitting_method: Method to use for fitting Beta-Binomial mixtures ("em" or "direct")
        max_rounds: Maximum number of rounds to analyze (None for all)
        n_restarts: Number of random restarts for model fitting
        verbose: Whether to print progress and results
        enforce_increasing_success: Whether to enforce that expected success
                                   probability doesn't decrease across rounds
        extract_func: Function to extract and normalize responses
        compare_func: Function to compare normalized responses with correct answer
        adaptive_stopping: Whether to use adaptive stopping for fitting
        ks_threshold: Threshold for KS statistic to consider distributions stable
        stability_rounds: Number of consecutive rounds below threshold to stop

    Returns:
        tuple: (aggregated_df, fit_results) where:
            - aggregated_df: DataFrame with correct rate distribution per round
            - fit_results: List of dictionaries containing fitted model parameters
    """
    # Load answers data
    try:
        df_answers = pd.read_csv(answers_csv_path)

        # Only remove missing values, no numerical conversion
        df_answers.dropna(subset=["id"], inplace=True)
        if verbose:
            print(f"Loaded answers data from {answers_csv_path}")
            print(df_answers.head())
    except Exception as e:
        raise ValueError(f"Failed to load answers data: {e}")

    # Load debate rounds data
    try:
        df_debates = pd.read_csv(debates_csv_path)

        # Don't convert task_id to numeric, only clean missing values
        # But keep round_number as numeric since it's needed for analysis
        df_debates["round_number"] = pd.to_numeric(
            df_debates["round_number"], errors="coerce"
        )
        df_debates.dropna(subset=["task_id", "round_number"], inplace=True)
        df_debates["round_number"] = df_debates["round_number"].astype(int)
        if verbose:
            print(f"Loaded debate rounds from {debates_csv_path}")
            print(df_debates.head())
    except Exception as e:
        raise ValueError(f"Error loading debate rounds data: {e}")

    # Get aggregated data for all rounds
    try:
        if verbose:
            print("Calculating correct rate distribution...")
        aggregated_df = calculate_correct_rate_distribution(
            df_answers=df_answers,
            df_debates=df_debates,
            max_rounds=max_rounds,
            extract_func=extract_func,
            compare_func=compare_func,
        )
    except Exception as e:
        raise ValueError(f"Error calculating correct rate distribution: {e}")

    if aggregated_df.empty:
        raise ValueError("No data available for analysis.")

    # Print the aggregated DataFrame if verbose
    if verbose:
        print("Aggregated DataFrame:")
        print(aggregated_df)

    prev_fit_result = None
    fit_results = []
    prev_exp_success = None

    # Initialize adaptive stopping variables
    consecutive_stable_rounds = 0
    stopped_early = False

    # Process each round in the aggregated data
    for _, row in aggregated_df.iterrows():
        round_number = int(row["round_number"])
        if verbose:
            print(
                f"Processing round {round_number} using fitting method: {fitting_method}"
            )

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

        # Fit the model with constraints if requested
        if enforce_increasing_success and prev_exp_success is not None:
            fit_result = fit_mixture_beta_binomial_with_constraints(
                counts_array,
                k=k,
                fitting_method=fitting_method,
                n_restarts=n_restarts,
                prev_exp_success=prev_exp_success,
            )
        else:
            # Standard fitting without constraints
            fit_result = fit_mixture_beta_binomial(
                counts_array, k=k, fitting_method=fitting_method, n_restarts=n_restarts
            )

        # Ensure consistent ordering
        fit_result = ensure_consistent_component_ordering(fit_result)
        fit_results.append(fit_result)

        # Check adaptive stopping criteria
        if adaptive_stopping and prev_fit_result is not None:
            # Calculate KS statistic between current and previous distribution
            ks_stat = ks_statistic_beta_mixtures(fit_result, prev_fit_result)

            if ks_stat < ks_threshold:
                consecutive_stable_rounds += 1
                if verbose:
                    print(
                        f"  KS statistic: {ks_stat:.4f} (below threshold {ks_threshold})"
                    )
                    print(
                        f"  Consecutive stable rounds: {consecutive_stable_rounds}/{stability_rounds}"
                    )

                if consecutive_stable_rounds >= stability_rounds:
                    if verbose:
                        print(
                            f"Adaptive stopping criteria met after round {round_number}."
                        )
                        print(
                            f"Distribution has stabilized for {stability_rounds} consecutive rounds."
                        )
                    stopped_early = True
                    break
            else:
                consecutive_stable_rounds = 0
                if verbose:
                    print(
                        f"  KS statistic: {ks_stat:.4f} (above threshold {ks_threshold})"
                    )
                    print("  Consecutive stable rounds reset to 0")

        # Calculate expected success probability for next round constraints
        if enforce_increasing_success:
            w = fit_result["w"]
            alpha1 = fit_result["alpha1"]
            beta1 = fit_result["beta1"]
            alpha2 = fit_result["alpha2"]
            beta2 = fit_result["beta2"]

            exp1 = alpha1 / (alpha1 + beta1)
            exp2 = alpha2 / (alpha2 + beta2)

            # Weighted average of the two components' expected probabilities
            curr_exp_success = w * exp1 + (1 - w) * exp2
            prev_exp_success = curr_exp_success

            if verbose:
                print(f"  Expected success probability: {curr_exp_success:.4f}")

        if verbose:
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
                print(
                    f"    Δ Mixture weight: {fit_result['w'] - prev_fit_result['w']:.4f}"
                )
                print(
                    f"    Δ Alpha1: {fit_result['alpha1'] - prev_fit_result['alpha1']:.4f}"
                )
                print(
                    f"    Δ Beta1: {fit_result['beta1'] - prev_fit_result['beta1']:.4f}"
                )
                print(
                    f"    Δ Alpha2: {fit_result['alpha2'] - prev_fit_result['alpha2']:.4f}"
                )
                print(
                    f"    Δ Beta2: {fit_result['beta2'] - prev_fit_result['beta2']:.4f}"
                )
            print("-" * 80)

        prev_fit_result = fit_result.copy()

    if adaptive_stopping and verbose and not stopped_early:
        print("Adaptive stopping criteria not met after all rounds.")

    return aggregated_df, fit_results


# -------------------------------------------------------------------
# Example usage in main
# -------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    from multi_llm_debate.run.llm_bar.utils import (
        compare_llm_bar_response,
        extract_1_2_answer,
    )

    answers_csv_path = (Path("output/llm_bar/processed_data.csv"),)
    model_config = ("Llama-3_1-8B-Instruct(11)",)

    FIT_METHOD = "direct"  # "direct" or "em" optimization approach
    N_RESTARTS = 2  # Number of random restarts for more stable fitting
    ENFORCE_INCREASING = False  # Enforce non-decreasing expected success probability
    MAX_ROUNDS = None  # or an int
    OUTPUT_DIR = Path("output/visualizations/llm_bar")
    task_name = "LLMBar"
    analyze_distributions_adaptive_stopping(
        answers_csv_path=Path("output/llm_bar/processed_data.csv"),
        debates_csv_path=Path(
            "data/llm_bar/Llama-3_1-8B-Instruct(11)/debate_rounds.csv"
        ),
        fitting_method=FIT_METHOD,
        max_rounds=MAX_ROUNDS,
        n_restarts=N_RESTARTS,
        verbose=True,
        enforce_increasing_success=ENFORCE_INCREASING,
        extract_func=extract_1_2_answer,
        compare_func=compare_llm_bar_response,
        adaptive_stopping=True,
        ks_threshold=0.05,
        stability_rounds=2,
    )
