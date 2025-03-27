from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..analysis.calculate_correct_rate_distribution import (
    calculate_correct_rate_distribution,
)
from .model_fitting import (
    ensure_consistent_component_ordering,
    fit_mixture_beta_binomial,
    fit_mixture_beta_binomial_with_constraints,
)


def analyze_rounds_distribution(
    answers_csv_path: Path,
    debates_csv_path: Path,
    fitting_method: str = "direct",
    max_rounds: Optional[int] = None,
    n_restarts: int = 2,
    verbose: bool = True,
    enforce_increasing_success: bool = False,
    extract_func: Callable = None,
    compare_func: Callable = None,
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

    return aggregated_df, fit_results


# -------------------------------------------------------------------
# Example usage in main
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Example import from your code:
    # PATHS (placeholders in this example)
    ANSWERS_CSV = Path("output/bool_q/processed_data.csv")  # your "id" -> "answer" file
    DEBATES_CSV = Path(
        "data/bool_q/llama3(11)/debate_rounds.csv"
    )  # the debate rounds CSV
    MAX_ROUNDS = None  # or an int

    # Choose which method to use for fitting
    FIT_METHOD = "em"  # or "direct"

    # Set to True to enforce increasing success probability constraint
    ENFORCE_INCREASING_SUCCESS = True

    try:
        # Call the analysis function with our parameters
        aggregated_df, fit_results = analyze_rounds_distribution(
            answers_csv_path=ANSWERS_CSV,
            debates_csv_path=DEBATES_CSV,
            fitting_method=FIT_METHOD,
            max_rounds=MAX_ROUNDS,
            verbose=True,
            enforce_increasing_success=ENFORCE_INCREASING_SUCCESS,
        )
        print(f"Successfully analyzed {len(fit_results)} rounds")
    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)
