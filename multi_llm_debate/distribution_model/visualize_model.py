import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .fit_distribution import analyze_rounds_distribution
from .pmf import beta_binomial_pmf


def plot_mixture_model(
    params: Dict[str, float],
    k: int,
    observed_data: Optional[Dict[int, int]] = None,
    title: str = "Beta-Binomial Mixture Model",
    ax: Optional[plt.Axes] = None,
    color: str = "blue",
    alpha: float = 0.7,
) -> plt.Axes:
    """Plot a fitted Beta-Binomial mixture model.

    Args:
        params: Dictionary with the fitted model parameters
            (w, alpha1, beta1, alpha2, beta2)
        k: Number of trials
        observed_data: Optional dictionary mapping bin values to counts
        title: Plot title
        ax: Optional matplotlib axes to plot on
        color: Color for the plot
        alpha: Transparency for the plot

    Returns:
        The matplotlib axes object with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Extract model parameters
    w = params["w"]
    alpha1 = params["alpha1"]
    beta1 = params["beta1"]
    alpha2 = params["alpha2"]
    beta2 = params["beta2"]

    # Generate points for plotting
    x = np.arange(k + 1)

    # Calculate PMF values for each component and the mixture
    y_comp1 = [beta_binomial_pmf(s, k, alpha1, beta1) for s in x]
    y_comp2 = [beta_binomial_pmf(s, k, alpha2, beta2) for s in x]
    y_mixture = [w * y_comp1[i] + (1 - w) * y_comp2[i] for i in range(len(x))]

    # Plot the mixture model
    ax.plot(
        x,
        y_mixture,
        marker="o",
        linestyle="-",
        color=color,
        alpha=alpha,
        label="Combined Mixture Model",
    )

    # Plot the individual components
    ax.plot(
        x,
        [w * y for y in y_comp1],
        linestyle="--",
        color=color,
        alpha=alpha * 0.7,
        label=f"Component 1 (weight={w:.2f})",
    )
    ax.plot(
        x,
        [(1 - w) * y for y in y_comp2],
        linestyle=":",
        color=color,
        alpha=alpha * 0.7,
        label=f"Component 2 (weight={1-w:.2f})",
    )

    # Plot observed data if provided
    if observed_data is not None:
        # Normalize counts to get probabilities
        total_count = sum(observed_data.values())
        observed_probs = {
            kval: cnt / total_count for kval, cnt in observed_data.items()
        }

        # Plot as a bar chart
        obs_y = [observed_probs.get(i, 0) for i in range(k + 1)]
        ax.bar(range(k + 1), obs_y, alpha=0.3, color="gray", label="Observed Data")

    # Configure the plot
    ax.set_title(title)
    ax.set_xlabel("Number of Agents with Correct Response")
    ax.set_ylabel("Probability Mass")
    ax.set_xticks(range(k + 1))
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


def plot_model_evolution(
    model_results: List[Dict[str, float]],
    k: int,
    observed_data: List[Dict[int, int]],
    output_dir: Optional[Path] = None,
    model_config: str = "",
    row_number: int = 2,
    task_name: str = "judge_bench",
) -> List[Figure]:
    """Plot the evolution of the mixture model across rounds.

    Args:
        model_results: List of dictionaries with fitted model parameters for each round
        k: Number of trials
        observed_data: List of dictionaries mapping bin values to counts for each round
        output_dir: Optional directory to save the plots
        model_config: Optional model configuration identifier for file naming
        row_number: Number of rows for the subplot grid (default: 2)
        task_name: Task name for the title

    Returns:
        List of generated figures
    """
    figures = []

    # Create a figure for all rounds combined
    fig, axes = plt.subplots(
        nrows=row_number,
        ncols=math.ceil(len(model_results) / row_number),
        figsize=(15, 5 * row_number),
    )
    axes = axes.flatten()

    # Create combined plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_results)))

    config_suffix = f"_{model_config}" if model_config else ""

    for i, (params, obs_data) in enumerate(zip(model_results, observed_data)):
        # Plot in the combined figure
        title = f"Debate Round {i+1}"
        plot_mixture_model(
            params,
            k,
            obs_data,
            title=title,
            ax=axes[i],
            color=colors[i],
        )

    # Hide any unused subplots
    for j in range(len(model_results), len(axes)):
        axes[j].set_visible(False)

    # Adjust the combined figure layout
    plt.tight_layout()
    fig.suptitle(
        f"Agent Performance Distribution Across All Debate Rounds {model_config} - {task_name}",
        fontsize=16,
    )
    fig.subplots_adjust(top=0.93)  # Make room for the title

    # Save the combined figure if output directory is provided
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(
            output_dir / f"agent_performance_all_rounds{config_suffix}.png", dpi=300
        )

    figures.append(fig)

    return figures


def visualize_parameter_trends(
    model_results: List[Dict[str, float]],
    output_dir: Optional[Path] = None,
    model_config: str = "",
    task_name: str = "judge_bench",
) -> Figure:
    """Visualize how model parameters change across rounds.

    Args:
        model_results: List of dictionaries with fitted model parameters for each round
        output_dir: Optional directory to save the plot
        model_config: Optional model configuration identifier for file naming
        task_name: Task name for the title

    Returns:
        The generated figure
    """
    # Extract parameters for each round
    rounds = list(range(len(model_results)))
    w_values = [r["w"] for r in model_results]
    alpha1_values = [r["alpha1"] for r in model_results]
    beta1_values = [r["beta1"] for r in model_results]
    alpha2_values = [r["alpha2"] for r in model_results]
    beta2_values = [r["beta2"] for r in model_results]

    # Calculate derived parameters
    # Expected success probabilities: alpha/(alpha+beta)
    success_prob1_values = [
        alpha1 / (alpha1 + beta1) for alpha1, beta1 in zip(alpha1_values, beta1_values)
    ]
    success_prob2_values = [
        alpha2 / (alpha2 + beta2) for alpha2, beta2 in zip(alpha2_values, beta2_values)
    ]

    # Expected failure probabilities: beta/(alpha+beta)
    failure_prob1_values = [
        beta1 / (alpha1 + beta1) for alpha1, beta1 in zip(alpha1_values, beta1_values)
    ]
    failure_prob2_values = [
        beta2 / (alpha2 + beta2) for alpha2, beta2 in zip(alpha2_values, beta2_values)
    ]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    axes = axes.flatten()  # Flatten for easier indexing
    fig.suptitle(
        f"Model Parameter Evolution Across Debate Rounds {model_config} - {task_name}",
        fontsize=16,
    )
    fig.subplots_adjust(top=0.92)  # Make room for the title

    # Plot mixture weight
    axes[0].plot(
        rounds, w_values, marker="o", linestyle="-", label="Mixture Weight (w)"
    )
    axes[0].set_title("Component 1 Weight Evolution")
    axes[0].set_xlabel("Round Number")
    axes[0].set_ylabel("Mixture Weight (w)")
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(rounds)
    axes[0].set_xticklabels([str(i + 1) for i in rounds])  # Just integers

    # Plot expected success probabilities
    axes[1].plot(
        rounds,
        success_prob1_values,
        marker="o",
        linestyle="-",
        label="Component 1: α₁/(α₁+β₁)",
        color="green",
    )
    axes[1].plot(
        rounds,
        success_prob2_values,
        marker="s",
        linestyle="-",
        label="Component 2: α₂/(α₂+β₂)",
        color="purple",
    )
    axes[1].set_title("Success Probability Evolution by Component")
    axes[1].set_xlabel("Round Number")
    axes[1].set_ylabel("Expected Success Probability")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(rounds)
    axes[1].set_xticklabels([str(i + 1) for i in rounds])  # Just integers
    axes[1].set_ylim(0, 1)  # Probabilities are between 0 and 1

    # Plot expected failure probabilities
    axes[2].plot(
        rounds,
        failure_prob1_values,
        marker="o",
        linestyle="-",
        label="Component 1: β₁/(α₁+β₁)",
        color="green",
    )
    axes[2].plot(
        rounds,
        failure_prob2_values,
        marker="s",
        linestyle="-",
        label="Component 2: β₂/(α₂+β₂)",
        color="purple",
    )
    axes[2].set_title("Failure Probability Evolution by Component")
    axes[2].set_xlabel("Round Number")
    axes[2].set_ylabel("Expected Failure Probability")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_xticks(rounds)
    axes[2].set_xticklabels([str(i + 1) for i in rounds])  # Just integers
    axes[2].set_ylim(0, 1)  # Probabilities are between 0 and 1

    # Plot alpha parameters
    axes[3].plot(
        rounds,
        alpha1_values,
        marker="o",
        linestyle="-",
        label="α₁ (Component 1)",
        color="green",
    )
    axes[3].plot(
        rounds,
        alpha2_values,
        marker="s",
        linestyle="-",
        label="α₂ (Component 2)",
        color="purple",
    )
    axes[3].set_title("Alpha Parameter Evolution (Success Parameter)")
    axes[3].set_xlabel("Round Number")
    axes[3].set_ylabel("Alpha Value")
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    axes[3].set_xticks(rounds)
    axes[3].set_xticklabels([str(i + 1) for i in rounds])  # Just integers

    # Plot beta parameters
    axes[4].plot(
        rounds,
        beta1_values,
        marker="o",
        linestyle="-",
        label="β₁ (Component 1)",
        color="green",
    )
    axes[4].plot(
        rounds,
        beta2_values,
        marker="s",
        linestyle="-",
        label="β₂ (Component 2)",
        color="purple",
    )
    axes[4].set_title("Beta Parameter Evolution (Failure Parameter)")
    axes[4].set_xlabel("Round Number")
    axes[4].set_ylabel("Beta Value")
    axes[4].legend()
    axes[4].grid(alpha=0.3)
    axes[4].set_xticks(rounds)
    axes[4].set_xticklabels([str(i + 1) for i in rounds])  # Just integers

    # Hide the unused 6th subplot
    axes[5].set_visible(False)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)  # Make room for the title

    # Save if output directory is provided
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        config_suffix = f"_{model_config}" if model_config else ""
        fig.savefig(
            output_dir / f"model_parameter_evolution{config_suffix}_{task_name}.png",
            dpi=300,
        )

    return fig


def run_visualization(
    answers_csv_path: Path,
    debates_csv_path: Path,
    output_dir: Path,
    max_rounds: Optional[int] = None,
    fitting_method: str = "direct",
    n_restarts: int = 2,
    verbose: bool = True,
    enforce_increasing_success: bool = False,
    extract_func: Callable = None,
    compare_func: Callable = None,
    model_config: str = "",
    row_number: int = 2,
    task_name: str = "judge_bench",
    ks_threshold: float = 0.05,
    adaptive_stopping: bool = False,
) -> tuple[pd.DataFrame, list[dict], List[Figure]]:
    """Run the complete visualization pipeline from data loading to generating plots.

    Args:
        answers_csv_path: Path to CSV with correct answers
        debates_csv_path: Path to CSV with debate rounds data
        output_dir: Directory to save visualization outputs
        max_rounds: Maximum number of rounds to analyze (None for all)
        fitting_method: Method to use for fitting models ("em" or "direct")
        n_restarts: Number of random restarts for model fitting
        verbose: Whether to print progress information
        enforce_increasing_success: Whether to ensure expected success probability
                                    doesn't decrease
        extract_func: Function to extract answers from responses
        compare_func: Function to compare extracted answers with ground truth
        model_config: Model configuration identifier for file naming
        row_number: Number of rows for the model evolution subplot grid (default: 2)
        task_name: Task name for the title
        ks_threshold: Threshold for KS test to determine model fit
        adaptive_stopping: Whether to use adaptive stopping criteria

    Returns:
        tuple: (aggregated_df, model_results, figures) containing the analysis
               results and generated figures
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    if verbose:
        print("Analyzing debate rounds and fitting models...")

    # Use the analysis function
    aggregated_df, model_results = analyze_rounds_distribution(
        answers_csv_path=answers_csv_path,
        debates_csv_path=debates_csv_path,
        fitting_method=fitting_method,
        max_rounds=max_rounds,
        n_restarts=n_restarts,
        verbose=verbose,
        enforce_increasing_success=enforce_increasing_success,
        extract_func=extract_func,
        compare_func=compare_func,
        ks_threshold=ks_threshold,
        adaptive_stopping=adaptive_stopping,
    )

    if aggregated_df.empty:
        if verbose:
            print("No data available for analysis")
        return aggregated_df, model_results, []

    # Extract bin columns (representing correct counts)
    bin_columns = [col for col in aggregated_df.columns if col.isdigit()]
    bin_columns.sort(key=int)

    # Find the maximum bin value to use as k
    k = max(int(col) for col in bin_columns if col.isdigit())

    # Prepare observed data for visualization
    observed_data = []
    for _, row in aggregated_df.iterrows():
        # Build a dictionary of bin -> frequency
        counts_dict = {int(bin_col): int(row[bin_col]) for bin_col in bin_columns}
        observed_data.append(counts_dict)

    # Generate visualizations
    if verbose:
        print("Generating visualizations...")

    figures = []

    # 1) Plot evolution of each round in subplots and individual figures
    evolution_figs = plot_model_evolution(
        model_results,
        k,
        observed_data,
        output_dir=output_dir,
        model_config=model_config,
        row_number=row_number,
        task_name=task_name,
    )
    if verbose:
        print(f"Saved model evolution plots to {output_dir}")
    figures.extend(evolution_figs)

    # 2) Plot parameter trends across rounds
    param_fig = visualize_parameter_trends(
        model_results,
        output_dir=output_dir,
        model_config=model_config,
        task_name=task_name,
    )
    if verbose:
        print(f"Saved parameter trend plot to {output_dir}")
    figures.append(param_fig)

    return aggregated_df, model_results, figures


if __name__ == "__main__":
    from multi_llm_debate.run.judge_bench.utils import (
        compare_judge_bench_response,
        extract_caption_a_b_answer,
    )

    OUTPUT_DIR = Path("output/judge_bench/visualization")
    MAX_ROUNDS = None  # or an int

    # Analysis settings
    FIT_METHOD = "direct"  # "direct" or "em" optimization approach
    N_RESTARTS = 2  # Number of random restarts for more stable fitting
    ENFORCE_INCREASING = False  # Enforce non-decreasing expected success probability

    # Call the visualization pipeline function
    run_visualization(
        answers_csv_path=Path("output/judge_bench/processed_data.csv"),
        debates_csv_path=Path("data/judge_bench/llama3(11)/debate_rounds.csv"),
        output_dir=OUTPUT_DIR,
        max_rounds=MAX_ROUNDS,
        fitting_method=FIT_METHOD,
        n_restarts=N_RESTARTS,
        verbose=True,
        enforce_increasing_success=ENFORCE_INCREASING,
        extract_func=extract_caption_a_b_answer,
        compare_func=compare_judge_bench_response,
        model_config="llama3(11)",
    )

    run_visualization(
        answers_csv_path=Path("output/judge_bench/processed_data.csv"),
        debates_csv_path=Path("data/judge_bench/gemma2:2b(11)/debate_rounds.csv"),
        output_dir=OUTPUT_DIR,
        max_rounds=MAX_ROUNDS,
        fitting_method=FIT_METHOD,
        n_restarts=N_RESTARTS,
        verbose=True,
        enforce_increasing_success=ENFORCE_INCREASING,
        extract_func=extract_caption_a_b_answer,
        compare_func=compare_judge_bench_response,
        model_config="gemma2:2b(11)",
    )
