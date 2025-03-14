#!/usr/bin/env python
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .model_fitting import (
    analyze_rounds_distribution,  # new function for analyzing rounds
)
from .model_fitting import beta_binomial_pmf


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
        label="Mixture Model",
    )

    # Plot the individual components
    ax.plot(
        x,
        [w * y for y in y_comp1],
        linestyle="--",
        color=color,
        alpha=alpha * 0.7,
        label="Component 1",
    )
    ax.plot(
        x,
        [(1 - w) * y for y in y_comp2],
        linestyle=":",
        color=color,
        alpha=alpha * 0.7,
        label="Component 2",
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
        ax.bar(range(k + 1), obs_y, alpha=0.3, color="gray", label="Observed")

    # Configure the plot
    ax.set_title(title)
    ax.set_xlabel("Number of Correct Agents")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(k + 1))
    ax.legend()
    ax.grid(alpha=0.3)

    return ax


def plot_model_evolution(
    model_results: List[Dict[str, float]],
    k: int,
    observed_data: List[Dict[int, int]],
    output_dir: Optional[Path] = None,
) -> List[Figure]:
    """Plot the evolution of the mixture model across rounds.

    Args:
        model_results: List of dictionaries with fitted model parameters for each round
        k: Number of trials
        observed_data: List of dictionaries mapping bin values to counts for each round
        output_dir: Optional directory to save the plots

    Returns:
        List of generated figures
    """
    figures = []

    # Create a figure for all rounds combined
    fig, axes = plt.subplots(
        nrows=2, ncols=math.ceil(len(model_results) / 2), figsize=(15, 10)
    )
    axes = axes.flatten()

    # Create individual plots and add to the combined figure
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_results)))

    for i, (params, obs_data) in enumerate(zip(model_results, observed_data)):
        # Plot in the combined figure
        title = f"Round {i}"
        plot_mixture_model(
            params,
            k,
            obs_data,
            title=title,
            ax=axes[i],
            color=colors[i],
        )

        # Create individual figure
        fig_ind = plt.figure(figsize=(10, 6))
        ax_ind = fig_ind.add_subplot(111)

        plot_mixture_model(
            params,
            k,
            obs_data,
            title=f"Beta-Binomial Mixture Model - Round {i}",
            ax=ax_ind,
            color=colors[i],
        )

        figures.append(fig_ind)

        # Optionally save each figure individually
        # if output_dir is not None:
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     fig_ind.savefig(output_dir / f"mixture_model_round_{i}.png", dpi=300)

    # Adjust the combined figure layout
    plt.tight_layout()

    # Save the combined figure if output directory is provided
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(output_dir / "mixture_models_all_rounds.png", dpi=300)

    figures.append(fig)

    return figures


def visualize_parameter_trends(
    model_results: List[Dict[str, float]],
    output_dir: Optional[Path] = None,
) -> Figure:
    """Visualize how model parameters change across rounds.

    Args:
        model_results: List of dictionaries with fitted model parameters for each round
        output_dir: Optional directory to save the plot

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
    success_prob1_values = [alpha1/(alpha1+beta1) for alpha1, beta1 
                            in zip(alpha1_values, beta1_values)]
    success_prob2_values = [alpha2/(alpha2+beta2) for alpha2, beta2 
                            in zip(alpha2_values, beta2_values)]
    
    # Expected failure probabilities: beta/(alpha+beta)
    failure_prob1_values = [beta1/(alpha1+beta1) for alpha1, beta1 
                           in zip(alpha1_values, beta1_values)]
    failure_prob2_values = [beta2/(alpha2+beta2) for alpha2, beta2 
                           in zip(alpha2_values, beta2_values)]

    # Create the figure with 5 subplots
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 20))

    # Plot mixture weight
    axes[0].plot(
        rounds, w_values, marker="o", linestyle="-", label="Mixture Weight (w)"
    )
    axes[0].set_title("Mixture Weight Evolution")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Mixture Weight")
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(rounds)

    # Plot expected success probabilities
    axes[1].plot(rounds, success_prob1_values, marker="o", linestyle="-", 
                label="Comp 1: α₁/(α₁+β₁)", color="green")
    axes[1].plot(rounds, success_prob2_values, marker="s", linestyle="-", 
                label="Comp 2: α₂/(α₂+β₂)", color="purple")
    axes[1].set_title("Expected Success Probability Evolution")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Success Probability")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(rounds)
    axes[1].set_ylim(0, 1)  # Probabilities are between 0 and 1

    # Plot expected failure probabilities
    axes[2].plot(rounds, failure_prob1_values, marker="o", linestyle="-", 
                label="Comp 1: β₁/(α₁+β₁)", color="green")
    axes[2].plot(rounds, failure_prob2_values, marker="s", linestyle="-", 
                label="Comp 2: β₂/(α₂+β₂)", color="purple")
    axes[2].set_title("Expected Failure Probability Evolution")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Failure Probability")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_xticks(rounds)
    axes[2].set_ylim(0, 1)  # Probabilities are between 0 and 1

    # Plot alpha parameters
    axes[3].plot(rounds, alpha1_values, marker="o", linestyle="-", label="Alpha1")
    axes[3].plot(rounds, alpha2_values, marker="s", linestyle="-", label="Alpha2")
    axes[3].set_title("Alpha Parameters Evolution")
    axes[3].set_xlabel("Round")
    axes[3].set_ylabel("Alpha Value")
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    axes[3].set_xticks(rounds)

    # Plot beta parameters
    axes[4].plot(rounds, beta1_values, marker="o", linestyle="-", label="Beta1")
    axes[4].plot(rounds, beta2_values, marker="s", linestyle="-", label="Beta2")
    axes[4].set_title("Beta Parameters Evolution")
    axes[4].set_xlabel("Round")
    axes[4].set_ylabel("Beta Value")
    axes[4].legend()
    axes[4].grid(alpha=0.3)
    axes[4].set_xticks(rounds)

    plt.tight_layout()

    # Save if output directory is provided
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(output_dir / "parameter_evolution.png", dpi=300)

    return fig


if __name__ == "__main__":
    import sys

    # Define paths for input and output
    ANSWERS_CSV = Path("output/bool_q/processed_data.csv")  # id -> answer file
    DEBATES_CSV = Path("data/bool_q/llama3(11)/debate_rounds.csv")  # debate rounds data
    OUTPUT_DIR = Path("output/visualizations/bool_q")
    MAX_ROUNDS = None  # or an int

    # Choose the fitting method: "em" or "direct"
    FIT_METHOD = "direct"  # Change to "direct" to use direct optimization approach
    N_RESTARTS = 2  # Number of random restarts for more stable fitting

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    try:
        # Use the new analysis function
        print("Analyzing debate rounds and fitting models...")
        aggregated_df, model_results = analyze_rounds_distribution(
            answers_csv_path=ANSWERS_CSV,
            debates_csv_path=DEBATES_CSV,
            fitting_method=FIT_METHOD,
            max_rounds=MAX_ROUNDS,
            n_restarts=N_RESTARTS,
            verbose=True,
        )
    except Exception as e:
        print(f"Error in analysis: {e}")
        sys.exit(1)

    if aggregated_df.empty:
        print("No data available for analysis")
        sys.exit(1)

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
    print("Generating visualizations...")

    # 1) Plot evolution of each round in subplots and individual figures
    evolution_figs = plot_model_evolution(
        model_results, k, observed_data, output_dir=OUTPUT_DIR
    )
    print(f"Saved model evolution plots to {OUTPUT_DIR}")

    # 2) Plot parameter trends across rounds
    param_fig = visualize_parameter_trends(model_results, output_dir=OUTPUT_DIR)
    print(f"Saved parameter trend plot to {OUTPUT_DIR}")
