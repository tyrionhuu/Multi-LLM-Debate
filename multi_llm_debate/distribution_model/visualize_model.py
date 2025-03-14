#!/usr/bin/env python
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..analysis.calculate_correct_rate_distribution import (
    calculate_correct_rate_distribution,
)
from .model_fitting import (
    fit_mixture_beta_binomial,  # this wrapper lets us choose "em" or "direct"
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

    # Create the figure
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    # Plot mixture weight
    axes[0].plot(
        rounds, w_values, marker="o", linestyle="-", label="Mixture Weight (w)"
    )
    axes[0].set_title("Mixture Weight Evolution")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Mixture Weight")
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(rounds)

    # Plot alpha parameters
    axes[1].plot(rounds, alpha1_values, marker="o", linestyle="-", label="Alpha1")
    axes[1].plot(rounds, alpha2_values, marker="s", linestyle="-", label="Alpha2")
    axes[1].set_title("Alpha Parameters Evolution")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Alpha Value")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(rounds)

    # Plot beta parameters
    axes[2].plot(rounds, beta1_values, marker="o", linestyle="-", label="Beta1")
    axes[2].plot(rounds, beta2_values, marker="s", linestyle="-", label="Beta2")
    axes[2].set_title("Beta Parameters Evolution")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Beta Value")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    axes[2].set_xticks(rounds)

    plt.tight_layout()

    # Save if output directory is provided
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(output_dir / "parameter_evolution.png", dpi=300)

    return fig


if __name__ == "__main__":
    import sys

    DATA_PATH = Path("output/bool_q/processed_data.csv")
    MODEL_DIR_PATH = Path("data/bool_q/llama3(7)")
    OUTPUT_DIR = Path("output/visualizations")

    # Choose the fitting method: "em" or "direct"
    FIT_METHOD = "direct"  # Change to "direct" to use direct optimization approach

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load data
    try:
        dataframe = pd.read_csv(DATA_PATH)
        print(f"Loaded data from {DATA_PATH}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    if not MODEL_DIR_PATH.exists() or not MODEL_DIR_PATH.is_dir():
        print(f"Model directory does not exist: {MODEL_DIR_PATH}")
        sys.exit(1)

    # Get aggregated data for all rounds
    try:
        aggregated_df = calculate_correct_rate_distribution(
            dataframe=dataframe, model_dir=MODEL_DIR_PATH
        )
    except Exception as e:
        print(f"Error calculating correct rate distribution: {e}")
        sys.exit(1)

    if aggregated_df.empty:
        print("No data available for analysis")
        sys.exit(1)

    # Extract bin columns (representing correct counts)
    bin_columns = [col for col in aggregated_df.columns if col.isdigit()]
    bin_columns.sort(key=int)

    # Find the maximum bin value to use as k
    k = max(int(col) for col in bin_columns if col.isdigit())

    # Store fit results + observed data per round
    model_results = []
    observed_data = []

    # Process each row (each round) in aggregated_df
    for _, row in aggregated_df.iterrows():
        round_number = int(row["round_number"])
        print(f"Processing round {round_number} (fitting method='{FIT_METHOD}')...")

        # Build a dictionary of bin -> frequency
        counts_dict = {int(bin_col): int(row[bin_col]) for bin_col in bin_columns}
        observed_data.append(counts_dict)

        # Expand to an array of counts
        all_counts = []
        for count_val, freq in counts_dict.items():
            all_counts.extend([count_val] * freq)

        counts_array = np.array(all_counts)

        # Fit model (uses the chosen method)
        fit_result = fit_mixture_beta_binomial(
            counts_array, k=k, fitting_method=FIT_METHOD
        )
        model_results.append(fit_result)

        print(f"  Fitted model for round {round_number}:")
        print(f"    w={fit_result['w']:.4f}")
        print(f"    alpha1={fit_result['alpha1']:.4f}, beta1={fit_result['beta1']:.4f}")
        print(f"    alpha2={fit_result['alpha2']:.4f}, beta2={fit_result['beta2']:.4f}")

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

    # Show plots if desired
    plt.show()
