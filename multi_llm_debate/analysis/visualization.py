import logging
from typing import Dict, List, Tuple
import math
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_distribution_data(
    result_df: pd.DataFrame,
    round_number: int,
) -> Dict[str, float]:
    """Process distribution data to get percentages for each bin (e.g. '0', '1', '2', ...).

    Args:
        result_df: DataFrame with distribution data from calculate_correct_rate_distribution_for_round_n.
        round_number: The round number being processed.

    Returns:
        Dictionary mapping bin labels (strings) to percentage of tasks in that bin.
    """
    bin_columns = [col for col in result_df.columns if col.isdigit()]
    bin_columns.sort(key=int)

    if not bin_columns or result_df.empty:
        logger.warning(f"No bins found for round {round_number}")
        return {}

    task_count = len(result_df)
    bin_sums = result_df[bin_columns].sum()
    bin_percentages = (bin_sums / task_count * 100).to_dict()

    return bin_percentages
def plot_all_rounds_multi_rows(
    all_distributions: List[Tuple[int, Dict[str, float]]],
    output_dir: Path,
    rows: int = 2,
    show_plot: bool = False,
    plot_title: str = "Distribution of Correct Agents per Round",
    file_name: str = "all_rounds_plot.png"
) -> None:
    """Plot the round distributions in multiple rows of subplots.
    
    Each subplot corresponds to a single round. The number of rows can be 
    specified as an argument.
    
    Args:
        all_distributions: List of (round_number, bin_percentages) tuples.
        output_dir: Path to save the resulting figure.
        rows: Number of rows to arrange the subplots in.
        show_plot: Whether to display the plot interactively.
        plot_title: Title for the overall figure.
        file_name: Filename to save the plot as.
    """
    if not all_distributions:
        logger.warning("No distributions to plot.")
        return
    
    all_distributions = sorted(all_distributions, key=lambda x: x[0])
    num_rounds = len(all_distributions)

    # Calculate how many columns we need based on the specified rows
    num_cols = math.ceil(num_rounds / rows)

    # Create figure with subplots
    fig, axs = plt.subplots(
        nrows=rows,
        ncols=num_cols,
        figsize=(5 * num_cols, 5 * rows),
        sharey=True  # share the Y-axis for comparison
    )
    
    # Handle the case where we have a single row (axs would be 1D)
    if rows == 1:
        axs = [axs] if num_cols == 1 else axs
    
    # Flatten the array of axes for easier indexing
    axs = axs.ravel() if hasattr(axs, 'ravel') else axs

    max_value = 0  # Track the maximum value for consistent y-axis limits
    
    # Plot each round in its subplot
    for i, (round_number, bin_percentages) in enumerate(all_distributions):
        if i >= len(axs):
            break  # Safety check
            
        ax = axs[i]
        bins = [int(b) for b in sorted(bin_percentages.keys(), key=int)]
        values = [bin_percentages[str(b)] for b in bins]
        
        # Update max value for consistent y-axis scaling
        if values:
            max_value = max(max_value, max(values))

        # Create bar chart
        bars = ax.bar(bins, values)
        
        # Add text labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                fontsize=9,
            )
        
        ax.set_title(f"Round {round_number}", fontsize=12)
        ax.set_xlabel("Correct Agents", fontsize=10)
        
        # Add y-labels only to leftmost subplots in each row
        if i % num_cols == 0:
            ax.set_ylabel("Tasks (%)", fontsize=10)
            
        ax.grid(axis="y", alpha=0.3)
    
    # Set consistent y-axis limits across all subplots
    if max_value > 0:
        for ax in axs[:num_rounds]:
            ax.set_ylim(0, max_value * 1.2)  # Add 20% headroom for labels
    
    # Turn off any extra subplots
    for j in range(num_rounds, len(axs)):
        axs[j].axis("off")

    # Set overall figure title
    layout_desc = f"{rows}-row layout" if rows > 1 else "single row"
    fig.suptitle(f"{plot_title} ({layout_desc})", fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_path = output_dir / file_name
    plt.savefig(output_path, dpi=300)
    
    if show_plot:
        plt.show()
    plt.close()
    logger.info(f"Saved {rows}-row subplot figure to {output_path}")