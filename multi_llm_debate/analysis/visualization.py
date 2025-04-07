import logging
import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .calculate_correct_rate_distribution import (
    calculate_correct_rate_distribution_for_round_n,
)
from .utils import load_debate_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_file_count_distribution(
    distribution: Dict[int, int], show_plot: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """Creates a plot of file count distribution across directories.

    Args:
        distribution: Dictionary mapping file counts to number of directories.
        show_plot: Whether to display the plot interactively.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The created plot figure and axes.
    """
    if not distribution:
        print("No directory data found to plot.")
        return None, None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    file_counts = list(distribution.keys())
    dir_counts = list(distribution.values())

    # Create bar chart
    bars = ax.bar(file_counts, dir_counts, color="salmon", edgecolor="darkred")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{int(height)}",
            ha="center",
            fontsize=9,
        )

    # Set chart attributes
    ax.set_title("Distribution of File Counts Across Directories", fontsize=14)
    ax.set_xlabel("Number of Files in Directory", fontsize=12)
    ax.set_ylabel("Number of Directories", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Adjust x-axis to show all integer values
    ax.set_xticks(range(min(file_counts), max(file_counts) + 1))

    plt.tight_layout()

    if show_plot:
        plt.show()
    logger.info("File count distribution plot created.")

    return fig, ax


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
    file_name: str = "all_rounds_plot.png",
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
        sharey=True,  # share the Y-axis for comparison
    )

    # Handle the case where we have a single row (axs would be 1D)
    if rows == 1:
        axs = [axs] if num_cols == 1 else axs

    # Flatten the array of axes for easier indexing
    axs = axs.ravel() if hasattr(axs, "ravel") else axs

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
    fig.suptitle(plot_title, fontsize=14)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    output_path = output_dir / file_name
    plt.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()
    plt.close()
    logger.info(f"Saved {rows}-row subplot figure to {output_path}")


def create_heatmap(
    all_distributions: List[Tuple[int, Dict[str, float]]],
    output_dir: Path,
    show_plot: bool = False,
    file_name: str = "correct_agent_heatmap.png",
) -> None:
    """Create a heatmap showing the evolution of distributions across rounds.

    Args:
        all_distributions: List of (round_number, bin_percentages) tuples.
        output_dir: Directory where the plot should be saved.
        show_plot: Whether to display the plot interactively.
    """
    if not all_distributions:
        logger.warning("No data to create heatmap")
        return

    # Create a DataFrame from the collected data
    data = []
    for round_num, bin_percentages in all_distributions:
        for bin_label, percentage in bin_percentages.items():
            data.append(
                {
                    "Round": round_num,
                    "Correct Agents": int(bin_label),
                    "Percentage": percentage,
                }
            )

    df = pd.DataFrame(data)

    # Create pivot table for heatmap
    pivot_df = df.pivot(
        index="Round", columns="Correct Agents", values="Percentage"
    ).fillna(0)

    # Create heatmap plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Percentage of Tasks (%)"},
    )

    plt.title("Evolution of Correct Agent Distribution Across Rounds", fontsize=16)
    plt.tight_layout()

    # Save the heatmap
    output_path = output_dir / file_name
    plt.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()
    plt.close()

    logger.info(f"Saved heatmap to {output_path}")


def correct_rate_main(
    data_path: Path,
    model_dir: Path,
    output_dir: Path,
    extract_func: Callable,
    compare_func: Callable,
    max_rounds: int = 6,
    show_plots: bool = False,
    model_config: str = "",
    show_heatmap: bool = False,
) -> None:
    """
    Loads data, calculates correct-rate distributions for each round,
    and plots them in two rows of subplots.

    Args:
        data_path: Path to the CSV file with task data (contains correct labels).
        model_dir: Directory containing model output data.
        output_dir: Directory where output plots should be saved.
        extract_func: Function to extract answers from the debate data.
        compare_func: Function to compare judge bench responses.
        max_rounds: Maximum number of rounds to process.
        show_plots: Whether to display plots interactively.
        show_heatmap: Whether to display the heatmap interactively.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load the "ground truth" answer data
        df_answers = pd.read_csv(data_path)
        logger.info(f"Loaded answer data from {data_path}")

        # Load the debate data
        df_debates = load_debate_data(model_dir)
        if df_debates is None:
            logger.error("Could not load debate data. Aborting.")
            return

        logger.info(f"Processed debate data: {len(df_debates)} records")

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    all_distributions = []

    # Process each round from 0 up to max_rounds-1
    for round_number in range(max_rounds):
        logger.info(f"Processing round {round_number}...")

        try:
            # Calculate distribution for this round
            result_df = calculate_correct_rate_distribution_for_round_n(
                df_answers=df_answers,
                df_debates=df_debates,
                round_number=round_number,
                extract_func=extract_func,
                compare_func=compare_func,
            )

            # Convert that distribution to a simple dict for plotting
            bin_percentages = process_distribution_data(result_df, round_number)

            if bin_percentages:
                all_distributions.append((round_number, bin_percentages))

        except Exception as err:
            logger.error(f"Error processing round {round_number}: {err}")

    # Create the single, combined plot in 2 rows if we have data
    title = f"Correct Agent Distribution by Round ({model_config})"
    if all_distributions:
        plot_all_rounds_multi_rows(
            all_distributions=all_distributions,
            output_dir=output_dir,
            rows=2,
            show_plot=show_plots,
            plot_title=title,
            file_name=f"correct_agent_distribution_{model_config}.png",
        )

    logger.info("Visualization complete!")

    # Create heatmap
    create_heatmap(
        all_distributions=all_distributions,
        output_dir=output_dir,
        show_plot=show_heatmap,
        file_name=f"correct_agent_heatmap_{model_config}.png",
    )
    
    logger.info("Heatmap creation complete.")


def main() -> None:
    """Test function for file count distribution visualization.

    Analyzes the file count in the Qwen2_5-3B-Instruct(11) directory and
    visualizes the distribution using plot_file_count_distribution.
    """
    # Set path to the model directory
    model_dir_path = Path(
        "/Users/tyrionhuu/projects/research_projects/Multi-LLM-Debate/data/judge_bench/Qwen2_5-3B-Instruct(11)"
    )

    if not model_dir_path.exists():
        print(f"Error: Directory {model_dir_path} does not exist.")
        return

    print(f"Analyzing file count distribution in: {model_dir_path}")

    # Count files in each subdirectory
    distribution = {}
    for subdir in [d for d in model_dir_path.iterdir() if d.is_dir()]:
        file_count = len([f for f in subdir.iterdir() if f.is_file()])

        # Add to distribution
        if file_count in distribution:
            distribution[file_count] += 1
        else:
            distribution[file_count] = 1

    if not distribution:
        print(f"No subdirectories with files found in {model_dir_path}")
        return

    print(f"Found file count distribution: {distribution}")

    # Visualize the distribution
    fig, ax = plot_file_count_distribution(distribution)
    if fig:
        plt.show()
    print("Plot displayed. Close the plot window to exit.")


if __name__ == "__main__":
    main()
