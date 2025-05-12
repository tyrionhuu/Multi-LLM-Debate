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
logger.setLevel(logging.ERROR)


def plot_file_count_distribution(
    distribution: Dict[int, int],
    show_plot: bool = False,
    model_config: str = "",
    task_name: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """Creates a plot of file count distribution across directories.

    Args:
        distribution: Dictionary mapping file counts to number of directories.
        show_plot: Whether to display the plot interactively.
        model_config: Configuration name of the model being analyzed.
        task_name: Name of the task being analyzed.

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
    ax.set_title(
        "Distribution of File Counts Across Directories "
        + model_config
        + " "
        + task_name,
        fontsize=14,
    )
    # Show total entry count as subtitle
    total_entries = sum(dir_counts)
    ax.set_title(
        "Distribution of File Counts Across Directories "
        + model_config
        + " "
        + task_name,
        fontsize=14,
        loc="center",
    )
    ax.set_title(
        f"Total entries: {total_entries}",
        fontsize=10,
        loc="left",
        pad=30,
        color="dimgray",
    )

    ax.set_xlabel("Number of Files in Directory", fontsize=12)
    ax.set_ylabel("Number of Directories", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    if task_name:
        ax.set_title(
            f"Distribution of File Counts for {task_name} " + model_config, fontsize=14
        )

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


def combine_correct_rate_plots(
    df: pd.DataFrame,
    model_dirs: List[Path],
    model_configs: List[str],
    output_dir: Path,
    extract_func: Callable,
    compare_func: Callable,
    task_name: str = "",
    max_rounds: int = 6,
    rows: int = None,
    columns: int = None,
    show_plot: bool = False,
    combined_title: str = "Comparison of Correct Agent Distributions",
    file_name: str = "combined_correct_rate_plots.png",
    progress_bar: bool = False,
) -> None:
    """Combine multiple correct rate plots into a single figure with subplots.
    
    This version integrates the functionality of correct_rate_main directly,
    calculating distributions for each model configuration and displaying them
    together. It assumes the same dataframe, extract_func, compare_func, and
    task_name for all configurations, with only model_dirs and model_configs
    being different.
    
    Args:
        df: DataFrame containing the "ground truth" answer data for all models.
        model_dirs: List of directories containing model output data.
        model_configs: List of model configuration names (same length as model_dirs).
        output_dir: Directory where the combined plot should be saved.
        extract_func: Function to extract answers from the debate data.
        compare_func: Function to compare judge bench responses.
        task_name: Name of the task for all configurations.
        max_rounds: Maximum number of rounds to process for all configurations.
        rows: Number of rows in the subplot grid. If None, will be calculated.
        columns: Number of columns in the subplot grid. If None, will be calculated.
        show_plot: Whether to display the plot interactively.
        combined_title: Title for the overall combined figure.
        file_name: Name of the file to save the combined plot as.
        progress_bar: Whether to show progress bars during processing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(model_dirs) != len(model_configs):
        logger.error("model_dirs and model_configs must have the same length")
        return
    
    # Collect all distributions from processing each model configuration
    all_config_distributions = []
    config_labels = []
    
    # Process each model configuration
    for model_dir, model_config in zip(model_dirs, model_configs):
        config_output_dir = output_dir / model_config.replace(" ", "_")
        config_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing model: {model_config}")
        
        try:
            # Load the debate data for this model configuration
            df_debates = load_debate_data(model_dir)
            if df_debates is None:
                logger.error(f"Could not load debate data for {model_config}. Skipping.")
                continue

            logger.info(f"Processed debate data: {len(df_debates)} records")
            
            all_distributions = []
            
            # Process each round for this model configuration
            for round_number in range(max_rounds):
                logger.info(f"Processing round {round_number}...")

                try:
                    # Calculate distribution for this round
                    result_df = calculate_correct_rate_distribution_for_round_n(
                        df_answers=df,
                        df_debates=df_debates,
                        round_number=round_number,
                        extract_func=extract_func,
                        compare_func=compare_func,
                        progress_bar=progress_bar,
                    )

                    # Convert distribution to a simple dict for plotting
                    bin_percentages = process_distribution_data(result_df, round_number)

                    if bin_percentages:
                        all_distributions.append((round_number, bin_percentages))

                except Exception as err:
                    logger.error(f"Error processing round {round_number} for {model_config}: {err}")
            
            # If we have distributions for this model, create individual plots and add to collection
            if all_distributions:
                # Create the individual plot for this model
                title = f"Correct Agent Distribution by Round {model_config} - {task_name}"
                plot_all_rounds_multi_rows(
                    all_distributions=all_distributions,
                    output_dir=config_output_dir,
                    rows=2,
                    show_plot=False,  # Don't show individual plots
                    plot_title=title,
                    file_name=f"correct_agent_distribution_{model_config}_{task_name}.png",
                )
                
                # Create individual heatmap
                create_heatmap(
                    all_distributions=all_distributions,
                    output_dir=config_output_dir,
                    show_plot=False,  # Don't show individual heatmaps
                    file_name=f"correct_agent_heatmap_{model_config}_{task_name}.png",
                )
                
                # Add to collection for combined plot
                all_config_distributions.append(all_distributions)
                config_labels.append(f"{model_config}" if not task_name else 
                                   f"{model_config} - {task_name}")

        except Exception as e:
            logger.error(f"Error processing {model_config}: {e}")
    
    # Create combined plot if we have data from multiple models
    if not all_config_distributions:
        logger.warning("No distributions to plot in the combined figure.")
        return
    
    # Determine grid dimensions if not specified
    if rows is None and columns is None:
        # Default to a square-ish grid
        n_configs = len(all_config_distributions)
        columns = math.ceil(math.sqrt(n_configs))
        rows = math.ceil(n_configs / columns)
    elif rows is None:
        rows = math.ceil(len(all_config_distributions) / columns)
    elif columns is None:
        columns = math.ceil(len(all_config_distributions) / rows)
    
    # Create figure for subplots
    fig = plt.figure(figsize=(6 * columns, 5 * rows))
    
    # Plot each configuration's distributions
    for idx, (distributions, label) in enumerate(zip(all_config_distributions, config_labels)):
        if idx >= rows * columns:
            logger.warning(f"Not enough subplots for all configs. Skipping {len(all_config_distributions) - rows * columns} configs.")
            break
        
        # Create subplot
        ax = fig.add_subplot(rows, columns, idx + 1)
        
        # Add mini-heatmap to the subplot
        data = []
        for round_num, bin_percentages in distributions:
            for bin_label, percentage in bin_percentages.items():
                data.append({
                    "Round": round_num,
                    "Correct Agents": int(bin_label),
                    "Percentage": percentage
                })
        
        df_heatmap = pd.DataFrame(data)
        if not df_heatmap.empty:
            pivot_df = df_heatmap.pivot(
                index="Round", columns="Correct Agents", values="Percentage"
            ).fillna(0)
            
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                linewidths=0.5,
                ax=ax,
                cbar_kws={"label": "%"},
            )
            
            ax.set_title(label, fontsize=12)
    
    # Set overall figure title
    task_subtitle = f" for {task_name}" if task_name else ""
    fig.suptitle(f"{combined_title}{task_subtitle}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for the suptitle
    
    # Save figure
    output_path = output_dir / file_name
    plt.savefig(output_path, dpi=300)
    
    if show_plot:
        plt.show()
    plt.close()
    
    logger.info(f"Saved combined plot to {output_path}")


if __name__ == "__main__":
    import logging
    from pathlib import Path

    from multi_llm_debate.run.judge_bench.utils import (
        extract_caption_a_b_answer,
        compare_judge_bench_response,
        load_judge_bench_dataset,
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    df = load_judge_bench_dataset()
    OUTPUT_DIR = Path("../output/visualizations/judge_bench")
    task_name = "Judge Bench"

    model_dirs = [
        Path("data/judge_bench/gemma-3-4b-it(7)"),
        Path("data/judge_bench/Llama-3_1-8B-Instruct(7)"),
        Path("data/judge_bench/Qwen2_5-7B-Instruct(7)"),
        Path("data/judge_bench/gemini-2_0-flash-001(7)"),
    ]
    
    model_configs = [
        "Gemini-3-4B",
        "Llama-3-1-8B",
        "Qwen-2.5-7B",
        "Gemini-2.0-Flash",
    ]
    extract_func = extract_caption_a_b_answer
    compare_func = compare_judge_bench_response
    max_rounds = 5
    rows = 1
    columns = 5
    show_plot = True
    combined_title = "Comparison of Correct Agent Distributions"
    file_name = "combined_correct_rate_plots.png"
    progress_bar = True
    combine_correct_rate_plots(
        df,
        model_dirs,
        model_configs,
        OUTPUT_DIR,
        extract_func,
        compare_func,
        task_name=task_name,
        max_rounds=max_rounds,
        rows=rows,
        columns=columns,
        show_plot=show_plot,
        combined_title=combined_title,
        file_name=file_name,
        progress_bar=progress_bar,
    )