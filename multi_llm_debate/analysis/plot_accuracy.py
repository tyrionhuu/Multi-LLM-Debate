import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .calculate_correct_rate_by_round import calculate_correct_rate_by_round
from .calculate_task_accuracy import analyze_task_accuracy

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def create_plot_majority_aggregated(
    aggregated_majority_by_round: np.ndarray,
    model_name: str,
    output_dir: Path,
    max_round_number: int = 10,
    task_name: str = "Judge Bench",
) -> None:
    """Plots the aggregated majority accuracy value by round.

    Args:
        aggregated_majority_by_round: Array of majority correct rates aggregated
            across all accuracies for each round.
        model_name: Name of the model for the plot title.
        output_dir: Directory to save the plot.
        max_round_number: Maximum round number for the plot (default is 10).
        task_name: Name of the task (default is "Judge Bench").

    """
    # Create x-axis values (round numbers)
    rounds = np.arange(len(aggregated_majority_by_round))

    # Plot the aggregated majority correct rate
    plt.figure(figsize=(10, 6))

    plt.plot(
        rounds,
        aggregated_majority_by_round,
        color="tab:blue",
        linestyle="-",
        linewidth=2,
        label="Aggregated Majority Correct Rate",
    )

    # Title and labels
    plt.title(
        f"Aggregated Majority Correct Rate by Round: {model_name} - {task_name}", pad=15
    )
    plt.xlabel("Round Number")
    plt.ylabel("Correct Rate")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.legend()

    # Set y-axis limits and ticks
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(range(min(11, max_round_number + 1)))

    plt.tight_layout()

    # Save the plot first
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{model_name}_majority_aggregated.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

    # Then show it (if needed)
    plt.show()


def process_multiple_models_majority_aggregated(
    model_dirs: list[Path],
    output_dir: Path,
    dataframe: pd.DataFrame,
    task_name: str = "Judge Bench",
    max_round_number: int = 10,
    extract_func: Callable = None,
    compare_func: Callable = None,
    custom_model_names: list[str] = None,
    plot_title: str = None,
) -> None:
    """Process multiple model data and create visualizations comparing their aggregated majority correct rates.

    Args:
        model_dirs: List of paths to the model directories containing debate data.
        output_dir: Path to the output directory for saving visualizations.
        dataframe: DataFrame containing task data.
        task_name: Name of the task (default is "Judge Bench").
        max_round_number: Maximum round number for the correct rate calculation.
        extract_func: Function to extract and normalize responses.
        compare_func: Function to compare normalized responses with correct answer.
        custom_model_names: Optional list of custom model names for the legend.
        plot_title: Optional custom plot title.

    This function:
    1) Analyzes accuracy for each model
    2) Calculates majority correct rates for each unique accuracy value
    3) Aggregates the results across all accuracy levels for each model
    4) Plots all models together in a single graph
    """
    # List to store aggregated majority values for each model
    all_models_data = []
    model_names = []

    # Process each model directory
    for idx, model_dir in enumerate(model_dirs):
        if custom_model_names and idx < len(custom_model_names):
            model_name = custom_model_names[idx]
        else:
            model_name = model_dir.name
        model_names.append(model_name)
        logger.info(f"\nProcessing model: {model_name}")

        # 1) Analyze accuracy
        result_df = analyze_task_accuracy(
            model_dir=model_dir,
            dataframe=dataframe,
            extract_func=extract_func,
            compare_func=compare_func,
        )

        # 2) Get all unique accuracy values from the result dataframe
        unique_accuracies = result_df["accuracy"].unique()

        # 3) Initialize a list to store the majority values by round for each accuracy
        majority_by_rounds_list = []

        length = len(result_df)
        # 4) For each unique accuracy value, calculate majority correct rates by round
        for accuracy in unique_accuracies:
            if accuracy < 0:
                continue

            # Filter tasks by accuracy
            filtered_df = result_df[result_df["accuracy"] == accuracy]

            # Calculate and print the percentage of tasks with this accuracy
            accuracy_percentage = (len(filtered_df) / length) * 100
            logger.info(
                f"Accuracy = {accuracy:.2f}: {accuracy_percentage:.2f}% of total tasks"
            )

            try:
                # Calculate majority correct rates for this accuracy
                cr_filtered_df = calculate_correct_rate_by_round(
                    filtered_df,
                    model_dir,
                    max_round_number=max_round_number,
                    extract_func=extract_func,
                    compare_func=compare_func,
                )

                # Check if we have results for the majority metric
                majority_rows = cr_filtered_df[cr_filtered_df["metric"] == "majority"]
                if not majority_rows.empty:
                    majority_by_rounds_list.append(majority_rows.iloc[0, 2:].values)

            except Exception as e:
                print(f"Error processing accuracy {accuracy}: {e}")
                continue

        # 5) Aggregate majority values across all accuracies
        if majority_by_rounds_list:
            aggregated_majority_by_round = np.mean(
                np.vstack(majority_by_rounds_list), axis=0
            )
            all_models_data.append(aggregated_majority_by_round)

    # 6) Create a combined plot for all models
    if all_models_data:
        _create_combined_plot(
            all_models_data=all_models_data,
            model_names=model_names,
            output_dir=output_dir,
            max_round_number=max_round_number,
            task_name=task_name,
            plot_title=plot_title,
        )


def _create_combined_plot(
    all_models_data: list[np.ndarray],
    model_names: list[str],
    output_dir: Path,
    max_round_number: int = 10,
    task_name: str = "Judge Bench",
    plot_title: str = None,
) -> None:
    """Creates a combined plot of aggregated majority accuracy for multiple models.

    Args:
        all_models_data: List of arrays containing aggregated majority data for each model.
        model_names: List of model names corresponding to the data.
        output_dir: Directory to save the plot.
        max_round_number: Maximum round number for the plot (default is 10).
        task_name: Name of the task (default is "Judge Bench").
        plot_title: Optional custom plot title.
    """
    plt.figure(figsize=(12, 8))

    # Create x-axis values (round numbers)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_models_data)))

    # Find min and max values for better y-axis scaling
    all_values = np.concatenate(all_models_data)
    min_val = max(0, np.min(all_values) - 0.05)  # Add some padding below
    max_val = min(1, np.max(all_values) + 0.05)  # Add some padding above

    for i, (model_data, model_name) in enumerate(zip(all_models_data, model_names)):
        rounds = np.arange(len(model_data))
        plt.plot(
            rounds,
            model_data,
            color=colors[i],
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=6,
            label=f"{model_name}",
        )

    # Title and labels
    if plot_title:
        plt.title(plot_title, pad=15)
    else:
        plt.title(
            f"Comparison of Aggregated Majority Correct Rates - {task_name}", pad=15
        )
    plt.xlabel("Round Number")
    plt.ylabel("Correct Rate")
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.legend(loc="best")

    # Set y-axis limits and ticks dynamically based on data
    # Ensure we maintain a reasonable minimum range for visibility
    y_range = max_val - min_val
    if y_range < 0.2:  # If range is too small, expand it
        mid = (max_val + min_val) / 2
        min_val = max(0, mid - 0.1)
        max_val = min(1, mid + 0.1)

    plt.ylim(min_val, max_val)
    # Calculate appropriate tick spacing
    tick_spacing = 0.05 if (max_val - min_val) <= 0.3 else 0.1
    plt.yticks(
        np.arange(
            np.floor(min_val * 20) / 20,  # Round down to nearest 0.05
            np.ceil(max_val * 20) / 20 + tick_spacing,  # Round up to nearest 0.05
            tick_spacing,
        )
    )
    plt.xticks(range(min(11, max_round_number + 1)))

    plt.tight_layout()

    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = (
        output_dir / f"{task_name.replace(' ', '_')}_comparison_majority_aggregated.png"
    )
    plt.savefig(plot_path)
    print(f"Combined plot saved to: {plot_path}")

    # Then show it
    plt.show()


# Keep the original function for backward compatibility
def process_model_majority_aggregated(
    model_dir: Path,
    output_dir: Path,
    dataframe: pd.DataFrame,
    task_name: str = "Judge Bench",
    max_round_number: int = 10,
    extract_func: Callable = None,
    compare_func: Callable = None,
) -> None:
    """Process model data and create visualizations for the aggregated majority correct rate.

    This is a wrapper around process_multiple_models_majority_aggregated for backward compatibility.

    Args:
        model_dir: Path to the model directory containing debate data.
        output_dir: Path to the output directory for saving visualizations.
        dataframe: pd.DataFrame,
        task_name: Name of the task (default is "Judge Bench").
        max_round_number: Maximum round number for the correct rate calculation.
        extract_func: Function to extract and normalize responses.
        compare_func: Function to compare normalized responses with correct answer.
    """
    process_multiple_models_majority_aggregated(
        model_dirs=[model_dir],
        output_dir=output_dir,
        dataframe=dataframe,
        task_name=task_name,
        max_round_number=max_round_number,
        extract_func=extract_func,
        compare_func=compare_func,
    )
