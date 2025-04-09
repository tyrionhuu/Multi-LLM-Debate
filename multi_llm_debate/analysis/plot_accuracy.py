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

# Maximum round number for your correct_rate_by_round function
MAX_ROUND_NUMBER = 10


def create_plot_majority_aggregated(
    aggregated_majority_by_round: np.ndarray,
    model_name: str,
    output_dir: Path,
    task_name: str = "Judge Bench",
) -> None:
    """Plots the aggregated majority accuracy value by round.

    Args:
        aggregated_majority_by_round: Array of majority correct rates aggregated
            across all accuracies for each round.
        model_name: Name of the model for the plot title.
        output_dir: Directory to save the plot.
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
        label=f"Aggregated Majority Correct Rate",
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
    plt.xticks(range(min(11, MAX_ROUND_NUMBER + 1)))

    plt.tight_layout()

    # Save the plot first
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{model_name}_majority_aggregated.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

    # Then show it (if needed)
    plt.show()


def process_model_majority_aggregated(
    model_dir: Path,
    output_dir: Path,
    dataframe: pd.DataFrame,
    task_name: str = "Judge Bench",
    extract_func: Callable = None,
    compare_func: Callable = None,
) -> None:
    """Process model data and create visualizations for the aggregated majority correct rate.

    Args:
        model_dir: Path to the model directory containing debate data.
        output_dir: Path to the output directory for saving visualizations.
        dataframe: pd.DataFrame,
        task_name: Name of the task (default is "Judge Bench").
        extract_func: Function to extract and normalize responses.
        compare_func: Function to compare normalized responses with correct answer.

    This function:
    1) Analyzes accuracy
    2) Calculates majority correct rates for each unique accuracy value
    3) Aggregates the results for the majority correct rate across all accuracy levels
    4) Plots the results for the aggregated majority correct rate
    """
    model_name = model_dir.name
    # print(f"\nProcessing model: {model_name}")

    # 1) Analyze accuracy
    result_df = analyze_task_accuracy(
        model_dir=model_dir,
        dataframe=dataframe,
        extract_fn=extract_func,
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
        print(f"Accuracy = {accuracy:.2f}: {accuracy_percentage:.2f}% of total tasks")

        try:
            # Calculate majority correct rates for this accuracy
            cr_filtered_df = calculate_correct_rate_by_round(
                filtered_df,
                model_dir,
                max_round_number=MAX_ROUND_NUMBER,
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
        # Stack the lists vertically and calculate the mean across the rows (across all accuracies)
        aggregated_majority_by_round = np.mean(
            np.vstack(majority_by_rounds_list), axis=0
        )

        # 6) Create the plot for aggregated majority correct rates
        create_plot_majority_aggregated(
            aggregated_majority_by_round, model_name, output_dir, task_name
        )
