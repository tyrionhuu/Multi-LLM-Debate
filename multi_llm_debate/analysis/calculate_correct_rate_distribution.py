import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..llm.parsers import extract_bool_answer
from .utils import get_final_round


def calculate_per_round_accuracy(
    dataframe: pd.DataFrame, model_dir: Path, max_round_number: int
) -> pd.DataFrame:
    """Calculate accuracy distribution for each round of debate across tasks.

    This function processes a directory of debate tasks and calculates a distribution
    of correct rates for each round. It uses responses from debate rounds to determine
    the accuracy for each task. If a debate converges before the maximum
    round number, the final round's result is used for subsequent rounds.

    Args:
        dataframe: DataFrame containing ground truth data with columns 'task_id'
            and 'ground_truth'.
        model_dir: Path to the directory containing subdirectories for each task.
        max_round_number: Maximum number of debate rounds to analyze.

    Returns:
        DataFrame with columns for each correct rate bin (0.0-0.1, 0.1-0.2, etc.),
        'task_id', and 'round_number'. Each bin column contains the count (0 or 1)
        indicating whether the task's correctness for that round falls into that bin.
    """
    subdirs = [subdir for subdir in model_dir.iterdir() if subdir.is_dir()]
    model_configuration = model_dir.name
    pbar = tqdm(subdirs, desc=f"Processing {model_configuration}")

    # Create bins for distribution (0-0.1, 0.1-0.2, ..., 0.9-1.0)
    bin_edges = np.linspace(0, 1, 11)
    bin_names = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(10)]

    result_data = []

    for task_dir in pbar:
        task_id = task_dir.name
        final_round = get_final_round(task_dir)

        if final_round == -1:
            continue

        # Filter dataframe for this task
        task_df = dataframe[dataframe["task_id"] == task_id]
        if task_df.empty:
            continue

        ground_truth = task_df["ground_truth"].iloc[0]

        # Convert ground truth to normalized boolean format
        processed_answer = str(ground_truth).lower().strip()
        if processed_answer in ["yes", "true", "1"]:
            answer_bool = True
        elif processed_answer in ["no", "false", "0"]:
            answer_bool = False
        else:
            continue

        # Process each round up to max_round_number
        for round_num in range(1, max_round_number + 1):
            actual_round = min(round_num, final_round)
            response_file = task_dir / f"debate_round_{actual_round}.json"

            if not response_file.exists():
                continue

            try:
                # Read the response file
                with open(response_file, "r") as f:
                    responses = json.load(f)

                # Count correct responses
                correct_count = 0
                total_responses = len(responses)

                # Count correct responses in the round
                for response in responses:
                    response_text = response["response"]
                    extracted_response = extract_bool_answer(response_text)

                    # Skip invalid responses
                    if extracted_response is None:
                        total_responses -= 1
                        continue

                    # Compare with ground truth
                    if str(extracted_response).lower() == str(answer_bool).lower():
                        correct_count += 1

                # Calculate correct rate
                if total_responses > 0:
                    correct_rate = correct_count / total_responses

                    # Find which bin this correct rate falls into
                    bin_idx = min(
                        int(correct_rate * 10), 9
                    )  # Ensure index is within range

                    # Create a row with zeros
                    row = {bin_name: 0 for bin_name in bin_names}
                    row[bin_names[bin_idx]] = 1  # Set the correct bin to 1
                    row["task_id"] = task_id
                    row["round_number"] = round_num

                    result_data.append(row)
            except Exception as e:
                print(f"Error processing task {task_id} for round {round_num}: {e}")
                continue

    # Create the result dataframe
    if result_data:
        result_df = pd.DataFrame(result_data)
    else:
        # If no data was collected, create an empty DataFrame with the correct columns
        columns = bin_names + ["task_id", "round_number"]
        result_df = pd.DataFrame(columns=columns)

    return result_df


if __name__ == "__main__":
    # Set up paths
    model_dir = Path("data/bool_q/llama3(7)")
    data_path = Path("output/bool_q/processed_data.csv")
    max_round_number = 10

    # Load dataset - adjust column names to match your actual data structure
    dataframe = pd.read_csv(data_path)

    # Rename columns if necessary for compatibility with the function
    if "id" in dataframe.columns and "answer" in dataframe.columns:
        dataframe = dataframe.rename(
            columns={"id": "task_id", "answer": "ground_truth"}
        )

    # Calculate accuracy distribution by round
    result_df = calculate_per_round_accuracy(dataframe, model_dir, max_round_number)

    # Print the results
    print("Accuracy distribution by round:")
    print(result_df.head())

    # Calculate and print summary statistics by round
    round_stats = {}
    for round_num in range(1, max_round_number + 1):
        round_data = result_df[result_df["round_number"] == round_num]
        if not round_data.empty:
            # Calculate average correct rate for this round
            bin_midpoints = [
                float(bin_name.split("-")[0]) + 0.05
                for bin_name in round_data.columns
                if "-" in bin_name
            ]
            bin_counts = [
                round_data[col].sum() for col in round_data.columns if "-" in col
            ]

            weighted_sum = sum(
                midpoint * count for midpoint, count in zip(bin_midpoints, bin_counts)
            )
            total_count = sum(bin_counts)

            avg_correct_rate = weighted_sum / total_count if total_count > 0 else 0
            round_stats[round_num] = {
                "avg_correct_rate": avg_correct_rate,
                "task_count": len(round_data),
            }

    # Print summary statistics
    print("\nSummary statistics by round:")
    for round_num, stats in round_stats.items():
        print(
            f"Round {round_num}: "
            f"Average Correct Rate: {stats['avg_correct_rate']:.3f}, "
            f"Tasks: {stats['task_count']}"
        )
