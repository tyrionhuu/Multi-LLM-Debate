from pathlib import Path
from .utils import get_final_round

import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_per_round_accuracy(
    dataframe: pd.DataFrame, model_dir: Path, max_round_number: int
) -> pd.DataFrame:
    """Calculate accuracy distribution for each round of debate across tasks.

    This function processes a directory of debate tasks and calculates a distribution
    of correct rates for each round. It uses majority voting to determine the
    consensus answer for each round. If a debate converges before the maximum
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

        # Process each round up to max_round_number
        for round_num in range(1, max_round_number + 1):
            round_dir = task_dir / f"round_{round_num}"

            # If this round doesn't exist, use the final round's results
            if not round_dir.exists():
                if final_round >= 0:
                    round_dir = task_dir / f"round_{final_round}"
                else:
                    continue

            # Get answers for this round
            answers = []
            for agent_dir in round_dir.glob("agent_*"):
                answer_file = agent_dir / "answer.txt"
                if answer_file.exists():
                    with open(answer_file, "r") as f:
                        answer = f.read().strip()
                        answers.append(answer)

            if not answers:
                continue

            # Get the majority vote answer
            if answers:
                answer_counts = {}
                for answer in answers:
                    if answer in answer_counts:
                        answer_counts[answer] += 1
                    else:
                        answer_counts[answer] = 1

                majority_answer = max(answer_counts, key=answer_counts.get)

                # Filter dataframe for this task
                task_df = dataframe[dataframe["task_id"] == task_id]

                if not task_df.empty:
                    # Calculate correct rate (1 if majority answer matches ground truth, 0 otherwise)
                    correct = int(majority_answer == task_df["ground_truth"].iloc[0])

                    # Find which bin this correct rate falls into
                    bin_idx = min(int(correct * 10), 9)  # Ensure index is within range

                    # Create a row with zeros
                    row = {bin_name: 0 for bin_name in bin_names}
                    row[bin_names[bin_idx]] = 1  # Set the correct bin to 1
                    row["task_id"] = task_id
                    row["round_number"] = round_num

                    result_data.append(row)

    # Create the result dataframe
    if result_data:
        result_df = pd.DataFrame(result_data)
    else:
        # If no data was collected, create an empty DataFrame with the correct columns
        columns = bin_names + ["task_id", "round_number"]
        result_df = pd.DataFrame(columns=columns)

    return result_df
