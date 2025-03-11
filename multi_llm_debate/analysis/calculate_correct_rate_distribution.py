from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_per_round_accuracy(
    dataframe: pd.DataFrame, model_dir: Path, max_round_number: int
) -> pd.DataFrame:
    subdirs = [subdir for subdir in model_dir.iterdir() if subdir.is_dir()]
    model_configuration = model_dir.name
    pbar = tqdm(subdirs, desc=f"Processing {model_configuration}")

    # Create bins for distribution (0-0.1, 0.1-0.2, ..., 0.9-1.0)
    bin_edges = np.linspace(0, 1, 11)
    bin_names = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(10)]

    result_data = []

    for task_dir in pbar:
        task_id = task_dir.name

        # Find the final round for this task
        rounds_present = []
        for round_num in range(1, max_round_number + 1):
            round_dir = task_dir / f"round_{round_num}"
            if round_dir.exists():
                rounds_present.append(round_num)

        final_round = max(rounds_present) if rounds_present else 0

        # Process each round up to max_round_number
        for round_num in range(1, max_round_number + 1):
            round_dir = task_dir / f"round_{round_num}"

            # If this round doesn't exist, use the final round's results
            if not round_dir.exists():
                if final_round > 0:
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
