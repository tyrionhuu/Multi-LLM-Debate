import json
from pathlib import Path

import pandas as pd

from ..llm.parsers import extract_bool_answer
from ..run.shared.utils import get_latest_round_file


def calculate_correct_rate_by_round(
    dataframe: pd.DataFrame, model_dir: Path, max_round_number: int
) -> pd.DataFrame:
    """Calculate the correct rate for each round in the dataframe.

    Args:
        dataframe: DataFrame containing the correct answers indexed by ID
        model_dir: Path to the model directory containing debate results
        max_round_number: Maximum number of debate rounds to analyze

    Returns:
        DataFrame with correct rates for each round
    """
    model_configuration = model_dir.name
    row_data = {"model_configuration": model_configuration}

    # Get all subdirectories
    subdirs = [d for d in model_dir.iterdir() if d.is_dir()]

    # Initialize counters for each round
    correct_counts = {i: 0 for i in range(1, max_round_number + 1)}
    total_counts = {i: 0 for i in range(1, max_round_number + 1)}

    for subdir in subdirs:
        # Get the ID from the directory name
        question_id = subdir.name

        # Skip if ID not in dataframe
        if question_id not in dataframe.index:
            continue

        correct_answer = dataframe.loc[question_id, "correct_answer"]

        # Find the last round for this debate
        try:
            latest_round_file = get_latest_round_file(subdir)
            last_round = int(latest_round_file.stem.split("_")[-1])
        except ValueError:
            continue

        # Process each round up to either max_round_number or last_round
        for round_num in range(1, min(max_round_number + 1, last_round + 1)):
            round_file = subdir / f"debate_round_{round_num}.json"
            if not round_file.exists():
                continue

            try:
                with open(round_file, "r") as f:
                    round_data = json.load(f)

                # Extract the predicted answer for this round
                predicted_answer = extract_bool_answer(round_data["conclusion"])
                if predicted_answer is not None:
                    total_counts[round_num] += 1
                    if predicted_answer == correct_answer:
                        correct_counts[round_num] += 1
            except (json.JSONDecodeError, KeyError):
                continue

    # Calculate correct rates for each round
    for round_num in range(1, max_round_number + 1):
        if total_counts[round_num] > 0:
            correct_rate = correct_counts[round_num] / total_counts[round_num]
        else:
            correct_rate = 0.0
        row_data[str(round_num)] = correct_rate

    return pd.DataFrame([row_data])
