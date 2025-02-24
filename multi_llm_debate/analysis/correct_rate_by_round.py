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
        question_id = subdir.name

        if question_id not in dataframe.index:
            continue

        correct_answer = str(dataframe.loc[question_id, "answer"]).lower()

        try:
            latest_round_file = get_latest_round_file(subdir)
            last_round = int(latest_round_file.stem.split("_")[-1])
        except (ValueError, FileNotFoundError):
            continue

        for round_num in range(1, min(max_round_number + 1, last_round + 1)):
            round_file = subdir / f"debate_round_{round_num}.json"
            if not round_file.exists():
                continue

            try:
                with open(round_file, "r") as f:
                    round_data = json.load(f)

                # Get all agent responses from this round
                responses = round_data.get("responses", [])
                if not responses:
                    continue

                # Extract and normalize all boolean answers
                normalized_responses = [
                    extract_bool_answer(response.get("response", ""))
                    for response in responses
                ]

                # Filter out invalid/empty responses
                valid_responses = [r for r in normalized_responses if r]
                if not valid_responses:
                    continue

                # Only count as correct if all valid responses are the same and match answer
                total_counts[round_num] += 1
                if (
                    len(set(valid_responses)) == 1
                    and valid_responses[0] == correct_answer
                ):
                    correct_counts[round_num] += 1

            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    # Calculate correct rates for each round
    for round_num in range(1, max_round_number + 1):
        if total_counts[round_num] > 0:
            correct_rate = correct_counts[round_num] / total_counts[round_num]
        else:
            correct_rate = 0.0
        row_data[str(round_num)] = correct_rate

    return pd.DataFrame([row_data])


if __name__ == "__main__":
    # Example usage
    model_dir = Path("data/bool_q/gemma2:2b(3)")
    dataframe = pd.read_csv("output/bool_q/processed_data.csv", index_col=0)
    max_round_number = 10

    result_df = calculate_correct_rate_by_round(
        dataframe, model_dir, max_round_number
    )
    print(result_df)