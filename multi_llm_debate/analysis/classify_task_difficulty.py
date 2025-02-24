import json
from pathlib import Path

import pandas as pd

from ..llm.parsers import extract_bool_answer


def classify_task_difficulty(task_dir: Path, dataframe: pd.DataFrame) -> int:
    """
    Classifies the difficulty of a task based on the number of examples in the task directory.

    Args:
        task_dir (Path): The path to the task directory.

    Returns:
        int: The difficulty level of the task, where 0 is easy, 1 is hard.
    """
    # Check if the task directory exists
    if not task_dir.exists():
        raise ValueError(f"Task directory {task_dir} does not exist.")

    task_id = task_dir.name
    # Check if the task ID is in the dataframe
    if task_id not in dataframe["id"].values:
        raise ValueError(f"Task ID {task_id} not found in the dataframe.")

    # Get answer
    answer = dataframe.loc[dataframe["id"] == task_id, "answer"].values[0]

    first_response_file = task_dir / "debate_round_0.json"
    # Check if the first response file exists
    if not first_response_file.exists():
        raise ValueError(f"First response file {first_response_file} does not exist.")

    # Read the first response file
    with open(first_response_file, "r") as f:
        responses = json.load(f)

    # Count the number of examples in the task directory
    correct_count = 0
    total_responses = len(responses)

    # Count correct responses in first round
    for response in responses:
        response_text = response["response"]
        extracted_response = extract_bool_answer(response_text)

        # Skip invalid responses
        if extracted_response is None:
            total_responses -= 1
            continue

        # Convert answer to normalized boolean format
        answer_bool = str(answer).lower().strip() in ["yes", "true", "1"]

        if extracted_response == answer_bool:
            correct_count += 1

    # Calculate accuracy
    accuracy = correct_count / total_responses if total_responses > 0 else 0

    # Classify difficulty - if more than 50% get it right, it's easy
    DIFFICULTY_THRESHOLD = 0.5
    return 0 if accuracy >= DIFFICULTY_THRESHOLD else 1
