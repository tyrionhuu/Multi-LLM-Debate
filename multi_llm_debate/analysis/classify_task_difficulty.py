import json
from pathlib import Path

import pandas as pd

from ..llm.parsers import extract_bool_answer


def analyze_task_difficulty(model_dir: Path, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the task difficulty for each task in the given DataFrame.

    Args:
        model_dir (Path): The path to the model directory.
        dataframe (pd.DataFrame): The DataFrame containing task information.

    Returns:
        pd.DataFrame: A DataFrame with an additional column 'difficulty' indicating the difficulty level of each task.
    """
    # Initialize a list to store difficulty levels
    difficulty_levels = []

    # Iterate through each row in the DataFrame
    for _, row in dataframe.iterrows():
        task_dir = model_dir / row["task"]
        task_id = task_dir.name
        answer = dataframe.loc[dataframe["task"] == task_id, "answer"].values[0]
        difficulty = classify_task_difficulty(task_dir, answer)
        difficulty_levels.append(difficulty)

    # Add the difficulty levels to the DataFrame
    dataframe["difficulty"] = difficulty_levels

    return dataframe


def classify_task_difficulty(task_dir: Path, answer: str) -> int:
    """
    Classifies the difficulty of a task based on the number of examples in the task directory.

    Args:
        task_dir (Path): The path to the task directory.
        answer (str): The correct answer for the task ('yes'/'no' or 'true'/'false').

    Returns:
        int: The difficulty level of the task, where:
                0 is easy
                1 is hard
                -1 indicates an error occurred
    """
    try:
        # Check if the task directory exists
        if not task_dir.exists():
            return -1

        first_response_file = task_dir / "debate_round_0.json"
        # Check if the first response file exists
        if not first_response_file.exists():
            return -1

        # Read the first response file
        with open(first_response_file, "r") as f:
            responses = json.load(f)

        # Count the number of examples in the task directory
        correct_count = 0
        total_responses = len(responses)

        # Convert answer to normalized boolean format
        answer_bool = str(answer).lower().strip() in ["yes", "true", "1"]

        # Count correct responses in first round
        for response in responses:
            response_text = response["response"]
            extracted_response = extract_bool_answer(response_text)

            # Skip invalid responses
            if extracted_response is None:
                total_responses -= 1
                continue

            if extracted_response == answer_bool:
                correct_count += 1

        # Calculate accuracy
        accuracy = correct_count / total_responses if total_responses > 0 else 0

        # Classify difficulty - if more than 50% get it right, it's easy
        DIFFICULTY_THRESHOLD = 0.5
        return 0 if accuracy >= DIFFICULTY_THRESHOLD else 1

    except Exception:
        return -1
