import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .utils import get_final_round, normalize_boolean_answer


def analyze_task_accuracy(
    model_dir: Path,
    dataframe: pd.DataFrame,
    extract_fn: Callable,
    compare_func: Callable[[str, str], bool] = lambda r, c: r == c,
) -> pd.DataFrame:
    """
    Analyzes the task accuracy for tasks that exist in the model directory.

    Args:
        model_dir (Path): The path to the model directory.
        dataframe (pd.DataFrame): The DataFrame containing task information.
        extract_fn (Callable[[str], Optional[bool]], optional): Function to extract
            boolean answers from response text. Defaults to extract_bool_answer.
        compare_func (Callable[[str, str], bool]): Function to compare normalized
            responses with correct answer, should take (response, correct_answer)
            and return a boolean.

    Returns:
        pd.DataFrame: A DataFrame with an additional column 'accuracy' indicating
        the accuracy for each task.
    """
    # Initialize accuracy dictionary
    accuracy_dict = {}

    # Check if 'id' column exists in the dataframe
    if "id" not in dataframe.columns:
        raise ValueError("DataFrame must contain an 'id' column")

    # Convert id column to string type if it isn't already
    dataframe["id"] = dataframe["id"].astype(str)

    # Iterate through existing task directories
    for task_dir in model_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_id = str(task_dir.name)
        if task_id not in dataframe["id"].values:
            print(f"Warning: Task ID {task_id} not found in dataframe")
            continue

        answer = dataframe.loc[dataframe["id"] == task_id, "answer"].values[0]
        accuracy = calculate_task_accuracy(
            task_dir, answer, extract_fn=extract_fn, compare_func=compare_func
        )
        accuracy_dict[task_id] = accuracy

    # Add accuracy column to dataframe
    dataframe["accuracy"] = dataframe["id"].map(
        lambda x: accuracy_dict.get(str(x), -1.0)
    )

    return dataframe


def calculate_task_accuracy(
    task_dir: Path,
    answer: str,
    extract_fn: Callable,
    compare_func: Callable[[str, str], bool] = lambda r, c: r == c,
    normalize_func: Callable[[str], str] = lambda x: x,
    round_number: int = 0,
) -> float:
    """
    Calculates the accuracy for a task based on the responses in the task directory.

    Args:
        task_dir (Path): The path to the task directory.
        answer (str): The correct answer for the task ('yes'/'no' or 'true'/'false').
        round_number (int, optional): The debate round number to analyze. Defaults to 0.
            If this round is larger than the final round, the final round's data will be used.
        extract_fn (Callable[[str], Optional[bool]], optional): Function to extract
            boolean answers from response text. Defaults to extract_bool_answer.
        compare_func (Callable[[str, str], bool]): Function to compare normalized
            responses with correct answer, should take (response, correct_answer)
            and return a boolean.

    Returns:
        float: The accuracy of the task, or -1.0 if an error occurred.
    """
    try:
        if not task_dir.exists():
            return -1.0

        final_round = get_final_round(task_dir)
        if final_round == -1:
            return -1.0

        actual_round = min(round_number, final_round)

        response_file = task_dir / f"debate_round_{actual_round}.json"
        # Check if the response file exists (should be true based on logic above)
        if not response_file.exists():
            return -1.0

        # Read the response file
        with open(response_file, "r") as f:
            responses = json.load(f)

        # Count correct responses
        correct_count = 0
        total_responses = len(responses)

        # Normalize the answer
        normalized_answer = normalize_func(answer)
        if normalized_answer is None:
            print(f"Warning: Ambiguous answer format '{answer}' for task {task_dir}")
            return -1.0

        # Count correct responses in the specified round
        for response in responses:
            response_text = response["response"]
            extracted_response = extract_fn(response_text)

            # Skip invalid responses
            if extracted_response is None:
                total_responses -= 1
                continue

            # Compare using the compare_func instead of direct equality
            if compare_func(extracted_response, normalized_answer):
                correct_count += 1

        # Calculate and return accuracy
        return correct_count / total_responses if total_responses > 0 else 0.0

    except Exception as e:
        print(
            f"Error processing task directory {task_dir} for round {round_number}: {e}"
        )
        return -1.0
