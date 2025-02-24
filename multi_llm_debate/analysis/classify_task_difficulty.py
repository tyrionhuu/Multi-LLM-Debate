import json
from pathlib import Path

import pandas as pd

from ..llm.parsers import extract_bool_answer


def analyze_task_difficulty(model_dir: Path, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the task difficulty for tasks that exist in the model directory.

    Args:
        model_dir (Path): The path to the model directory.
        dataframe (pd.DataFrame): The DataFrame containing task information.

    Returns:
        pd.DataFrame: A DataFrame with an additional column 'difficulty' indicating
        the difficulty level of each task.
    """
    # Initialize difficulty dictionary
    difficulty_dict = {}

    # Convert id column to string type if it isn't already
    dataframe["id"] = dataframe["id"].astype(str)

    # Iterate through existing task directories
    for task_dir in model_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_id = str(task_dir.name)  # Ensure task_id is string
        if task_id not in dataframe["id"].values:
            print(f"Warning: Task ID {task_id} not found in dataframe")
            continue

        answer = dataframe.loc[dataframe["id"] == task_id, "answer"].values[0]
        difficulty = classify_task_difficulty(task_dir, answer)
        difficulty_dict[task_id] = difficulty

    # Add difficulty column to dataframe
    dataframe["difficulty"] = dataframe["id"].map(
        lambda x: difficulty_dict.get(str(x), -1)
    )

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

            # Convert both to lowercase strings for comparison
            if str(extracted_response).lower() == str(answer_bool).lower():
                correct_count += 1

        # Calculate accuracy
        accuracy = correct_count / total_responses if total_responses > 0 else 0
        # print(f"Task ID: {task_dir.name}, Correct: {correct_count}, Total: {total_responses}, Accuracy: {accuracy:.2f}")
        # Classify difficulty - if more than 50% get it right, it's easy
        DIFFICULTY_THRESHOLD = 0.7
        return 0 if accuracy >= DIFFICULTY_THRESHOLD else 1

    except Exception as e:
        print(f"Error processing task directory {task_dir}: {e}")
        # print(traceback.format_exc())
        return -1


if __name__ == "__main__":
    # Set up paths
    model_dir = Path("data/bool_q/gemma2:2b(3)")
    data_path = Path("output/bool_q/processed_data.csv")

    # Load dataset
    dataframe = pd.read_csv(data_path)

    # Analyze task difficulty
    result_df = analyze_task_difficulty(model_dir, dataframe)

    # Print summary statistics
    print("\nDifficulty Distribution:")
    print(result_df["difficulty"].value_counts().sort_index())

    # Print error cases (difficulty = -1)
    error_cases = result_df[result_df["difficulty"] == -1]
    if not error_cases.empty:
        print("\nError cases:")
        print(error_cases[["id", "question"]])
