import json
from pathlib import Path

import pandas as pd

from ..llm.parsers import extract_bool_answer


def analyze_task_accuracy(model_dir: Path, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes the task accuracy for tasks that exist in the model directory.

    Args:
        model_dir (Path): The path to the model directory.
        dataframe (pd.DataFrame): The DataFrame containing task information.

    Returns:
        pd.DataFrame: A DataFrame with an additional column 'accuracy' indicating
        the accuracy for each task.
    """
    # Initialize accuracy dictionary
    accuracy_dict = {}

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
        accuracy = calculate_task_accuracy(task_dir, answer)
        accuracy_dict[task_id] = accuracy

    # Add accuracy column to dataframe
    dataframe["accuracy"] = dataframe["id"].map(
        lambda x: accuracy_dict.get(str(x), -1.0)
    )

    return dataframe


def calculate_task_accuracy(task_dir: Path, answer: str) -> float:
    """
    Calculates the accuracy for a task based on the responses in the task directory.

    Args:
        task_dir (Path): The path to the task directory.
        answer (str): The correct answer for the task ('yes'/'no' or 'true'/'false').

    Returns:
        float: The accuracy of the task, or -1.0 if an error occurred.
    """
    try:
        # Check if the task directory exists
        if not task_dir.exists():
            return -1.0

        first_response_file = task_dir / "debate_round_0.json"
        # Check if the first response file exists
        if not first_response_file.exists():
            return -1.0

        # Read the first response file
        with open(first_response_file, "r") as f:
            responses = json.load(f)

        # Count correct responses
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

        # Calculate and return accuracy
        return correct_count / total_responses if total_responses > 0 else 0.0

    except Exception as e:
        print(f"Error processing task directory {task_dir}: {e}")
        return -1.0


if __name__ == "__main__":
    # Set up paths
    model_dir = Path("data/bool_q/llama3(7)")
    data_path = Path("output/bool_q/processed_data.csv")

    # Load dataset
    dataframe = pd.read_csv(data_path)

    # Analyze task accuracy
    result_df = analyze_task_accuracy(model_dir, dataframe)
    print(result_df)
    
    # Print summary statistics
    print("\nAccuracy Statistics:")
    print(result_df["accuracy"].describe())

    # Print error cases (accuracy = -1)
    error_cases = result_df[result_df["accuracy"] == -1.0]
    if not error_cases.empty:
        print("\nError cases:")

        print(error_cases[["id", "question"]])
