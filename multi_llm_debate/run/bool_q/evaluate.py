import glob
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ...llm.parsers import extract_bool_answer


def evaluate_responses(
    responses: List[Dict],
    answer: str | bool,
) -> bool:
    """Evaluate the responses from the debate.

    Args:
        responses: List of agent responses from the most recent round of debate.
        answer: The correct answer to the question ("yes"/"no", "true"/"false", or bool).

    Returns:
        bool: True if all responses are the same and match the answer, False otherwise.
    """
    try:
        raw_responses = [response["response"] for response in responses]
        normalized_responses = [
            extract_bool_answer(response) for response in raw_responses
        ]
        answer_string = str(answer).lower()

        # print(f"Normalized answer: {normalized_responses}")
        # Filter out empty responses
        valid_responses = [r for r in normalized_responses if r]
        if not valid_responses:
            # print("Warning: No valid responses found")
            return False

        # Check if all valid responses are the same
        if len(set(valid_responses)) == 1:
            return valid_responses[0] == answer_string
        return False
    except Exception as e:
        print(f"Error evaluating responses: {e}")
        return False


def _get_latest_round_file(responses_dir: Path) -> Path:
    """Get the file path for the latest debate round.

    Args:
        responses_dir: Directory containing debate round files

    Returns:
        Path to the latest debate round file
    """
    pattern = str(responses_dir / "debate_round_*.json")
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No debate round files found in {responses_dir}")

    # Extract round numbers and find max
    rounds = [int(re.search(r"debate_round_(\d+)", f).group(1)) for f in files]
    latest_round = max(rounds)
    return Path(responses_dir / f"debate_round_{latest_round}.json")


def evaluate_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
) -> float:
    """Evaluate the Boolean Question task on a DataFrame.

    Args:
        response_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.

    Returns:
        float: Accuracy score (number of correct answers / total valid responses)
    """
    correct_count = 0
    valid_count = 0

    for _, entry in dataframe.iterrows():
        try:
            answer = entry["answer"]
            id_ = str(entry["id"])

            # Load responses from the corresponding file
            responses_dir = response_base_dir / id_

            # Get the final response file
            final_response_file = _get_latest_round_file(responses_dir)

            with open(final_response_file, "r") as f:
                responses = json.load(f)

            # Skip if no valid responses
            if not responses:
                continue

            # Evaluate the responses
            is_correct = evaluate_responses(responses, answer)
            valid_count += 1
            if is_correct:
                correct_count += 1

        except Exception:
            continue

    # Calculate and output accuracy using valid responses
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Valid responses: {valid_count}/{len(dataframe)}")

    return accuracy


def evaluate_single_llm_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
) -> float:
    """Evaluate the Boolean Question task using first answer as single LLM response.

    Args:
        response_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.

    Returns:
        float: Accuracy score using first answer as single LLM response.
    """
    correct_count = 0
    valid_count = 0

    for _, entry in dataframe.iterrows():
        try:
            answer = entry["answer"]
            id_ = str(entry["id"])

            # Load responses from the first debate round file
            responses_dir = response_base_dir / id_
            first_response_file = responses_dir / "debate_round_0.json"

            with open(first_response_file, "r") as f:
                responses = json.load(f)

            # Skip if no valid responses
            if not responses:
                continue

            # Only use the first response
            first_response = responses[0]
            is_correct = evaluate_responses([first_response], answer)
            valid_count += 1
            if is_correct:
                correct_count += 1

        except Exception:
            continue

    # Calculate and output accuracy using valid responses
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    print(f"\nSingle LLM Accuracy: {accuracy:.2%}")
    print(f"Valid single LLM responses: {valid_count}/{len(dataframe)}")

    return accuracy


def main() -> None:
    from ...utils.download_dataset import load_save_dataset_df
    from .utils import process_bool_q_df

    dataset_path = Path("datasets/boolq")

    # Example usage
    response_base_dir = Path("data/bool_q/phi3")
    dataframe = load_save_dataset_df(
        dataset_name="google/boolq",
        dataset_path=dataset_path,
        force_download=False,
    )

    # Process the DataFrame
    processed_dataframe = process_bool_q_df(dataframe)
    # Evaluate the debate responses
    evaluate_df(response_base_dir, processed_dataframe)

    # Evaluate the single LLM responses
    evaluate_single_llm_df(response_base_dir, processed_dataframe)


if __name__ == "__main__":
    main()
