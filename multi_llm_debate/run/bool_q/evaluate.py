import glob
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


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

    def normalize_answer(ans: str | bool) -> str:
        if isinstance(ans, bool):
            return "true" if ans else "false"
        ans = str(ans).lower().strip()
        return "true" if ans in ("yes", "true") else "false" if ans in ("no", "false") else ans

    raw_responses = [response["response"] for response in responses]
    normalized_responses = [normalize_answer(r["answer"]) for r in raw_responses]
    normalized_answer = normalize_answer(answer)

    # Check if all responses are the same
    if len(set(normalized_responses)) == 1:
        # Check if the common answer matches the expected answer
        return normalized_responses[0] == normalized_answer
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
        float: Accuracy score (number of correct answers / total questions)
    """
    correct_count = 0
    total_count = len(dataframe)

    for _, entry in dataframe.iterrows():
        answer = entry["answer"]
        id_ = str(entry["id"])

        # Load responses from the corresponding file
        responses_dir = response_base_dir / id_

        # Get the final response file
        final_response_file = _get_latest_round_file(responses_dir)

        with open(final_response_file, "r") as f:
            responses = json.load(f)

        # Evaluate the responses
        is_correct = evaluate_responses(responses, answer)
        if is_correct:
            correct_count += 1

        # Output individual result
        # print(f"ID: {id_}, Correct: {is_correct}")

    # Calculate and output accuracy
    accuracy = correct_count / total_count
    print(f"\nOverall Accuracy: {accuracy:.2%}")

    return accuracy
