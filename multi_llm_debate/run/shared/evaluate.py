import pandas as pd
from pathlib import Path
from ..utils import get_latest_round_file
import json
from typing import Callable
def evaluate_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    evaluation_func: Callable = None,
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
            final_response_file = get_latest_round_file(responses_dir)

            with open(final_response_file, "r") as f:
                responses = json.load(f)

            # Skip if no valid responses
            if not responses:
                continue

            # Evaluate the responses
            is_correct = evaluation_func(responses, answer)
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