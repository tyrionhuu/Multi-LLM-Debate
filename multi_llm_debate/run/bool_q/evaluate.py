from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from ...llm.parsers import extract_bool_answer
from ..shared.evaluate import EvaluationResults, evaluate_all


def evaluate_bool_q_responses(
    responses: List[Dict],
    answer: Union[str, bool],
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


def evaluate_all_bool_q(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    multiple_models: bool = False,
) -> EvaluationResults:
    """Run all boolean evaluations with bool-specific settings.

    A convenience wrapper around evaluate_all that uses bool-specific functions.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing boolean questions data.
        multiple_models: Whether multiple model types are being evaluated.

    Returns:
        EvaluationResults: Named tuple containing accuracies for all three methods.
    """
    return evaluate_all(
        response_base_dir=response_base_dir,
        dataframe=dataframe,
        extract_func=extract_bool_answer,
        evaluation_func=evaluate_bool_q_responses,
        multiple_models=multiple_models,
    )


def main() -> EvaluationResults:
    """Run all evaluations and return the results.

    Returns:
        EvaluationResults: Named tuple containing accuracies for all three methods.
    """
    from ...utils.download_dataset import load_save_dataset_df
    from .utils import process_bool_q_df

    dataset_path = Path("datasets/boolq")
    response_base_dir = Path("data/bool_q/phi3")

    # Load and process the dataset
    dataframe = load_save_dataset_df(
        dataset_name="google/boolq",
        dataset_path=dataset_path,
        force_download=False,
    )
    processed_dataframe = process_bool_q_df(dataframe)

    # Run all bool evaluations and return results
    return evaluate_all_bool_q(response_base_dir, processed_dataframe)


if __name__ == "__main__":
    results = main()
    # Can access results as:
    # results.debate_accuracy
    # results.single_llm_accuracy
    # results.ensemble_accuracy
