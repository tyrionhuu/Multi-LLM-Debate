from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..shared.evaluate import EvaluationResults, evaluate_all
from .utils import compare_prm800k_response, extract_int_list


def evaluate_prm800k_responses(
    responses: List[Dict],
    answer: List[int],
) -> bool:
    """Evaluate the responses from the debate.

    Args:
        responses: List of agent responses from the most recent round of debate.
        answer: The correct answer to the question as a list of integers.

    Returns:
        bool: True if all responses match the answer, False otherwise.
    """
    return (
        all(
            compare_prm800k_response(extract_int_list(response["response"]), answer)
            for response in responses
        )
        and len(responses) > 0
    )


def evaluate_all_prm800k(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
) -> EvaluationResults:
    """Run all PRM800K evaluations with PRM800K-specific settings.

    A convenience wrapper around evaluate_all that uses PRM800K-specific functions.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing judge bench data.

    Returns:
        EvaluationResults: Results of the evaluation.
    """
    return evaluate_all(
        response_base_dir=response_base_dir,
        dataframe=dataframe,
        extract_func=extract_int_list,
        evaluation_func=evaluate_prm800k_responses,
    )
