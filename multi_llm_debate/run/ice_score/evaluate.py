from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..shared.evaluate import EvaluationResults, evaluate_all
from .utils import compare_ice_score_response, extract_0_4_answer


def evaluate_ice_score_responses(
    responses: List[Dict],
    answer: str,
) -> bool:
    """Evaluate the responses from the debate.

    Args:
        responses: List of agent responses from the most recent round of debate.
        answer: The correct answer to the question ("A"/"B").

    Returns:
        bool: True if all responses are the same and match the answer, False otherwise.
    """
    return all(
        compare_ice_score_response(extract_0_4_answer(response["response"]), answer)
        for response in responses
    )


def evaluate_all_ice_score(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
) -> EvaluationResults:
    """Run all ice score evaluations with ice score-specific settings.

    A convenience wrapper around evaluate_all that uses ice score-specific functions.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing judge bench data.

    Returns:
        EvaluationResults: Results of the evaluation.
    """
    return evaluate_all(
        response_base_dir=response_base_dir,
        dataframe=dataframe,
        extract_func=extract_0_4_answer,
        evaluation_func=evaluate_ice_score_responses,
    )
