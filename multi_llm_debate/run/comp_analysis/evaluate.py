from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..shared.evaluate import EvaluationResults, evaluate_all
from .utils import compare_comp_analysis_response, extract_1_to_5_answer


def evaluate_comp_analysis_responses(
    responses: List[Dict],
    answer: str,
) -> bool:
    """Evaluate the responses from the debate.

    Args:
        responses: List of agent responses from the most recent round of debate.
        answer: The correct answer to the question (an integer between 1 and 5).

    Returns:
        bool: True if all responses are the same and match the answer, False otherwise.
    """
    return all(
        compare_comp_analysis_response(
            extract_1_to_5_answer(response["response"]), answer
        )
        for response in responses
    )


def evaluate_all_comp_analysis(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    multiple_models: bool = False,
) -> EvaluationResults:
    """Run all COMP Analysis evaluations with COMP Analysis-specific settings.

    A convenience wrapper around evaluate_all that uses COMP Analysis-specific functions.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing judge bench data.
        multiple_models: Whether multiple model types are being evaluated.

    Returns:
        EvaluationResults: Results of the evaluation.
    """
    return evaluate_all(
        response_base_dir=response_base_dir,
        dataframe=dataframe,
        extract_func=extract_1_to_5_answer,
        evaluation_func=evaluate_comp_analysis_responses,
        multiple_models=multiple_models,
    )
