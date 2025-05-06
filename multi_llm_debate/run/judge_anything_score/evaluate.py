from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..shared.evaluate import EvaluationResults, evaluate_all
from .utils import compare_judge_anything_score_response, extract_1_5_answer


def evaluate_judge_anything_score_responses(
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
        compare_judge_anything_score_response(
            extract_1_5_answer(response["response"]), answer
        )
        for response in responses
    )


def evaluate_all_judge_anything_score(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
) -> EvaluationResults:
    """Run all MLLM judge score evaluations with judge pair-specific settings.

    A convenience wrapper around evaluate_all that uses judge pair-specific functions.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing judge pair data.

    Returns:
        EvaluationResults: Results of the evaluation.
    """
    return evaluate_all(
        response_base_dir=response_base_dir,
        dataframe=dataframe,
        extract_func=extract_1_5_answer,
        evaluation_func=evaluate_judge_anything_score_responses,
    )
