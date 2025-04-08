from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..shared.evaluate import EvaluationResults, evaluate_all
from .utils import compare_llm_bar_response, extract_1_2_answer


def evaluate_llm_bar_responses(
    responses: List[Dict],
    answer: str,
) -> bool:
    """Evaluate the responses from the debate.

    Args:
        responses: List of agent responses from the most recent round of debate.
        answer: The correct answer to the question ("1" or "2").

    Returns:
        bool: True if all responses are the same and match the answer, False otherwise.
    """
    return (
        all(
            compare_llm_bar_response(extract_1_2_answer(response["response"]), answer)
            for response in responses
        )
        and len(responses) > 0
    )


def evaluate_all_llm_bar(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    multiple_models: bool = False,
) -> EvaluationResults:
    """Run all LLM Bar evaluations with LLM Bar-specific settings.

    A convenience wrapper around evaluate_all that uses LLM Bar-specific functions.

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
        extract_func=extract_1_2_answer,
        evaluation_func=evaluate_llm_bar_responses,
        multiple_models=multiple_models,
    )
