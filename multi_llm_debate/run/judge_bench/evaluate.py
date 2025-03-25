from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from .utils import extract_caption_a_b_answer
from ..shared.evaluate import EvaluationResults, evaluate_all


def evaluate_judge_bench_responses(
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
    try:
        raw_responses = [response["response"] for response in responses]
        normalized_responses = [
            extract_caption_a_b_answer(response) for response in raw_responses
        ]

        if len(set(normalized_responses)) == 1:
            return normalized_responses[0] == answer.upper()
        return False
    except Exception as e:
        print(f"Error evaluating responses: {e}")
        return False


def evaluate_all_judge_bench(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    multiple_models: bool = False,
) -> EvaluationResults:
    """Run all judge bench evaluations with judge bench-specific settings.

    A convenience wrapper around evaluate_all that uses judge bench-specific functions.

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
        extract_func=extract_caption_a_b_answer,
        evaluation_func=evaluate_judge_bench_responses,
        multiple_models=multiple_models,
    )
