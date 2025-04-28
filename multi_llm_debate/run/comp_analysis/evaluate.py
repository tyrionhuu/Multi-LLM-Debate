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
        compare_comp_analysis_response(extract_1_to_5_answer(response["response"]), answer)
        for response in responses
    )