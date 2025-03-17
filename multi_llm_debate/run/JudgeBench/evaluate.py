from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from ...llm.parsers import extract_caption_a_b_answer
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