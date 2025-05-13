from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..shared.evaluate import EvaluationResults, evaluate_all
from .utils import compare_truthful_qa_response, extract_caption_a_b_c_answer


def evaluate_truthful_qa_responses(
    responses: List[Dict],
    answer: str,
) -> bool:
    """Evaluate the responses from the debate.

    Args:
        responses: List of agent responses from the most recent round of debate.
        answer: The correct answer to the question ("A"/"B"/"C").

    Returns:
        bool: True if all responses are the same and match the answer, False otherwise.
    """
    return (
        all(
            compare_truthful_qa_response(
                extract_caption_a_b_c_answer(response["response"]), answer
            )
            for response in responses
        )
        and len(responses) > 0
    )


def evaluate_all_truthful_qa(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
) -> EvaluationResults:
    """Run all truthful QA evaluations with truthful QA-specific settings.

    A convenience wrapper around evaluate_all that uses truthful QA-specific functions.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing judge bench data.
        max_rounds: Maximum number of debate rounds.

    Returns:
        EvaluationResults: Results of the evaluation.
    """
    return evaluate_all(
        response_base_dir=response_base_dir,
        dataframe=dataframe,
        extract_func=extract_caption_a_b_c_answer,
        evaluation_func=evaluate_truthful_qa_responses,
        max_rounds=max_rounds,
    )


if __name__ == "__main__":
    from pathlib import Path

    from .utils import load_truthful_qa_dataset

    df = load_truthful_qa_dataset()

    result = evaluate_all_truthful_qa(
        response_base_dir=Path("data/truthful_qa/gemini-2_0-flash-001(7)"),
        dataframe=df,
        max_rounds=5,
    )
    print(result)
