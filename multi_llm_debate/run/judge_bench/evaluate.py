from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..shared.evaluate import EvaluationResults, evaluate_all
from .utils import compare_judge_bench_response, extract_caption_a_b_answer


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
    return all(
        compare_judge_bench_response(
            extract_caption_a_b_answer(response["response"]), answer
        )
        for response in responses
    )


def evaluate_all_judge_bench(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
) -> EvaluationResults:
    """Run all judge bench evaluations with judge bench-specific settings.

    A convenience wrapper around evaluate_all that uses judge bench-specific functions.

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
        extract_func=extract_caption_a_b_answer,
        evaluation_func=evaluate_judge_bench_responses,
        max_rounds=max_rounds,
    )


if __name__ == "__main__":
    from pathlib import Path

    from .utils import load_judge_bench_dataset

    # Load the dataset
    dataframe = load_judge_bench_dataset()
    # print(dataframe.columns)
    result = evaluate_all_judge_bench(
        response_base_dir=Path("data/judge_bench/gemini-2_0-flash-001(7)"),
        dataframe=dataframe,
        max_rounds=6,
    )
    print(result)
