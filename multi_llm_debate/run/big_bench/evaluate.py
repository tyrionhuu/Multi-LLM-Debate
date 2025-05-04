from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..shared.evaluate import EvaluationResults, evaluate_all
from .utils import compare_big_bench_response, extract_0_1_answer


def evaluate_big_bench_responses(
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
        compare_big_bench_response(extract_0_1_answer(response["response"]), answer)
        for response in responses
    )


def evaluate_all_big_bench(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
) -> EvaluationResults:
    """Run all BIG-Bench evaluations with BIG-Bench-specific settings.

    A convenience wrapper around evaluate_all that uses BIG-Bench-specific functions.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing judge bench data.

    Returns:
        EvaluationResults: Results of the evaluation.
    """
    return evaluate_all(
        response_base_dir=response_base_dir,
        dataframe=dataframe,
        extract_func=extract_0_1_answer,
        evaluation_func=evaluate_big_bench_responses,
    )
if __name__ == "__main__":
    from pathlib import Path
    
    from .utils import load_big_bench_dataset

    df = load_big_bench_dataset(sample_size=1000)

    result = evaluate_all_big_bench(
        response_base_dir=Path("data/big_bench/Qwen2_5-7B-Instruct(11)"),
        dataframe=df,
    )
    print(result)