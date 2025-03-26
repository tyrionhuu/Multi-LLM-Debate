from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..shared.evaluate import EvaluationResults, evaluate_all
from .utils import extract_bigger_char, extract_caption_a_b_answer


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
        gold_answer = extract_bigger_char(answer)
        # print(f"Normalized answer: {normalized_responses}")
        # print(f"Gold answer: {gold_answer}")
        if len(set(normalized_responses)) == 1:
            return normalized_responses[0] == gold_answer.upper()
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


if __name__ == "__main__":
    from pathlib import Path

    from ..shared.utils import Parser
    from .evaluate import evaluate_all_judge_bench
    from .utils import load_judge_bench_dataset

    args = Parser(description="Run JudgeBench evaluation").parse_args()

    # Load the dataset
    dataset_path = Path("datasets/JudgeBench")
    dataframe = load_judge_bench_dataset(
        dataset_path=dataset_path,
    )

    evaluate_all_judge_bench(
        response_base_dir=Path("data/judge_bench/phi3"),
        dataframe=dataframe,
        extract_func=extract_caption_a_b_answer,
        evaluation_func=evaluate_judge_bench_responses,
        multiple_models=False,
    )