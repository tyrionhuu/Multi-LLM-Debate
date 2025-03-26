from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..shared.evaluate import EvaluationResults, evaluate_all
from .utils import extract_bigger_char, extract_caption_a_b_answer



    

def compare_judge_bench_responses(
    responses: str,
    answer: str,
) -> bool:
    """Compare the responses from the judge bench.

    Args:
        responses: The response string from the LLM.
        answer: The correct answer to the question ("A"/"B").

    Returns:
        bool: True if the response matches the answer, False otherwise.
    """
    try:
        normalized_response = extract_caption_a_b_answer(responses)
        gold_answer = extract_bigger_char(answer)
        return normalized_response == gold_answer.upper()
    except Exception as e:
        print(f"Error comparing responses: {e}")
        return False
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
        compare_judge_bench_responses(response["response"], answer)
        for response in responses
    )
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
    print(dataframe.columns)
    evaluate_all_judge_bench(
        response_base_dir=Path("data/JudgeBench/llama3(11)"),
        dataframe=dataframe,
        multiple_models=False,
    )
