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
    multiple_models: bool = False,
) -> EvaluationResults:
    """Run all truthful QA evaluations with truthful QA-specific settings.

    A convenience wrapper around evaluate_all that uses truthful QA-specific functions.

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
        extract_func=extract_caption_a_b_c_answer,
        evaluation_func=evaluate_truthful_qa_responses,
        multiple_models=multiple_models,
    )


if __name__ == "__main__":
    from pathlib import Path

    from ..shared.utils import Parser
    from .utils import load_truthful_qa_dataset, preprocess_truthful_qa_dataframe

    args = Parser(description="Run TruthfulQA evaluation").parse_args()

    # Load the dataset
    dataset_path = Path("datasets/TruthfulQA")
    dataframe = load_truthful_qa_dataset(
        dataset_path=dataset_path,
    )
    dataframe = preprocess_truthful_qa_dataframe(dataframe)
    evaluate_all_truthful_qa(
        response_base_dir=Path("data/truthful_qa/Llama-3_1-8B-Instruct(11)"),
        dataframe=dataframe,
        multiple_models=False,
    )
