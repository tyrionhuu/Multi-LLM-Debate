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
        compare_comp_analysis_response(
            extract_1_to_5_answer(response["response"]), answer
        )
        for response in responses
    )


def evaluate_all_comp_analysis(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
) -> EvaluationResults:
    """Run all COMP Analysis evaluations with COMP Analysis-specific settings.

    A convenience wrapper around evaluate_all that uses COMP Analysis-specific functions.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing judge bench data.

    Returns:
        EvaluationResults: Results of the evaluation.
    """
    return evaluate_all(
        response_base_dir=response_base_dir,
        dataframe=dataframe,
        extract_func=extract_1_to_5_answer,
        evaluation_func=evaluate_comp_analysis_responses,
    )

if __name__ == "__main__":
    from pathlib import Path
    from multi_llm_debate.utils.logging_config import setup_logging
    import logging
    from .utils import load_comp_analysis_dataset
    logger = setup_logging(__name__, log_level=logging.INFO)
        
    df = load_comp_analysis_dataset(sample_size=1000)

    result = evaluate_all_comp_analysis(
        response_base_dir=Path("data/comp_analysis/gemma-3-4b-it(11)"),
        dataframe=df,
    )
    print(result)
