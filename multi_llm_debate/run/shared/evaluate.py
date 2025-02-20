import json
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Union

import pandas as pd

from ..utils import get_latest_round_file

# Add type alias for the evaluation function
EvaluationFunc = Callable[[List[Dict], Union[str, bool]], bool]

# Add type alias for extract functions
ExtractFunc = Callable[[str], Optional[str]]


def evaluate_debate_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    evaluation_func: Optional[EvaluationFunc] = None,
) -> float:
    """Evaluate the Boolean Question task on a DataFrame.

    Args:
        response_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.
        evaluation_func: Function that takes (responses, answer) and returns bool.
            Must accept List[Dict] as responses and str/bool as answer.

    Returns:
        float: Accuracy score (number of correct answers / total valid responses)
    """
    if evaluation_func is None:
        raise ValueError("evaluation_func must be provided")

    correct_count = 0
    valid_count = 0

    for _, entry in dataframe.iterrows():
        try:
            answer = entry["answer"]
            id_ = str(entry["id"])

            # Load responses from the corresponding file
            responses_dir = response_base_dir / id_

            # Get the final response file
            final_response_file = get_latest_round_file(responses_dir)

            with open(final_response_file, "r") as f:
                responses = json.load(f)

            # Skip if no valid responses
            if not responses:
                continue

            # Evaluate the responses
            is_correct = evaluation_func(responses, answer)
            valid_count += 1
            if is_correct:
                correct_count += 1

        except Exception:
            continue

    # Calculate and output accuracy using valid responses
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Valid responses: {valid_count}/{len(dataframe)}")

    return accuracy


def evaluate_single_llm_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    evaluation_func: Optional[EvaluationFunc] = None,
) -> float:
    """Evaluate the Boolean Question task using first answer as single LLM response.

    Args:
        response_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.
        evaluation_func: Function that takes (responses, answer) and returns bool.
            Must accept List[Dict] as responses and str/bool as answer.

    Returns:
        float: Accuracy score using first answer as single LLM response.
    """
    if evaluation_func is None:
        raise ValueError("evaluation_func must be provided")

    correct_count = 0
    valid_count = 0

    for _, entry in dataframe.iterrows():
        try:
            answer = entry["answer"]
            id_ = str(entry["id"])

            # Load responses from the first debate round file
            responses_dir = response_base_dir / id_
            first_response_file = responses_dir / "debate_round_0.json"

            with open(first_response_file, "r") as f:
                responses = json.load(f)

            # Skip if no valid responses
            if not responses:
                continue

            # Only use the first response
            first_response = responses[0]
            # Create a list with single response for consistent interface
            is_correct = evaluation_func([first_response], answer)
            valid_count += 1
            if is_correct:
                correct_count += 1

        except Exception:
            continue

    # Calculate and output accuracy using valid responses
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    print(f"\nSingle LLM Accuracy: {accuracy:.2%}")
    print(f"Valid single LLM responses: {valid_count}/{len(dataframe)}")

    return accuracy


def get_majority_vote(
    responses: List[Dict],
    extract_func: ExtractFunc,
) -> Optional[str]:
    """Get the majority vote from a list of responses.

    Args:
        responses: List of response dictionaries containing 'response' key.
        extract_func: Function to extract and normalize the response string.

    Returns:
        str or None: The majority response or None if no valid majority.
    """
    # Get all responses and their normalized answers
    raw_responses = [response["response"] for response in responses]
    normalized_responses = [extract_func(response) for response in raw_responses]
    valid_responses = [r for r in normalized_responses if r]

    if not valid_responses:
        return None

    # Count occurrences of each response
    response_counts: Dict[str, int] = {}
    for response in valid_responses:
        response_counts[response] = response_counts.get(response, 0) + 1

    # Get majority vote (most common response)
    majority_response = max(response_counts.items(), key=lambda x: x[1])[0]

    # Check if it's a true majority (more than half)
    total_votes = sum(response_counts.values())
    if response_counts[majority_response] > total_votes / 2:
        return majority_response
    return None


def evaluate_ensemble_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    extract_func: ExtractFunc,
    evaluation_func: EvaluationFunc,
) -> float:
    """Evaluate using majority vote from first round responses.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.
        extract_func: Function to extract and normalize response strings.
        evaluation_func: Function to evaluate if response matches answer.

    Returns:
        float: Accuracy score using majority vote from first round responses.
    """
    correct_count = 0
    valid_count = 0

    for _, entry in dataframe.iterrows():
        try:
            answer = entry["answer"]
            id_ = str(entry["id"])

            # Load responses from the first debate round file
            responses_dir = response_base_dir / id_
            first_response_file = responses_dir / "debate_round_0.json"

            with open(first_response_file, "r") as f:
                responses = json.load(f)

            # Skip if no valid responses
            if not responses:
                continue

            # Get majority vote
            majority_response = get_majority_vote(responses, extract_func)
            if majority_response is None:
                continue

            # Compare with correct answer
            is_correct = evaluation_func([{"response": majority_response}], answer)
            valid_count += 1
            if is_correct:
                correct_count += 1

        except Exception:
            continue

    # Calculate and output accuracy using valid responses
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    print(f"\nEnsemble Accuracy (First Round Majority): {accuracy:.2%}")
    print(f"Valid ensemble responses: {valid_count}/{len(dataframe)}")

    return accuracy


class EvaluationResults(NamedTuple):
    """Container for evaluation results from all methods."""

    debate_accuracy: float
    single_llm_accuracy: float
    ensemble_accuracy: float


def evaluate_all(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    extract_func: ExtractFunc,
    evaluation_func: EvaluationFunc,
) -> EvaluationResults:
    """Run all evaluation methods and return their results.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.
        extract_func: Function to extract and normalize response strings.
        evaluation_func: Function to evaluate if response matches answer.

    Returns:
        EvaluationResults: Named tuple containing accuracies for all three methods.
    """
    print("\nRunning debate evaluation...")
    debate_acc = evaluate_debate_df(
        response_base_dir, dataframe, evaluation_func=evaluation_func
    )

    print("\nRunning single LLM evaluation...")
    single_acc = evaluate_single_llm_df(
        response_base_dir, dataframe, evaluation_func=evaluation_func
    )

    print("\nRunning ensemble evaluation...")
    ensemble_acc = evaluate_ensemble_df(
        response_base_dir,
        dataframe,
        extract_func=extract_func,
        evaluation_func=evaluation_func,
    )

    print("\nSummary of all evaluation methods:")
    print(f"Debate accuracy:     {debate_acc:.2%}")
    print(f"Single LLM accuracy: {single_acc:.2%}")
    print(f"Ensemble accuracy:   {ensemble_acc:.2%}")

    return EvaluationResults(debate_acc, single_acc, ensemble_acc)
