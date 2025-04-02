import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional

import pandas as pd

from .utils import get_latest_round_file

# Set up logger with proper configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EvaluationResults(NamedTuple):
    """Container for evaluation results from all methods."""

    debate_accuracy: float
    single_llm_accuracy: float
    ensemble_accuracy: float


def evaluate_debate_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    evaluation_func: Optional[Callable] = None,
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

    logger.info(f"Starting debate evaluation on {len(dataframe)} entries...")

    for i, (_, entry) in enumerate(dataframe.iterrows()):
        if i % 10 == 0:
            logger.info(f"Processing entry {i}/{len(dataframe)}")

        try:
            answer = entry["answer"]
            id_ = str(entry["id"])

            logger.debug(f"Evaluating ID: {id_}, expected answer: {answer}")

            # Load responses from the corresponding file
            responses_dir = response_base_dir / id_
            logger.debug(f"Looking for responses in: {responses_dir}")

            # Get the final response file
            final_response_file = get_latest_round_file(responses_dir)
            logger.debug(f"Using final response file: {final_response_file}")

            with open(final_response_file, "r") as f:
                responses = json.load(f)
                logger.debug(f"Loaded {len(responses)} responses")

            # Skip if no valid responses
            if not responses:
                logger.warning("No valid responses found, skipping")
                continue

            # Evaluate the responses
            is_correct = evaluation_func(responses, answer)
            valid_count += 1
            if is_correct:
                correct_count += 1
                logger.debug(
                    f"Correct! Current accuracy: {correct_count}/{valid_count}"
                )
            else:
                logger.debug(
                    f"Incorrect. Current accuracy: {correct_count}/{valid_count}"
                )

        except Exception as e:
            logger.error(f"Error processing entry {id_}: {str(e)}")
            continue

    # Calculate and output accuracy using valid responses
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    logger.info(f"Overall Accuracy: {accuracy:.2%}")
    logger.info(f"Valid responses: {valid_count}/{len(dataframe)}")

    return accuracy


def evaluate_single_llm_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    evaluation_func: Optional[Callable] = None,
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

    logger.info(f"Starting single LLM evaluation on {len(dataframe)} entries...")

    for i, (_, entry) in enumerate(dataframe.iterrows()):
        if i % 10 == 0:
            logger.info(f"Processing entry {i}/{len(dataframe)}")

        try:
            answer = entry["answer"]
            id_ = str(entry["id"])

            logger.debug(f"Evaluating ID: {id_}, expected answer: {answer}")

            # Load responses from the first debate round file
            responses_dir = response_base_dir / id_
            first_response_file = responses_dir / "debate_round_0.json"
            logger.debug(f"Looking for first round response: {first_response_file}")

            with open(first_response_file, "r") as f:
                responses = json.load(f)
                logger.debug(f"Loaded {len(responses)} responses from first round")

            # Skip if no valid responses
            if not responses:
                logger.warning("No valid responses found, skipping")
                continue

            # Only use the first response
            first_response = responses[0]
            logger.debug("Using first response for evaluation")

            # Create a list with single response for consistent interface
            is_correct = evaluation_func([first_response], answer)
            valid_count += 1
            if is_correct:
                correct_count += 1
                logger.debug(
                    f"Correct! Current accuracy: {correct_count}/{valid_count}"
                )
            else:
                logger.debug(
                    f"Incorrect. Current accuracy: {correct_count}/{valid_count}"
                )

        except Exception as e:
            logger.error(f"Error processing entry {id_}: {str(e)}")
            continue

    # Calculate and output accuracy using valid responses
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    logger.info(f"Single LLM Accuracy: {accuracy:.2%}")
    logger.info(f"Valid single LLM responses: {valid_count}/{len(dataframe)}")

    return accuracy


def get_majority_vote(
    responses: List[Dict],
    extract_func: Callable,
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
    extract_func: Callable,
    evaluation_func: Callable,
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

    logger.info(f"Starting ensemble evaluation on {len(dataframe)} entries...")

    for i, (_, entry) in enumerate(dataframe.iterrows()):
        if i % 10 == 0:
            logger.info(f"Processing entry {i}/{len(dataframe)}")

        try:
            answer = entry["answer"]
            id_ = str(entry["id"])

            logger.debug(f"Evaluating ID: {id_}, expected answer: {answer}")

            # Load responses from the first debate round file
            responses_dir = response_base_dir / id_
            first_response_file = responses_dir / "debate_round_0.json"
            logger.debug(f"Looking for first round responses: {first_response_file}")

            with open(first_response_file, "r") as f:
                responses = json.load(f)
                logger.debug(f"Loaded {len(responses)} responses from first round")

            # Skip if no valid responses
            if not responses:
                logger.warning("No valid responses found, skipping")
                continue

            # Get majority vote
            majority_response = get_majority_vote(responses, extract_func)
            if majority_response is None:
                logger.warning("No majority response found, skipping")
                continue

            logger.debug(f"Majority response: {majority_response}")

            # Compare with correct answer
            is_correct = evaluation_func([{"response": majority_response}], answer)
            valid_count += 1
            if is_correct:
                correct_count += 1
                logger.debug(
                    f"Correct! Current accuracy: {correct_count}/{valid_count}"
                )
            else:
                logger.debug(
                    f"Incorrect. Current accuracy: {correct_count}/{valid_count}"
                )

        except Exception as e:
            logger.error(f"Error processing entry {id_}: {str(e)}")
            continue

    # Calculate and output accuracy using valid responses
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    logger.info(f"Ensemble Accuracy (First Round Majority): {accuracy:.2%}")
    logger.info(f"Valid ensemble responses: {valid_count}/{len(dataframe)}")

    return accuracy


def evaluate_all(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    extract_func: Callable,
    evaluation_func: Callable,
    multiple_models: bool = False,
) -> EvaluationResults:
    """Run all evaluation methods and return their results.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.
        extract_func: Function to extract and normalize response strings.
        evaluation_func: Function to evaluate if response matches answer.
        multiple_models: Whether multiple model types are being evaluated.

    Returns:
        EvaluationResults: Named tuple containing accuracies for all three methods.
    """
    logger.info("Running debate evaluation...")
    logger.info(f"Processing data directory: {response_base_dir}")
    logger.info(f"Dataset contains {len(dataframe)} entries")

    debate_acc = evaluate_debate_df(
        response_base_dir, dataframe, evaluation_func=evaluation_func
    )

    # Only run single LLM evaluation for single model type
    single_acc = 0.0
    if not multiple_models:
        logger.info("Running single LLM evaluation...")
        single_acc = evaluate_single_llm_df(
            response_base_dir, dataframe, evaluation_func=evaluation_func
        )

    logger.info("Running ensemble evaluation...")
    ensemble_acc = evaluate_ensemble_df(
        response_base_dir,
        dataframe,
        extract_func=extract_func,
        evaluation_func=evaluation_func,
    )

    logger.info("Summary of all evaluation methods:")
    logger.info(f"Debate accuracy:     {debate_acc:.2%}")
    if not multiple_models:
        logger.info(f"Single LLM accuracy: {single_acc:.2%}")
    logger.info(f"Ensemble accuracy:   {ensemble_acc:.2%}")

    return EvaluationResults(debate_acc, single_acc, ensemble_acc)
