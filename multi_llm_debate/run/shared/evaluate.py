import concurrent.futures
import json
import logging
import math
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import pandas as pd

from .utils import get_latest_round_file

logger = logging.getLogger(__name__)


class EvaluationResults(NamedTuple):
    """Container for evaluation results from all methods."""

    debate_accuracy: float
    single_llm_accuracy: float
    ensemble_accuracy: float
    # Confidence intervals (lower, upper) for each accuracy
    debate_ci: Tuple[float, float]
    single_llm_ci: Tuple[float, float]
    ensemble_ci: Tuple[float, float]


def wilson_score_interval(
    correct: int, total: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion.

    Args:
        correct: Number of successful trials
        total: Total number of trials
        confidence: Confidence level (default: 0.95 for 95% confidence)

    Returns:
        Tuple[float, float]: Lower and upper bounds of confidence interval
    """
    if total == 0:
        return (0.0, 0.0)

    # z-score for the given confidence level
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)

    p = float(correct) / total
    denominator = 1.0 + z * z / total
    centre_adjusted_probability = p + z * z / (2 * total)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)

    lower = (
        centre_adjusted_probability - z * adjusted_standard_deviation
    ) / denominator
    upper = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator

    return (max(0.0, lower), min(1.0, upper))


def _process_debate_entry(
    entry: pd.Series, response_base_dir: Path, evaluation_func: Callable
) -> Optional[bool]:
    """Process a single entry for debate evaluation.

    Args:
        entry: DataFrame row containing question data
        response_base_dir: Directory containing response files
        evaluation_func: Function to evaluate correctness

    Returns:
        Optional[bool]: True if correct, False if incorrect, None if entry skipped
    """
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
            return None

        # Evaluate the responses
        return evaluation_func(responses, answer)

    except Exception as e:
        logger.error(f"Error processing entry {entry.get('id', 'unknown')}: {str(e)}")
        return None


def evaluate_debate_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    evaluation_func: Optional[Callable] = None,
    num_workers: int = 4,
    use_processes: bool = True,
) -> Tuple[float, Tuple[float, float]]:
    """Evaluate the Boolean Question task on a DataFrame.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.
        evaluation_func: Function that takes (responses, answer) and returns bool.
            Must accept List[Dict] as responses and str/bool as answer.
        num_workers: Number of parallel workers to use.
        use_processes: If True, use ProcessPoolExecutor, otherwise ThreadPoolExecutor.

    Returns:
        Tuple[float, Tuple[float, float]]: Accuracy score and confidence interval tuple
    """
    if evaluation_func is None:
        raise ValueError("evaluation_func must be provided")

    logger.info(
        f"Starting debate evaluation on {len(dataframe)} entries with {num_workers} workers..."
    )

    executor_class = (
        concurrent.futures.ProcessPoolExecutor
        if use_processes
        else concurrent.futures.ThreadPoolExecutor
    )

    results = []
    with executor_class(max_workers=num_workers) as executor:
        futures = []
        for _, entry in dataframe.iterrows():
            future = executor.submit(
                _process_debate_entry, entry, response_base_dir, evaluation_func
            )
            futures.append(future)

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if i % 10 == 0:
                logger.info(f"Completed {i}/{len(futures)} tasks")
            results.append(future.result())

    # Filter out None results and count correct ones
    valid_results = [result for result in results if result is not None]
    correct_count = sum(1 for result in valid_results if result)
    valid_count = len(valid_results)

    # Calculate accuracy and confidence interval
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    ci = wilson_score_interval(correct_count, valid_count)

    logger.info(f"Overall Accuracy: {accuracy:.2%} (95% CI: {ci[0]:.2%}-{ci[1]:.2%})")
    logger.info(f"Valid responses: {valid_count}/{len(dataframe)}")

    return accuracy, ci


def evaluate_single_llm_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    evaluation_func: Optional[Callable] = None,
    num_workers: int = 4,
    use_processes: bool = True,
) -> Tuple[float, Tuple[float, float]]:
    """Evaluate the task using all first round answers, averaging correct rate per entry.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.
        evaluation_func: Function that takes (responses, answer) and returns bool.
            Must accept List[Dict] as responses and str/bool as answer.
        num_workers: Number of parallel workers to use.
        use_processes: If True, use ProcessPoolExecutor, otherwise ThreadPoolExecutor.

    Returns:
        Tuple[float, Tuple[float, float]]: Average correct rate and confidence interval
    """
    if evaluation_func is None:
        raise ValueError("evaluation_func must be provided")

    logger.info(
        f"Starting single LLM evaluation (average correct rate) on {len(dataframe)} entries with {num_workers} workers..."
    )

    def _entry_correct_fraction(
        entry: pd.Series, response_base_dir: Path, evaluation_func: Callable
    ) -> Optional[float]:
        """Compute fraction of correct responses for a single entry."""
        try:
            answer = entry["answer"]
            id_ = str(entry["id"])
            responses_dir = response_base_dir / id_
            first_response_file = responses_dir / "debate_round_0.json"
            with open(first_response_file, "r") as f:
                responses = json.load(f)
            if not responses:
                return None
            correct = 0
            total = 0
            for resp in responses:
                # Evaluate each response individually
                if evaluation_func([resp], answer):
                    correct += 1
                total += 1
            if total == 0:
                return None
            return correct / total
        except Exception as e:
            logger.error(
                f"Error processing entry {entry.get('id', 'unknown')}: {str(e)}"
            )
            return None

    executor_class = (
        concurrent.futures.ProcessPoolExecutor
        if use_processes
        else concurrent.futures.ThreadPoolExecutor
    )

    results = []
    with executor_class(max_workers=num_workers) as executor:
        futures = []
        for _, entry in dataframe.iterrows():
            future = executor.submit(
                _entry_correct_fraction, entry, response_base_dir, evaluation_func
            )
            futures.append(future)

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if i % 10 == 0:
                logger.info(f"Completed {i}/{len(futures)} tasks")
            results.append(future.result())

    # Filter out None results and compute mean and standard error
    valid_results = [result for result in results if result is not None]
    valid_count = len(valid_results)

    if valid_count == 0:
        return 0.0, (0.0, 0.0)

    mean_correct_rate = sum(valid_results) / valid_count

    # For averaging rates, we can use normal approximation
    # Standard error of the mean
    if valid_count > 1:
        std_dev = math.sqrt(
            sum((x - mean_correct_rate) ** 2 for x in valid_results) / (valid_count - 1)
        )
        std_error = std_dev / math.sqrt(valid_count)
        margin = 1.96 * std_error  # 95% confidence
        ci = (
            max(0.0, mean_correct_rate - margin),
            min(1.0, mean_correct_rate + margin),
        )
    else:
        # Can't compute confidence interval with just one sample
        ci = (0.0, 1.0)

    logger.info(
        f"Single LLM Average Correct Rate: {mean_correct_rate:.2%} (95% CI: {ci[0]:.2%}-{ci[1]:.2%})"
    )
    logger.info(f"Valid single LLM entries: {valid_count}/{len(dataframe)}")

    return mean_correct_rate, ci


def _process_ensemble_entry(
    entry: pd.Series,
    response_base_dir: Path,
    extract_func: Callable,
    evaluation_func: Callable,
    answer_entry: str = "answer",
    id_entry: str = "id",
    response_entry: str = "response",
) -> Optional[bool]:
    """Process a single entry for ensemble evaluation.

    Args:
        entry: DataFrame row containing question data
        response_base_dir: Directory containing response files
        extract_func: Function to extract and normalize responses
        evaluation_func: Function to evaluate correctness
        answer_entry: Column name for the correct answer
        id_entry: Column name for the unique identifier
        response_entry: Column name for the response

    Returns:
        Optional[bool]: True if correct, False if incorrect, None if entry skipped
    """
    try:
        id_ = str(entry[id_entry])

        # Load responses from the first debate round file
        responses_dir = response_base_dir / id_
        first_response_file = responses_dir / "debate_round_0.json"

        with open(first_response_file, "r") as f:
            responses = json.load(f)

        # Skip if no valid responses
        if not responses:
            return None

        # Get all responses and their normalized answers
        raw_responses = [response[response_entry] for response in responses]

        # Create a list of (normalized_response, raw_response, original_normalized) tuples
        response_pairs = []
        for raw in raw_responses:
            normalized = extract_func(raw)
            if normalized is not None:  # Only include valid normalized responses
                # Handle lists by converting to string representation for hashing
                if isinstance(normalized, List):
                    # Store original value and use string representation as key
                    response_pairs.append((str(normalized), raw, normalized))
                else:
                    response_pairs.append((normalized, raw, normalized))

        if not response_pairs:
            return None

        # Count occurrences of each normalized response (using string representation if needed)
        response_counts: Dict[str, int] = {}
        for key, _, _ in response_pairs:
            response_counts[key] = response_counts.get(key, 0) + 1

        # Get majority normalized response key
        majority_key = max(response_counts.items(), key=lambda x: x[1])[0]

        # Check if it's a true majority (more than half)
        total_votes = sum(response_counts.values())
        if response_counts[majority_key] <= total_votes / 2:
            return None

        # Find the original raw response that corresponds to the majority normalized response
        majority_raw = next(
            raw for key, raw, _ in response_pairs if key == majority_key
        )

        # Compare with correct answer using the raw response
        return evaluation_func([{response_entry: majority_raw}], entry[answer_entry])

    except Exception as e:
        logger.error(
            f"Error processing entry {entry.get(id_entry, 'unknown')}: {str(e)}"
        )
        return None


def evaluate_ensemble_df(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    extract_func: Callable,
    evaluation_func: Callable,
    answer_entry: str = "answer",
    id_entry: str = "id",
    response_entry: str = "response",
    num_workers: int = 4,
    use_processes: bool = True,
) -> Tuple[float, Tuple[float, float]]:
    """Evaluate using majority vote from first round responses.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.
        extract_func: Function to extract and normalize response strings.
        evaluation_func: Function to evaluate if response matches answer.
        answer_entry: Column name for the correct answer in the DataFrame.
        id_entry: Column name for the unique identifier in the DataFrame.
        response_entry: Column name for the response in the DataFrame.
        num_workers: Number of parallel workers to use.
        use_processes: If True, use ProcessPoolExecutor, otherwise ThreadPoolExecutor.

    Returns:
        Tuple[float, Tuple[float, float]]: Accuracy score and confidence interval tuple
    """
    logger.info(
        f"Starting ensemble evaluation on {len(dataframe)} entries with {num_workers} workers..."
    )

    executor_class = (
        concurrent.futures.ProcessPoolExecutor
        if use_processes
        else concurrent.futures.ThreadPoolExecutor
    )

    results = []
    with executor_class(max_workers=num_workers) as executor:
        futures = []
        for _, entry in dataframe.iterrows():
            future = executor.submit(
                _process_ensemble_entry,
                entry,
                response_base_dir,
                extract_func,
                evaluation_func,
                answer_entry,
                id_entry,
                response_entry,
            )
            futures.append(future)

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            if i % 10 == 0:
                logger.info(f"Completed {i}/{len(futures)} tasks")
            results.append(future.result())

    # Filter out None results and count correct ones
    valid_results = [result for result in results if result is not None]
    correct_count = sum(1 for result in valid_results if result)
    valid_count = len(valid_results)

    # Calculate accuracy and confidence interval
    accuracy = correct_count / valid_count if valid_count > 0 else 0
    ci = wilson_score_interval(correct_count, valid_count)

    logger.info(
        f"Ensemble Accuracy (First Round Majority): {accuracy:.2%} (95% CI: {ci[0]:.2%}-{ci[1]:.2%})"
    )
    logger.info(f"Valid ensemble responses: {valid_count}/{len(dataframe)}")

    return accuracy, ci


def evaluate_all(
    response_base_dir: Path,
    dataframe: pd.DataFrame,
    extract_func: Callable,
    evaluation_func: Callable,
    multiple_models: bool = False,
    answer_entry: str = "answer",
    id_entry: str = "id",
    response_entry: str = "response",
    num_workers: int = 4,
    use_processes: bool = True,
) -> EvaluationResults:
    """Run all evaluation methods and return their results.

    Args:
        response_base_dir: Directory containing response files.
        dataframe: Pandas DataFrame containing question, answer, passage and id.
        extract_func: Function to extract and normalize response strings.
        evaluation_func: Function to evaluate if response matches answer.
        multiple_models: Whether multiple model types are being evaluated.
        answer_entry: Column name for the correct answer in the DataFrame.
        id_entry: Column name for the unique identifier in the DataFrame.
        response_entry: Column name for the response in the DataFrame.
        num_workers: Number of parallel workers to use.
        use_processes: If True, use ProcessPoolExecutor, otherwise ThreadPoolExecutor.

    Returns:
        EvaluationResults: Named tuple containing accuracies and confidence intervals
    """
    logger.info("Running debate evaluation...")
    logger.info(f"Processing data directory: {response_base_dir}")
    logger.info(f"Dataset contains {len(dataframe)} entries")
    logger.info(
        f"Using {num_workers} workers with {'processes' if use_processes else 'threads'}"
    )

    debate_acc, debate_ci = evaluate_debate_df(
        response_base_dir,
        dataframe,
        evaluation_func=evaluation_func,
        num_workers=num_workers,
        use_processes=use_processes,
    )

    # Only run single LLM evaluation for single model type
    single_acc = 0.0
    single_ci = (0.0, 0.0)
    if not multiple_models:
        logger.info("Running single LLM evaluation...")
        single_acc, single_ci = evaluate_single_llm_df(
            response_base_dir,
            dataframe,
            evaluation_func=evaluation_func,
            num_workers=num_workers,
            use_processes=use_processes,
        )

    logger.info("Running ensemble evaluation...")
    ensemble_acc, ensemble_ci = evaluate_ensemble_df(
        response_base_dir,
        dataframe,
        extract_func=extract_func,
        evaluation_func=evaluation_func,
        answer_entry=answer_entry,
        id_entry=id_entry,
        response_entry=response_entry,
        num_workers=num_workers,
        use_processes=use_processes,
    )

    logger.info("Summary of all evaluation methods:")
    logger.info(
        f"Debate accuracy:     {debate_acc:.2%} (95% CI: {debate_ci[0]:.2%}-{debate_ci[1]:.2%})"
    )
    if not multiple_models:
        logger.info(
            f"Single LLM accuracy: {single_acc:.2%} (95% CI: {single_ci[0]:.2%}-{single_ci[1]:.2%})"
        )
    logger.info(
        f"Ensemble accuracy:   {ensemble_acc:.2%} (95% CI: {ensemble_ci[0]:.2%}-{ensemble_ci[1]:.2%})"
    )

    return EvaluationResults(
        debate_acc, single_acc, ensemble_acc, debate_ci, single_ci, ensemble_ci
    )
