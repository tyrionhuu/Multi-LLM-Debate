import json
import logging
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_debate_round(
    round_file: Path,
    correct_answer: str,
    extract_func: Callable,
    compare_func: Callable[[str, str], bool] = lambda r, c: r == c,
) -> Tuple[Optional[Tuple[bool, float]], bool]:
    """Process a single debate round file and calculate both majority and absolute rates.

    Args:
        round_file: Path to the debate round JSON file
        correct_answer: The expected correct answer
        extract_func: Function to extract and normalize responses, defaults to
            extract_bool_answer
        compare_func: Function to compare normalized responses with the correct answer,
            should take (response, correct_answer) and return a boolean

    Returns:
        Tuple containing:
        - Tuple of (majority_correct, absolute_rate) or None if invalid
        - Boolean indicating if debate should end
    """
    try:
        with open(round_file, "r") as f:
            responses = json.load(f)

        try:
            normalized_responses = [
                extract_func(response.get("response", ""))
                for response in responses
                if response.get("response")
            ]
        except ValueError as e:
            logger.debug(f"Error processing responses: {str(e)}")
            return None, True

        if not normalized_responses:
            logger.debug("No valid responses found, skipping round")
            return None, True

        # Calculate both metrics
        absolute_rate = sum(
            1 for r in normalized_responses if compare_func(r, correct_answer)
        ) / len(normalized_responses)

        # Calculate majority vote
        majority_correct = (
            sum(1 for r in normalized_responses if compare_func(r, correct_answer))
            > len(normalized_responses) / 2
        )

        return (majority_correct, absolute_rate), False

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug(f"Error processing round: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None, True


def process_debate_directory(
    subdir: Path,
    dataframe: pd.DataFrame,
    max_round_number: int,
    extract_func: Callable,
    compare_func: Callable[[str, str], bool] = lambda r, c: r == c,
) -> Tuple[Dict[int, Dict[str, int]], Dict[int, int]]:
    """Process a single debate directory and calculate correctness counts.

    Args:
        subdir: Path to the debate directory
        dataframe: DataFrame containing correct answers
        max_round_number: Maximum number of rounds to process
        extract_func: Function to extract and normalize responses, defaults to
            extract_bool_answer
        compare_func: Function to compare normalized responses with correct answer,
            should take (response, correct_answer) and return a boolean

    Returns:
        Tuple of (correct_counts, total_counts) where correct_counts contains
        both 'majority' and 'absolute' metrics
    """
    question_id = subdir.name
    logger.debug(f"Processing question ID: {question_id}")

    str_id = str(question_id)
    try:
        int_id = int(question_id)
    except ValueError:
        int_id = None

    matching_rows = dataframe[
        (dataframe["id"] == str_id)
        | (dataframe["id"] == int_id if int_id is not None else False)
    ]
    if matching_rows.empty:
        logger.debug(f"Skipping {question_id} - not found in dataframe")
        return {}, {}

    correct_answer = str(matching_rows.iloc[0]["answer"]).lower()
    correct_counts = {
        i: {"majority": 0, "absolute": 0.0} for i in range(max_round_number + 1)
    }
    total_counts = {i: 0 for i in range(0, max_round_number + 1)}

    last_majority = None
    last_absolute = 0.0
    debate_ended = False

    for round_num in range(0, max_round_number + 1):
        if debate_ended:
            total_counts[round_num] += 1
            if last_majority is not None:
                correct_counts[round_num]["majority"] += int(last_majority)
                correct_counts[round_num]["absolute"] += last_absolute
            continue

        round_file = subdir / f"debate_round_{round_num}.json"
        if not round_file.exists():
            debate_ended = True
            if last_majority is not None:
                total_counts[round_num] += 1
                correct_counts[round_num]["majority"] += int(last_majority)
                correct_counts[round_num]["absolute"] += last_absolute
            continue

        round_result, should_end = process_debate_round(
            round_file, correct_answer, extract_func, compare_func
        )
        if should_end:
            debate_ended = True
            continue

        if round_result is not None:
            majority_correct, absolute_rate = round_result
            total_counts[round_num] += 1
            correct_counts[round_num]["majority"] += int(majority_correct)
            correct_counts[round_num]["absolute"] += absolute_rate
            last_majority = majority_correct
            last_absolute = absolute_rate

    return correct_counts, total_counts


def count_absolute_correct_rate(
    responses: List[dict],
    correct_answer: str,
    extract_func: Callable,
    compare_func: Callable[[str, str], bool] = lambda r, c: r == c,
) -> Optional[float]:
    """Calculate the absolute correct rate from responses.

    Args:
        responses: List of response dictionaries
        correct_answer: The expected correct answer
        extract_func: Function to extract and normalize responses, defaults to
            extract_bool_answer
        compare_func: Function to compare normalized responses with correct answer,
            should take (response, correct_answer) and return a boolean

    Returns:
        Float indicating correct rate, or None if invalid
    """
    try:
        valid_responses = [
            extract_func(response.get("response", ""))
            for response in responses
            if response.get("response")
        ]
    except ValueError:
        return None

    if not valid_responses:
        return None

    return sum(1 for r in valid_responses if compare_func(r, correct_answer)) / len(valid_responses)


def calculate_correct_rate_by_round(
    dataframe: pd.DataFrame,
    model_dir: Path,
    max_round_number: int,
    extract_func: Callable,
    compare_func: Callable[[str, str], bool] = lambda r, c: r == c,
) -> pd.DataFrame:
    """Calculate both majority and absolute correct rates for each round.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'id' and 'answer' columns.
        model_dir (Path): Path to the model directory containing debate results.
        max_round_number (int): Maximum number of debate rounds to analyze.
        extract_func: Function to extract and normalize responses, defaults to
            extract_bool_answer
        compare_func: Function to compare normalized responses with correct answer,
            should take (response, correct_answer) and return a boolean

    Returns:
        pd.DataFrame: A DataFrame with two rows (majority and absolute) containing
            correct rates for each round, plus the model configuration.
    """
    model_configuration = model_dir.name
    majority_data = {"model_configuration": model_configuration, "metric": "majority"}
    absolute_data = {"model_configuration": model_configuration, "metric": "absolute"}

    subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
    pbar = tqdm(subdirs, desc=f"Processing {model_configuration}")

    correct_counts = {
        i: {"majority": 0, "absolute": 0.0} for i in range(max_round_number + 1)
    }
    total_counts = {i: 0 for i in range(0, max_round_number + 1)}
    total_debates = 0

    for subdir in pbar:
        round_correct_counts, round_total_counts = process_debate_directory(
            subdir, dataframe, max_round_number, extract_func, compare_func
        )
        for round_num in range(0, max_round_number + 1):
            if round_num in round_correct_counts:
                correct_counts[round_num]["majority"] += round_correct_counts[
                    round_num
                ]["majority"]
                correct_counts[round_num]["absolute"] += round_correct_counts[
                    round_num
                ]["absolute"] 
            total_counts[round_num] += round_total_counts.get(round_num, 0)
        if round_total_counts:
            total_debates += 1

    for round_num in range(0, max_round_number + 1):
        if total_counts[round_num] > 0:
            majority_rate = (
                correct_counts[round_num]["majority"] / total_counts[round_num]
            )
            absolute_rate = (
                correct_counts[round_num]["absolute"] / total_counts[round_num]
            )
        else:
            majority_rate = absolute_rate = 0.0

        majority_data[str(round_num)] = majority_rate
        absolute_data[str(round_num)] = absolute_rate

    return pd.DataFrame([majority_data, absolute_data])
