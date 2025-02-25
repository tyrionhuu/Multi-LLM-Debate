import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from ..llm.parsers import extract_bool_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_debate_round(
    round_file: Path, correct_answer: str
) -> Tuple[Optional[bool], bool]:
    """Process a single debate round file and determine if the majority answer is correct.

    Args:
        round_file: Path to the debate round JSON file
        correct_answer: The expected correct answer

    Returns:
        Tuple containing:
        - Boolean indicating if majority answer was correct (None if invalid)
        - Boolean indicating if debate should end
    """
    try:
        with open(round_file, "r") as f:
            responses = json.load(f)

        try:
            normalized_responses = [
                extract_bool_answer(response.get("response", ""))
                for response in responses
                if response.get("response")
            ]
        except ValueError as e:
            logger.debug(f"Error processing responses: {str(e)}")
            return None, True

        if not normalized_responses:
            logger.debug("No valid responses found, skipping round")
            return None, True

        correct_ratio = sum(
            1 for r in normalized_responses if r == correct_answer
        ) / len(normalized_responses)
        
        return correct_ratio >= 0.5, False

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug(f"Error processing round: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None, True

def process_debate_directory(
    subdir: Path, 
    dataframe: pd.DataFrame, 
    max_round_number: int
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Process a single debate directory and calculate correctness counts.

    Args:
        subdir: Path to the debate directory
        dataframe: DataFrame containing correct answers
        max_round_number: Maximum number of rounds to process

    Returns:
        Tuple of (correct_counts, total_counts) dictionaries
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
    correct_counts = {i: 0 for i in range(0, max_round_number + 1)}
    total_counts = {i: 0 for i in range(0, max_round_number + 1)}
    
    last_result = None
    debate_ended = False

    for round_num in range(0, max_round_number + 1):
        if debate_ended:
            total_counts[round_num] += 1
            if last_result:
                correct_counts[round_num] += 1
            continue

        round_file = subdir / f"debate_round_{round_num}.json"
        if not round_file.exists():
            debate_ended = True
            if last_result is not None:
                total_counts[round_num] += 1
                if last_result:
                    correct_counts[round_num] += 1
            continue

        round_result, should_end = process_debate_round(round_file, correct_answer)
        if should_end:
            debate_ended = True
            continue

        if round_result is not None:
            total_counts[round_num] += 1
            if round_result:
                correct_counts[round_num] += 1
            last_result = round_result

    return correct_counts, total_counts

def count_majority_responses(responses: List[dict], correct_answer: str) -> Optional[bool]:
    """Count and determine if majority of responses match correct answer.

    Args:
        responses: List of response dictionaries
        correct_answer: The expected correct answer

    Returns:
        Boolean indicating if majority was correct, or None if invalid/tie
    """
    try:
        valid_responses = [
            extract_bool_answer(response.get("response", ""))
            for response in responses
            if response.get("response")
        ]
    except ValueError:
        return None

    if not valid_responses:
        return None

    response_counts = {}
    for response in valid_responses:
        response_counts[response] = response_counts.get(response, 0) + 1

    max_count = max(response_counts.values())
    most_common = [r for r, c in response_counts.items() if c == max_count]

    if len(most_common) > 1:
        return None

    return most_common[0] == correct_answer

def calculate_correct_rate_by_round(
    dataframe: pd.DataFrame, model_dir: Path, max_round_number: int
) -> pd.DataFrame:
    """Calculate the correct rate for each round of debates.

    This function processes debate data stored in JSON files for various
    rounds. It reads the correct answers from the provided dataframe, then
    compares them with normalized responses in each debate round. When a
    debate ends early, the last known correctness is replicated across
    remaining rounds.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'id' and 'answer' columns.
        model_dir (Path): Path to the model directory containing debate results.
        max_round_number (int): Maximum number of debate rounds to analyze.

    Returns:
        pd.DataFrame: A single-row DataFrame containing correct rates
            for each round, keyed by round number, plus the model configuration.
    """
    model_configuration = model_dir.name
    row_data = {"model_configuration": model_configuration}

    subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
    pbar = tqdm(subdirs, desc=f"Processing {model_configuration}")

    correct_counts = {i: 0 for i in range(0, max_round_number + 1)}
    total_counts = {i: 0 for i in range(0, max_round_number + 1)}
    total_debates = 0

    for subdir in pbar:
        round_correct_counts, round_total_counts = process_debate_directory(
            subdir, dataframe, max_round_number
        )
        for round_num in range(0, max_round_number + 1):
            correct_counts[round_num] += round_correct_counts.get(round_num, 0)
            total_counts[round_num] += round_total_counts.get(round_num, 0)
        if round_total_counts:
            total_debates += 1

    for round_num in range(0, max_round_number + 1):
        correct_rate = (
            correct_counts[round_num] / total_counts[round_num]
            if total_counts[round_num] > 0
            else 0.0
        )
        row_data[str(round_num)] = correct_rate

    return pd.DataFrame([row_data])


def calculate_majority_vote_correct_rate(
    dataframe: pd.DataFrame,
    model_dir: Path,
) -> float:
    """Calculate the majority vote correct rate for a given model directory.

    Cases with equal votes for different answers are not counted in the total.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'id' and 'answer' columns.
        model_dir (Path): Path to the model directory containing debate results.

    Returns:
        float: Majority vote correct rate, excluding tied votes.
    """
    correct_count = 0
    total_count = 0

    for _, row in dataframe.iterrows():
        question_id = str(row["id"])
        correct_answer = str(row["answer"]).lower()
        round_file = model_dir / question_id / "debate_round_0.json"

        if not round_file.exists():
            continue

        with open(round_file, "r") as f:
            responses = json.load(f)

        result = count_majority_responses(responses, correct_answer)
        if result is not None:
            total_count += 1
            if result:
                correct_count += 1

    return correct_count / total_count if total_count > 0 else 0.0


if __name__ == "__main__":
    model_dir = Path("data/bool_q/gemma2:2b(3)")
    dataframe = pd.read_csv("output/bool_q/processed_data.csv", index_col=0)
    max_round_number = 10

    result_df = calculate_correct_rate_by_round(dataframe, model_dir, max_round_number)
    print(result_df)
