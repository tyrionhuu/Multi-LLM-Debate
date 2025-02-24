import json
import logging
from pathlib import Path

import pandas as pd

from ..llm.parsers import extract_bool_answer
from ..run.shared.utils import get_latest_round_file

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def calculate_correct_rate_by_round(
    dataframe: pd.DataFrame, model_dir: Path, max_round_number: int
) -> pd.DataFrame:
    """Calculate the correct rate for each round in the dataframe.

    Args:
        dataframe: DataFrame containing the correct answers indexed by ID (str or int)
        model_dir: Path to the model directory containing debate results
        max_round_number: Maximum number of debate rounds to analyze

    Returns:
        DataFrame with correct rates for each round
    """
    model_configuration = model_dir.name
    row_data = {"model_configuration": model_configuration}

    # Get all subdirectories
    subdirs = [d for d in model_dir.iterdir() if d.is_dir()]

    # Initialize counters for each round
    correct_counts = {i: 0 for i in range(1, max_round_number + 1)}
    total_counts = {i: 0 for i in range(1, max_round_number + 1)}

    for subdir in subdirs:
        question_id = subdir.name
        logger.debug(f"Processing question ID: {question_id}")

        # Try both string and int versions of the ID
        str_id = str(question_id)
        try:
            int_id = int(question_id)
        except ValueError:
            int_id = None

        # Check if either version of the ID exists in the dataframe
        if str_id in dataframe.index:
            actual_id = str_id
        elif int_id in dataframe.index:
            actual_id = int_id
        else:
            logger.debug(f"Skipping {question_id} - not found in dataframe")
            continue

        correct_answer = str(dataframe.loc[actual_id, "answer"]).lower()
        logger.debug(f"Correct answer: {correct_answer}")

        try:
            latest_round_file = get_latest_round_file(subdir)
            last_round = int(latest_round_file.stem.split("_")[-1])
            logger.debug(f"Latest round: {last_round}")
        except (ValueError, FileNotFoundError) as e:
            logger.debug(f"Error getting latest round: {e}")
            continue

        for round_num in range(1, min(max_round_number + 1, last_round + 1)):
            round_file = subdir / f"debate_round_{round_num}.json"
            if not round_file.exists():
                continue

            try:
                with open(round_file, "r") as f:
                    round_data = json.load(f)

                responses = round_data.get("responses", [])
                logger.debug(f"Round {round_num} responses: {responses}")

                normalized_responses = [
                    extract_bool_answer(response.get("response", ""))
                    for response in responses
                ]
                logger.debug(f"Normalized responses: {normalized_responses}")

                valid_responses = [r for r in normalized_responses if r]
                logger.debug(f"Valid responses: {valid_responses}")

                total_counts[round_num] += 1
                if (
                    len(set(valid_responses)) == 1
                    and valid_responses[0] == correct_answer
                ):
                    logger.debug(f"Round {round_num}: Correct answer found!")
                    correct_counts[round_num] += 1
                else:
                    logger.debug(
                        f"Round {round_num}: Incorrect - "
                        f"unique responses: {set(valid_responses)}, "
                        f"expected: {correct_answer}"
                    )

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"Error processing round {round_num}: {e}")
                continue

    # Calculate correct rates for each round
    for round_num in range(1, max_round_number + 1):
        if total_counts[round_num] > 0:
            correct_rate = correct_counts[round_num] / total_counts[round_num]
        else:
            correct_rate = 0.0
        logger.debug(
            f"Round {round_num}: {correct_counts[round_num]} correct out of "
            f"{total_counts[round_num]} total = {correct_rate:.2%}"
        )
        row_data[str(round_num)] = correct_rate

    return pd.DataFrame([row_data])


if __name__ == "__main__":
    # Example usage
    model_dir = Path("data/bool_q/gemma2:2b(3)")
    dataframe = pd.read_csv("output/bool_q/processed_data.csv", index_col=0)
    max_round_number = 10

    result_df = calculate_correct_rate_by_round(dataframe, model_dir, max_round_number)
    print(result_df)
