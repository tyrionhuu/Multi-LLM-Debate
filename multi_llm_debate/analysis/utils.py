import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..llm.parsers import extract_bool_answer


def compare_bool(value_a: Any, value_b: Any) -> bool:
    """Compare two boolean values with robust type conversion.

    Handles various string representations, boolean values, and numeric values.

    Args:
        value_a: First value to compare
        value_b: Second value to compare

    Returns:
        True if values are equivalent booleans, False otherwise
    """

    # Helper function to normalize to boolean
    def normalize_value(value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        if isinstance(value, str):
            # Normalize string to lowercase
            value_lower = value.lower().strip()

            # Handle common "true" string formats
            if value_lower in ("true", "t", "yes", "y", "1", "correct"):
                return True

            # Handle common "false" string formats
            if value_lower in ("false", "f", "no", "n", "0", "incorrect"):
                return False

            # Handle special case for "right/wrong" answers
            if value_lower in ("right"):
                return True
            if value_lower in ("wrong"):
                return False

            # Try numeric conversion as last resort
            try:
                return bool(float(value))
            except ValueError:
                pass

        # If all else fails, use the boolean value of the object
        return bool(value)

    try:
        # Normalize both values to booleans and compare
        return normalize_value(value_a) == normalize_value(value_b)
    except Exception:
        # If any error occurs, they're not comparable
        return False


def compare_int_as_str(a: Any, b: Any) -> bool:
    """Compares two values that may be integers represented as strings.

    Args:
        a: The first value to compare.
        b: The second value to compare.

    Returns:
        bool: True if the values are equal, False otherwise.
    """
    return str(a) == str(b)


def get_final_round(task_dir: Path) -> int:
    """Gets the final available round number for a given task directory.

    Args:
        task_dir: Path to the task directory.

    Returns:
        int: The highest round number found, or -1 if no rounds exist.
    """
    debate_files = list(task_dir.glob("debate_round_*.json"))
    if not debate_files:
        return -1

    round_numbers = [int(f.stem.split("_")[-1]) for f in debate_files]
    return max(round_numbers)


def normalize_boolean_answer(answer: Any) -> Optional[bool]:
    """Normalize an answer to a boolean value.

    Args:
        answer: The answer to normalize, can be any type.

    Returns:
        A boolean value or None if the answer cannot be normalized.
    """
    processed_answer = str(answer).lower().strip()
    if processed_answer in ["yes", "true", "1"]:
        return True
    elif processed_answer in ["no", "false", "0"]:
        return False
    else:
        return None


def calculate_majority_vote_correct_rate_for_round_n(
    dataframe: pd.DataFrame,
    model_dir: Path,
    round_number: int = 0,
) -> float:
    """Calculate the majority vote correct rate for a given model directory.

    Args:
        dataframe: DataFrame containing 'id' and 'answer' columns.
        model_dir: Path to the model directory containing debate results.
        round_number: The debate round number to analyze. Defaults to 0.

    Returns:
        The fraction of debates where the majority vote was correct.
    """
    total_correct = 0
    total_count = 0

    for _, row in dataframe.iterrows():
        question_id = str(row["id"])
        correct_answer = str(row["answer"]).lower()
        question_dir = model_dir / question_id

        # Skip if question directory doesn't exist
        if not question_dir.exists() or not question_dir.is_dir():
            continue

        # Find the final round number for this question
        final_round = get_final_round(question_dir)
        if final_round == -1:
            continue

        # Use the specified round or the final round if the specified round exceeds it
        actual_round = min(round_number, final_round)
        round_file = question_dir / f"debate_round_{actual_round}.json"

        if not round_file.exists():
            continue

        try:
            with open(round_file, "r") as f:
                responses = json.load(f)

            try:
                normalized_responses = [
                    extract_bool_answer(response.get("response", ""))
                    for response in responses
                    if response.get("response")
                ]
            except ValueError:
                continue

            if not normalized_responses:
                continue

            # Check if majority is correct (more than 50% match correct answer)
            normalized_answer = normalize_boolean_answer(correct_answer)
            if normalized_answer is None:
                continue

            correct_votes = sum(
                1 for r in normalized_responses if r == normalized_answer
            )
            if correct_votes > len(normalized_responses) / 2:
                total_correct += 1
            total_count += 1

        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return total_correct / total_count if total_count > 0 else 0.0
