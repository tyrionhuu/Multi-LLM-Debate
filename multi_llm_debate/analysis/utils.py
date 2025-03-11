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


def draw_console_histogram(
    data: dict,
    title: str = "Distribution",
    height: int = 15,
    show_percentages: bool = True,
) -> str:
    """Draw a vertical ASCII histogram in the console.

    Bins are arranged horizontally, with bars extending upward.

    Args:
        data: Dictionary of {label: value} pairs to visualize
        title: Title for the histogram
        height: Maximum height of the bars
        show_percentages: Whether to show percentage values above bars

    Returns:
        String containing the complete ASCII histogram
    """
    if not data:
        return "No data to display"

    # Find the maximum value for scaling
    max_value = max(data.values())
    if max_value == 0:
        return "All values are zero"

    # Sort data by keys if they're sortable (assumes bin labels)
    try:
        sorted_data = {k: data[k] for k in sorted(data.keys())}
    except TypeError:
        sorted_data = data

    # Calculate column width
    # We want bin labels to be fully visible, so use their length
    column_width = max(len(str(label)) for label in sorted_data.keys())
    # Ensure columns aren't too narrow
    column_width = max(column_width, 5)
    # Use fewer characters if we have many bins
    if len(sorted_data) > 10:
        column_width = min(column_width, 7)
    
    # Column padding (spaces between columns)
    padding = 1
    total_column_width = column_width + padding
    
    # Convert values to heights based on max_value
    heights = {
        label: int(height * value / max_value) if max_value > 0 else 0
        for label, value in sorted_data.items()
    }

    # Calculate percentages
    total_sum = sum(sorted_data.values())
    percentages = {
        label: (value / total_sum * 100) if total_sum > 0 else 0
        for label, value in sorted_data.items()
    }

    # Build the histogram
    result = []
    result.append(title)
    result.append("=" * (len(sorted_data) * total_column_width))

    # Add percentage values at the top
    if show_percentages:
        percent_line = ""
        for label in sorted_data.keys():
            percent_text = f"{percentages[label]:.1f}%"
            percent_line += f"{percent_text:^{total_column_width}}"
        result.append(percent_line)
        
        # Add count values
        count_line = ""
        for label, value in sorted_data.items():
            count_text = f"({value:.0f})" if value >= 1 else f"({value:.2f})"
            count_line += f"{count_text:^{total_column_width}}"
        result.append(count_line)
        
        # Add a separator
        result.append("-" * (len(sorted_data) * total_column_width))

    # Draw the bars from top to bottom
    for h in range(height, 0, -1):
        row = ""
        for label in sorted_data.keys():
            bar_height = heights[label]
            if h <= bar_height:
                row += f"{'█' * column_width:{total_column_width}}"
            else:
                row += f"{'':{total_column_width}}"
        result.append(row)
    
    # Add axis line
    result.append("=" * (len(sorted_data) * total_column_width))

    # Add bin labels at the bottom - split into multiple lines if needed
    if column_width >= len(max(sorted_data.keys(), key=len)):
        # Single line for labels if they fit
        label_line = ""
        for label in sorted_data.keys():
            label_line += f"{str(label):^{total_column_width}}"
        result.append(label_line)
    else:
        # Split labels into multiple lines
        max_label_len = max(len(str(label)) for label in sorted_data.keys())
        chars_per_line = column_width
        lines_needed = (max_label_len + chars_per_line - 1) // chars_per_line
        
        for line_idx in range(lines_needed):
            label_line = ""
            for label in sorted_data.keys():
                start = line_idx * chars_per_line
                end = start + chars_per_line
                part = str(label)[start:end] if start < len(str(label)) else ""
                label_line += f"{part:^{total_column_width}}"
            result.append(label_line)

    return "\n".join(result)
