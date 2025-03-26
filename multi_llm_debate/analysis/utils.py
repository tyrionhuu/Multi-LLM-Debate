import json
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
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
    extract_func: Optional[Callable] = None,
) -> float:
    """Calculate the majority vote correct rate for a given model directory.

    Args:
        dataframe: DataFrame containing 'id' and 'answer' columns.
        model_dir: Path to the model directory containing debate results.
        round_number: The debate round number to analyze. Defaults to 0.

    Returns:
        The fraction of debates where the majority vote was correct.
    """
    if extract_func is None:
        raise ValueError("An extraction function must be provided")

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
                    extract_func(response.get("response", ""))
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
    sort_by: Optional[str] = None,
    scale: Optional[float] = None,
    fine_grained: bool = True,
) -> str:
    """Draw a vertical ASCII histogram in the console.

    Bins are arranged horizontally, with bars extending upward.

    Args:
        data: Dictionary of {label: value} pairs to visualize
        title: Title for the histogram
        height: Maximum height of the bars (rows)
        show_percentages: Whether to show percentage values above bars
        sort_by: Sort bins by 'key', 'value', 'value_desc', or None for no sorting
        bar_char: Character to use for drawing bars
        scale: Optional manual scaling factor
        fine_grained: Whether to use half-block characters for more precision

    Returns:
        String containing the complete ASCII histogram

    Raises:
        ValueError: For invalid parameters or data
    """
    if not data:
        return "No data to display"

    # Validate data values
    for key, value in data.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Value for key '{key}' is not numeric")
        if value < 0:
            raise ValueError(f"Value for key '{key}' is negative")

    max_value = max(data.values(), default=0)
    if max_value == 0:
        return "All values are zero"

    # Sort data based on sort_by parameter
    if sort_by == "key":
        sorted_items = sorted(data.items(), key=lambda item: item[0])
    elif sort_by == "value":
        sorted_items = sorted(data.items(), key=lambda item: item[1])
    elif sort_by == "value_desc":
        sorted_items = sorted(data.items(), key=lambda item: -item[1])
    elif sort_by is None:
        sorted_items = data.items()  # Preserve insertion order
    else:
        raise ValueError("sort_by must be None, 'key', 'value', or 'value_desc'")

    sorted_data = {k: v for k, v in sorted_items}

    # Calculate column width based on longest label
    column_width = max(len(str(label)) for label in sorted_data.keys())
    column_width = max(column_width, 5)  # Minimum width
    if len(sorted_data) > 10:
        column_width = min(column_width, 7)

    padding = 1
    total_column_width = column_width + padding
    total_width = len(sorted_data) * total_column_width

    # Calculate scaling factor
    scaling_factor = scale if scale is not None else (height / max_value)

    # Characters for fine-grained display
    full_block = "█"
    half_block = "▄"
    empty_block = " "

    # Calculate bar heights with higher precision if fine_grained is enabled
    if fine_grained:
        exact_heights = {
            label: value * scaling_factor for label, value in sorted_data.items()
        }
        # Integer part of the height (full blocks)
        int_heights = {label: int(height) for label, height in exact_heights.items()}
        # Fractional part (for potential half blocks)
        frac_heights = {
            label: height - int_height
            for label, height, int_height in zip(
                sorted_data.keys(), exact_heights.values(), int_heights.values()
            )
        }
    else:
        # Original rounding behavior
        int_heights = {
            label: round(value * scaling_factor) for label, value in sorted_data.items()
        }
        frac_heights = {label: 0 for label in sorted_data}

    # Calculate percentages
    total_sum = sum(sorted_data.values())
    percentages = {
        label: (value / total_sum * 100) for label, value in sorted_data.items()
    }

    # Build histogram
    result = []
    result.append(title.center(total_width))
    result.append("=" * total_width)

    if show_percentages:
        # Percentage line
        percent_line = ""
        for label in sorted_data:
            percent_text = f"{percentages[label]:.1f}%"
            percent_line += f"{percent_text:^{total_column_width}}"
        result.append(percent_line)

        # Count line
        count_line = ""
        for label, value in sorted_data.items():
            if isinstance(value, float) and value.is_integer():
                count_text = f"({int(value)})"
            else:
                count_text = f"({value:.1f})"
            count_line += f"{count_text:^{total_column_width}}"
        result.append(count_line)
        result.append("-" * total_width)

    # Draw bars
    for h in range(height, 0, -1):
        row = ""
        for label in sorted_data:
            int_height = int_heights[label]
            char = empty_block

            if h <= int_height:
                char = full_block
            elif fine_grained and h == int_height + 1 and frac_heights[label] >= 0.5:
                char = half_block

            row += f"{char * column_width:^{total_column_width}}"
        result.append(row)

    # Add axis line
    result.append("=" * total_width)

    # Add labels with multi-line splitting
    max_label_len = max(len(str(label)) for label in sorted_data)
    if max_label_len <= column_width:
        label_line = ""
        for label in sorted_data:
            label_line += f"{str(label):^{total_column_width}}"
        result.append(label_line)
    else:
        chars_per_line = column_width
        lines_needed = (max_label_len + chars_per_line - 1) // chars_per_line
        for line_idx in range(lines_needed):
            current_line = ""
            for label in sorted_data:
                label_str = str(label)
                start = line_idx * chars_per_line
                end = start + chars_per_line
                part = label_str[start:end] if start < len(label_str) else ""
                current_line += f"{part:^{total_column_width}}"
            result.append(current_line)

    return "\n".join(result)
def load_debate_data(model_dir: Path) -> Optional[pd.DataFrame]:
    """Loads debate data from model directory.
    
    This function attempts to find and load debate data, first looking for
    a debate_rounds.csv file, then searching for debate data in directories
    organized by rounds.
    
    Args:
        model_dir: Directory containing model output data.
        
    Returns:
        DataFrame with debate data or None if data cannot be found/loaded.
    """
    # First try the standard debate_rounds.csv file
    debates_path = model_dir / "debate_rounds.csv"
    if debates_path.exists():
        logger.info(f"Found debate_rounds.csv at {debates_path}")
        return pd.read_csv(debates_path)
    
    # If not found, try to construct data from directories
    logger.info("No debate_rounds.csv found. Attempting to load from directories.")
    
    all_data = []
    
    # Check for round directories (round_0, round_1, etc.)
    for item in model_dir.glob("*"):
        if item.is_dir() and item.name.startswith("round_"):
            try:
                round_num = int(item.name.split("_")[1])
                logger.info(f"Processing round directory: {item.name}")
                
                # Process all files in this round directory
                for task_file in item.glob("*.json"):
                    try:
                        task_id = int(task_file.stem)  # Assumes filename is numeric task ID
                        
                        with open(task_file, "r") as f:
                            task_data = json.load(f)
                        
                        # Extract agent responses from the JSON data
                        # Structure may vary, so this might need adaptation
                        if "responses" in task_data:
                            for agent_idx, response in enumerate(task_data["responses"]):
                                all_data.append({
                                    "task_id": task_id,
                                    "round_number": round_num,
                                    "agent_index": agent_idx,
                                    "agent_id": f"agent_{agent_idx}",
                                    "model": task_data.get("model", "unknown"),
                                    "response": response
                                })
                        else:
                            logger.warning(f"Unexpected JSON structure in {task_file}")
                    
                    except Exception as e:
                        logger.warning(f"Error processing file {task_file}: {e}")
            
            except ValueError:
                logger.warning(f"Could not parse round number from directory: {item.name}")
    
    if not all_data:
        logger.error("Could not find any debate data in directories.")
        return None
    
    # Create DataFrame from collected data
    df = pd.DataFrame(all_data)
    logger.info(f"Constructed debate data from directories: {len(df)} rows")
    return df