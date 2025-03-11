import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..llm.parsers import extract_bool_answer
from .utils import get_final_round, normalize_boolean_answer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_response_answer(response_text: str) -> Optional[bool]:
    """Extract a boolean answer from a response text with error handling.

    Args:
        response_text: The text to extract the answer from.

    Returns:
        A boolean value representing the answer, or None if extraction fails.
    """
    try:
        return extract_bool_answer(response_text)
    except ValueError:
        return None


def process_responses(
    responses: List[Dict[str, Any]], ground_truth: bool
) -> Tuple[int, int, int]:
    """Process debate responses and count correct answers.

    Args:
        responses: List of response dictionaries from the debate.
        ground_truth: The correct answer as a boolean value.

    Returns:
        Tuple containing (correct_count, total_valid_responses, invalid_count).
    """
    correct_count = 0
    invalid_count = 0
    total_responses = len(responses)

    for response in responses:
        response_text = response.get("response", "")
        extracted_response = extract_response_answer(response_text)

        if extracted_response is None:
            invalid_count += 1
            continue

        if extracted_response == ground_truth:
            correct_count += 1

    valid_count = total_responses - invalid_count
    return correct_count, valid_count, invalid_count


def process_task_round(
    task_id: Union[int, str],
    task_dir: Path,
    round_num: int,
    actual_round: int,
    ground_truth: bool,
    bin_names: List[str],
) -> Optional[Dict[str, Any]]:
    """Process a single round for a debate task.

    Args:
        task_id: ID of the task being processed.
        task_dir: Path to the task directory.
        round_num: The round number to report in the result.
        actual_round: The actual round number to read data from.
        ground_truth: The correct answer as a boolean.
        bin_names: List of bin names for result categorization.

    Returns:
        A dictionary representing the results row, or None if processing fails.
    """
    logger.debug(
        f"Task {task_id}: Processing round {round_num} "
        f"(using data from round {actual_round})"
    )
    response_file = task_dir / f"debate_round_{actual_round}.json"

    if not response_file.exists():
        logger.debug(
            f"Task {task_id}: Response file for round {actual_round} not found"
        )
        return None

    try:
        # Read the response file
        with open(response_file, "r") as f:
            responses = json.load(f)

        logger.debug(
            f"Task {task_id}: Loaded {len(responses)} responses for round {actual_round}"
        )

        correct_count, valid_count, invalid_count = process_responses(
            responses, ground_truth
        )

        logger.debug(
            f"Task {task_id}: Round {actual_round} summary - "
            f"{correct_count}/{valid_count} correct, "
            f"{invalid_count} invalid responses"
        )

        # Calculate correct rate
        if valid_count > 0:
            correct_rate = correct_count / valid_count
            logger.debug(
                f"Task {task_id}: Round {actual_round} correct rate: {correct_rate:.3f}"
            )

            # Find which bin this correct rate falls into
            bin_idx = min(int(correct_rate * 10), 9)  # Ensure index is within range
            logger.debug(
                f"Task {task_id}: Round {actual_round} falls into bin {bin_names[bin_idx]}"
            )

            # Create a row with zeros
            row = {bin_name: 0 for bin_name in bin_names}
            row[bin_names[bin_idx]] = 1  # Set the correct bin to 1
            row["id"] = task_id
            row["round_number"] = round_num

            return row

    except Exception as e:
        logger.error(
            f"Error processing task {task_id} for round {round_num}: {e}", exc_info=True
        )

    return None


def process_debate_task(
    task_dir: Path,
    dataframe: pd.DataFrame,
    max_round_number: int,
    bin_names: List[str],
) -> List[Dict[str, Any]]:
    """Process a single debate task and calculate accuracy by round.

    Args:
        task_dir: Path to the task directory.
        dataframe: DataFrame containing ground truth data.
        max_round_number: Maximum number of rounds to process.
        bin_names: List of bin names for result categorization.

    Returns:
        List of dictionaries representing results for each valid round.
    """
    task_results = []
    task_id = task_dir.name

    try:
        if isinstance(task_id, str) and task_id.isdigit():
            task_id = int(task_id)
    except ValueError:
        pass

    logger.debug(f"Processing task ID: {task_id}")

    final_round = get_final_round(task_dir)
    if final_round == -1:
        logger.debug(f"Skipping task {task_id}: Final round not found")
        return []

    # Filter dataframe for this task
    task_df = dataframe[dataframe["id"] == task_id]
    if task_df.empty:
        logger.debug(f"Skipping task {task_id}: Not found in dataframe")
        return []

    ground_truth = task_df["answer"].iloc[0]
    logger.debug(f"Task {task_id} ground truth: {ground_truth}")

    # Convert ground truth to normalized boolean format
    normalized_truth = normalize_boolean_answer(ground_truth)
    if normalized_truth is None:
        logger.debug(
            f"Skipping task {task_id}: Invalid ground truth format '{ground_truth}'"
        )
        return []

    # Process each round up to max_round_number
    for round_num in range(1, max_round_number + 1):
        actual_round = min(round_num, final_round)

        row = process_task_round(
            task_id, task_dir, round_num, actual_round, normalized_truth, bin_names
        )

        if row:
            task_results.append(row)

    return task_results


def calculate_per_round_accuracy(
    dataframe: pd.DataFrame, model_dir: Path, max_round_number: int
) -> pd.DataFrame:
    """Calculate accuracy distribution for each round of debate across tasks.

    This function processes a directory of debate tasks and calculates a distribution
    of correct rates for each round. It uses responses from debate rounds to determine
    the accuracy for each task. If a debate converges before the maximum
    round number, the final round's result is used for subsequent rounds.

    Args:
        dataframe: DataFrame containing ground truth data with columns 'id'
            and 'answer'.
        model_dir: Path to the directory containing subdirectories for each task.
        max_round_number: Maximum number of debate rounds to analyze.

    Returns:
        DataFrame with columns for each correct rate bin (0.0-0.1, 0.1-0.2, etc.),
        'id', and 'round_number'. Each bin column contains the count (0 or 1)
        indicating whether the task's correctness for that round falls into that bin.
    """
    subdirs = [subdir for subdir in model_dir.iterdir() if subdir.is_dir()]
    model_configuration = model_dir.name
    logger.info(f"Processing model configuration: {model_configuration}")
    logger.info(f"Found {len(subdirs)} task directories to process")
    pbar = tqdm(subdirs, desc=f"Processing {model_configuration}")

    # Create bins for distribution (0-0.1, 0.1-0.2, ..., 0.9-1.0)
    bin_edges = np.linspace(0, 1, 11)
    bin_names = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(10)]
    logger.debug(f"Created {len(bin_names)} bins: {bin_names}")

    result_data = []

    for task_dir in pbar:
        task_results = process_debate_task(
            task_dir, dataframe, max_round_number, bin_names
        )
        result_data.extend(task_results)

    logger.info(f"Processed {len(result_data)} valid task-round combinations")

    # Create the result dataframe
    if result_data:
        result_df = pd.DataFrame(result_data)
        logger.info(f"Created result DataFrame with shape {result_df.shape}")
    else:
        # If no data was collected, create an empty DataFrame with the correct columns
        columns = bin_names + ["id", "round_number"]
        result_df = pd.DataFrame(columns=columns)
        logger.warning("No valid data collected, returning empty DataFrame")

    return result_df


def print_console_bar_chart(
    data: Dict[str, float], title: str, max_width: int = 50
) -> None:
    """Print a simple bar chart in the console.
    
    Args:
        data: Dictionary of labels and values to plot.
        title: Title of the chart.
        max_width: Maximum width of the bars in characters.
    """
    if not data:
        print(f"No data to plot for '{title}'")
        return
        
    print(f"\n{title}")
    print("=" * (len(title) + 10))
    
    max_val = max(data.values())
    max_label_len = max(len(str(label)) for label in data.keys())
    
    for label, value in data.items():
        bar_width = int((value / max_val) * max_width) if max_val > 0 else 0
        bar = "█" * bar_width
        print(f"{label:{max_label_len}} | {bar} {value:.3f}")


if __name__ == "__main__":
    # Enable debug logging for testing
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # Set up paths
    model_dir = Path("data/bool_q/llama3(7)")
    data_path = Path("output/bool_q/processed_data.csv")
    max_round_number = 5

    logger.info(f"Testing calculate_per_round_accuracy with:")
    logger.info(f"  - Model dir: {model_dir}")
    logger.info(f"  - Data path: {data_path}")
    logger.info(f"  - Max rounds: {max_round_number}")

    # Load dataset - adjust column names to match your actual data structure
    try:
        dataframe = pd.read_csv(data_path)
        logger.info(f"Loaded dataframe with {len(dataframe)} rows")
        logger.debug(f"Dataframe columns: {dataframe.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading dataframe: {e}", exc_info=True)
        exit(1)

    # Calculate accuracy distribution by round
    logger.info("Starting accuracy distribution calculation")
    result_df = calculate_per_round_accuracy(dataframe, model_dir, max_round_number)

    # Print overall summary statistics instead of per-round details
    print(f"Total tasks processed: {len(result_df['id'].unique())}")
    print(f"Total rounds processed: {result_df['round_number'].nunique()}")

    # Calculate overall correct rate across all rounds
    bin_columns = [col for col in result_df.columns if "-" in col]
    bin_midpoints = [float(bin_name.split("-")[0]) + 0.05 for bin_name in bin_columns]
    bin_counts = [result_df[col].sum() for col in bin_columns]

    # Print bin distribution
    print("\nOverall correct rate distribution:")
    for bin_name, count in zip(bin_columns, bin_counts):
        percentage = (count / len(result_df)) * 100 if len(result_df) > 0 else 0
        print(f"  {bin_name}: {count} ({percentage:.1f}%)")

    # Calculate overall weighted average correct rate
    weighted_sum = sum(
        midpoint * count for midpoint, count in zip(bin_midpoints, bin_counts)
    )
    total_count = sum(bin_counts)
    overall_avg_correct_rate = weighted_sum / total_count if total_count > 0 else 0

    print(f"\nOverall average correct rate: {overall_avg_correct_rate:.3f}")

    # Compute model performance by round (summarized)
    round_avg_rates = {}
    for round_num in range(1, max_round_number + 1):
        round_data = result_df[result_df["round_number"] == round_num]
        if not round_data.empty:
            # Calculate average correct rate for this round
            round_bin_counts = [round_data[col].sum() for col in bin_columns]
            round_weighted_sum = sum(
                midpoint * count
                for midpoint, count in zip(bin_midpoints, round_bin_counts)
            )
            round_total_count = sum(round_bin_counts)

            if round_total_count > 0:
                round_avg_rates[round_num] = round_weighted_sum / round_total_count

    # Print concise round performance summary
    print("\nCorrect rate by round:")
    for round_num in sorted(round_avg_rates.keys()):
        print(f"  Round {round_num}: {round_avg_rates[round_num]:.3f}")
    
    # Generate bar charts for each round
    print("\nVisualization of correct rate distributions by round:")
    
    # For each round, create a bar chart showing bin distributions
    for round_num in range(1, max_round_number + 5):
        round_data = result_df[result_df["round_number"] == round_num]
        if not round_data.empty:
            # Calculate distribution for this round
            distribution = {}
            for bin_name in bin_columns:
                bin_count = round_data[bin_name].sum()
                bin_percentage = (bin_count / len(round_data)) * 100
                distribution[bin_name] = bin_percentage
                
            # Print bar chart
            chart_title = f"Round {round_num} Correct Rate Distribution (% of tasks)"
            print_console_bar_chart(distribution, chart_title)
    
    # If available, show first-to-last round improvement
    first_round = min(round_avg_rates.keys()) if round_avg_rates else None
    last_round = max(round_avg_rates.keys()) if round_avg_rates else None
    if first_round is not None and last_round is not None and first_round != last_round:
        improvement = round_avg_rates[last_round] - round_avg_rates[first_round]
        print(
            f"\nImprovement from round {first_round} to {last_round}: "
            f"{improvement:.3f} ({improvement/round_avg_rates[first_round]*100:.1f}%)"
        )
    
    # Generate bar chart comparing rounds
    round_comparison = {f"Round {r}": rate for r, rate in round_avg_rates.items()}
    print_console_bar_chart(
        round_comparison, 
        "Comparison of Average Correct Rates by Round"
    )
