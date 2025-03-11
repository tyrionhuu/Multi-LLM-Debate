import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..llm.parsers import extract_bool_answer
from .utils import compare_int_as_str, get_final_round, normalize_boolean_answer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("analysis.log"),
    ],
)
logger = logging.getLogger(__name__)


def calculate_correct_rate_distribution(
    dataframe: pd.DataFrame,
    model_dir: Path,
    max_round: Optional[int] = None,
):
    pass


def calculate_correct_rate_distribution_for_round_n(
    dataframe: pd.DataFrame,
    model_dir: Path,
    round_number: int,
) -> pd.DataFrame:
    """Calculate the correct rate distribution for a specific round.

    Args:
        dataframe: DataFrame containing the experiment results.
        model_dir: Directory containing the model outputs.
        round_number: The round number to analyze.

    Returns:
        DataFrame with correct rate distribution. Each row represents a task,
        with columns for bins (0-0.1, 0.1-0.2, etc.), task_id, and round_number.
    """
    # Define the bins for correct rate distribution
    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

    # Create an empty DataFrame to store the distribution
    result_data = []

    # Process each unique task
    task_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    pbar = tqdm(
        task_dirs,
        desc=f"Calculating correct rate distribution for round {round_number}",
    )

    for task_dir in pbar:
        task_id = task_dir.name
        # Convert to string for consistent comparison
        task_id_str = str(task_id)

        # Filter dataframe for this task using string comparison
        task_df = dataframe[dataframe["id"].astype(str) == task_id_str]

        if task_df.empty:
            logger.debug(f"Skipping task {task_id}: Not found in dataframe")
            continue

        # Get the correct answer for the task
        answer = task_df["answer"].iloc[0]
        correct_answer = normalize_boolean_answer(answer)
        if correct_answer is None:
            logger.warning(f"Task {task_id} has an invalid answer: {answer}")
            continue

        # Load the debate data for the specified round
        final_round = get_final_round(task_dir)
        if final_round == -1:
            logger.warning(f"No debate data found for task {task_id}")
            continue

        # Use the specified round or the final round if the specified round exceeds it
        actual_round = min(round_number, final_round)
        round_file = task_dir / f"debate_round_{actual_round}.json"

        if not round_file.exists():
            logger.warning(
                f"No debate data found for task {task_id} in round {actual_round}"
            )
            continue

        try:
            # Read responses from the round file
            with open(round_file, "r") as f:
                responses = json.load(f)

            # Extract and normalize responses
            normalized_responses = []
            for response in responses:
                try:
                    extracted = extract_bool_answer(response.get("response", ""))
                    if extracted is not None:
                        normalized_responses.append(extracted)
                except Exception as e:
                    logger.debug(f"Error extracting response: {e}")

            if not normalized_responses:
                logger.debug(
                    f"No valid responses for task {task_id} in round {actual_round}"
                )
                continue

            # Calculate correct rate for this task
            correct_count = sum(1 for r in normalized_responses if r == correct_answer)
            correct_rate = correct_count / len(normalized_responses)

            # Determine which bin this correct rate falls into
            bin_idx = min(int(correct_rate * 10), 9)  # Ensure index is within range

            # Create a row with zeros for all bins
            row = {bin_label: 0 for bin_label in bin_labels}
            # Set the appropriate bin to 1
            row[bin_labels[bin_idx]] = 1
            row["task_id"] = task_id
            row["round_number"] = round_number

            result_data.append(row)

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            continue

    # Create result DataFrame
    if result_data:
        result_df = pd.DataFrame(result_data)
        logger.info(f"Created distribution DataFrame with {len(result_df)} tasks")
    else:
        # Create empty DataFrame with correct columns if no data
        result_df = pd.DataFrame(columns=bin_labels + ["task_id", "round_number"])
        logger.warning("No valid data collected for correct rate distribution")

    return result_df
