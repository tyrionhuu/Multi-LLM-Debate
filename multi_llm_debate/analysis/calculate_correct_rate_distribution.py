import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..llm.parsers import extract_bool_answer
from .utils import compare_bool, get_final_round

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level
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
    logger.debug(f"Using bin labels: {bin_labels}")

    # Create an empty DataFrame to store the distribution
    result_data = []

    # Process each unique task
    task_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(task_dirs)} task directories to process")
    
    pbar = tqdm(
        task_dirs,
        desc=f"Calculating correct rate distribution for round {round_number}",
    )

    successful_tasks = 0
    skipped_tasks = 0

    for task_dir in pbar:
        task_id = task_dir.name
        # Convert to string for consistent comparison
        task_id_str = str(task_id)
        
        logger.debug(f"Processing task ID: {task_id}")

        # Filter dataframe for this task using string comparison
        task_df = dataframe[dataframe["id"].astype(str) == task_id_str]

        if task_df.empty:
            logger.debug(f"Skipping task {task_id}: Not found in dataframe")
            skipped_tasks += 1
            continue

        # Get the correct answer for the task
        answer = task_df["answer"].iloc[0]
        logger.debug(f"Task {task_id} correct answer: {answer}")

        # Load the debate data for the specified round
        final_round = get_final_round(task_dir)
        if final_round == -1:
            logger.warning(f"No debate data found for task {task_id}")
            skipped_tasks += 1
            continue

        # Use the specified round or the final round if the specified round exceeds it
        actual_round = min(round_number, final_round)
        logger.debug(f"Using round {actual_round} for task {task_id} (final round was {final_round})")
        
        round_file = task_dir / f"debate_round_{actual_round}.json"

        if not round_file.exists():
            logger.warning(
                f"No debate data found for task {task_id} in round {actual_round}"
            )
            skipped_tasks += 1
            continue

        try:
            # Read responses from the round file
            with open(round_file, "r") as f:
                responses = json.load(f)
            
            logger.debug(f"Loaded {len(responses)} responses for task {task_id}")

            # Extract and normalize responses
            normalized_responses = []
            for i, response in enumerate(responses):
                try:
                    extracted = extract_bool_answer(response.get("response", ""))
                    if extracted is not None:
                        normalized_responses.append(extracted)
                        logger.debug(f"Response {i}: Extracted {extracted}")
                    else:
                        logger.debug(f"Response {i}: Could not extract boolean answer")
                except Exception as e:
                    logger.debug(f"Error extracting response {i}: {e}")

            if not normalized_responses:
                logger.debug(
                    f"No valid responses for task {task_id} in round {actual_round}"
                )
                skipped_tasks += 1
                continue

            # Calculate correct rate for this task
            correct_count = sum(
                1 for r in normalized_responses if compare_bool(r, answer)
            )
            correct_rate = correct_count / len(normalized_responses)
            
            logger.debug(
                f"Task {task_id}: correct_count={correct_count}, "
                f"total={len(normalized_responses)}, correct_rate={correct_rate:.2f}"
            )

            # Determine which bin this correct rate falls into
            bin_idx = min(int(correct_rate * 10), 9)  # Ensure index is within range
            logger.debug(f"Task {task_id}: assigned to bin {bin_labels[bin_idx]}")

            # Create a row with zeros for all bins
            row = {bin_label: 0 for bin_label in bin_labels}
            # Set the appropriate bin to 1
            row[bin_labels[bin_idx]] = 1
            row["task_id"] = task_id
            row["round_number"] = round_number

            result_data.append(row)
            successful_tasks += 1

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            skipped_tasks += 1
            continue

    # Create result DataFrame
    if result_data:
        result_df = pd.DataFrame(result_data)
        logger.info(f"Created distribution DataFrame with {len(result_df)} tasks")
        
        # Add more detailed statistics about distribution
        bin_sums = {bin_label: sum(row[bin_label] for row in result_data) 
                   for bin_label in bin_labels}
        logger.debug(f"Raw bin counts: {bin_sums}")
    else:
        # Create empty DataFrame with correct columns if no data
        result_df = pd.DataFrame(columns=bin_labels + ["task_id", "round_number"])
        logger.warning("No valid data collected for correct rate distribution")

    logger.info(f"Successfully processed {successful_tasks} tasks")
    logger.info(f"Skipped {skipped_tasks} tasks")
    
    return result_df


if __name__ == "__main__":
    import sys

    # Hardcoded configuration
    data_path = "output/bool_q/processed_data.csv"
    model_dir = "data/bool_q/llama3(7)"
    round_number = 1
    output_path = (
        "output/distribution_round_1.csv"  # Set to None if you don't want to save
    )

    # Load data
    try:
        dataframe = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}: {len(dataframe)} rows")
        logger.debug(f"DataFrame columns: {dataframe.columns.tolist()}")
        logger.debug(f"DataFrame 'id' column sample: {dataframe['id'].head().tolist()}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    model_dir_path = Path(model_dir)
    if not model_dir_path.exists() or not model_dir_path.is_dir():
        logger.error(f"Model directory does not exist: {model_dir}")
        sys.exit(1)

    # Calculate distribution
    logger.info("Starting distribution calculation")
    result_df = calculate_correct_rate_distribution_for_round_n(
        dataframe=dataframe, model_dir=model_dir_path, round_number=round_number
    )

    # Print summary
    bin_columns = [col for col in result_df.columns if "-" in col]
    task_count = len(result_df)

    logger.info(f"Results for round {round_number}:")
    logger.info(f"Total tasks analyzed: {task_count}")

    # Calculate bin distribution
    if not result_df.empty and bin_columns:
        bin_sums = result_df[bin_columns].sum()
        bin_percentages = (bin_sums / task_count * 100).to_dict()

        # Log raw counts in addition to percentages
        logger.info("Bin distribution raw counts:")
        for bin_label in bin_columns:
            count = bin_sums[bin_label]
            logger.info(f"  {bin_label}: {count}")

        logger.info("Correct rate distribution:")
        for bin_label, percentage in bin_percentages.items():
            logger.info(f"  {bin_label}: {percentage:.2f}%")

    # Save results if needed
    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
