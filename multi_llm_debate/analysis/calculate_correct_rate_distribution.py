import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..llm.parsers import extract_bool_answer
from .utils import get_final_round

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
            and 'ground_truth'.
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
        id = task_dir.name
        if isinstance(id, str):
            id = int(id)
        logger.debug(f"Processing task ID: {id}")
        final_round = get_final_round(task_dir)

        if final_round == -1:
            logger.debug(f"Skipping task {id}: Final round not found")
            continue

        # Filter dataframe for this task
        task_df = dataframe[dataframe["id"] == id]
        if task_df.empty:
            logger.debug(f"Skipping task {id}: Not found in dataframe")
            continue

        ground_truth = task_df["answer"].iloc[0]
        logger.debug(f"Task {id} ground truth: {ground_truth}")

        # Convert ground truth to normalized boolean format
        processed_answer = str(ground_truth).lower().strip()
        if processed_answer in ["yes", "true", "1"]:
            answer_bool = True
        elif processed_answer in ["no", "false", "0"]:
            answer_bool = False
        else:
            logger.debug(f"Skipping task {id}: Invalid ground truth format '{processed_answer}'")
            continue

        # Process each round up to max_round_number
        for round_num in range(1, max_round_number + 1):
            actual_round = min(round_num, final_round)
            logger.debug(f"Task {id}: Processing round {round_num} (using data from round {actual_round})")
            response_file = task_dir / f"debate_round_{actual_round}.json"

            if not response_file.exists():
                logger.debug(f"Task {id}: Response file for round {actual_round} not found")
                continue

            try:
                # Read the response file
                with open(response_file, "r") as f:
                    responses = json.load(f)
                logger.debug(f"Task {id}: Loaded {len(responses)} responses for round {actual_round}")

                # Count correct responses
                correct_count = 0
                total_responses = len(responses)
                invalid_responses = 0

                # Count correct responses in the round
                for i, response in enumerate(responses):
                    response_text = response["response"]
                    extracted_response = extract_bool_answer(response_text)

                    # Skip invalid responses
                    if extracted_response is None:
                        invalid_responses += 1
                        total_responses -= 1
                        logger.debug(f"Task {id}: Invalid response {i} in round {actual_round}")
                        continue

                    # Compare with ground truth
                    is_correct = str(extracted_response).lower() == str(answer_bool).lower()
                    if is_correct:
                        correct_count += 1
                    logger.debug(f"Task {id}: Response {i} in round {actual_round} - "
                                f"extracted: {extracted_response}, correct: {is_correct}")

                logger.debug(f"Task {id}: Round {actual_round} summary - "
                           f"{correct_count}/{total_responses} correct, "
                           f"{invalid_responses} invalid responses")

                # Calculate correct rate
                if total_responses > 0:
                    correct_rate = correct_count / total_responses
                    logger.debug(f"Task {id}: Round {actual_round} correct rate: {correct_rate:.3f}")

                    # Find which bin this correct rate falls into
                    bin_idx = min(
                        int(correct_rate * 10), 9
                    )  # Ensure index is within range
                    logger.debug(f"Task {id}: Round {actual_round} falls into bin {bin_names[bin_idx]}")

                    # Create a row with zeros
                    row = {bin_name: 0 for bin_name in bin_names}
                    row[bin_names[bin_idx]] = 1  # Set the correct bin to 1
                    row["id"] = id
                    row["round_number"] = round_num

                    result_data.append(row)
            except Exception as e:
                logger.error(f"Error processing task {id} for round {round_num}: {e}", exc_info=True)
                continue

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


if __name__ == "__main__":
    # Enable debug logging for testing
    logger.setLevel(logging.WARNING)
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
        logger.debug(f"First few rows: \n{dataframe.head().to_string()}")
    except Exception as e:
        logger.error(f"Error loading dataframe: {e}", exc_info=True)
        exit(1)

    # Calculate accuracy distribution by round
    logger.info("Starting accuracy distribution calculation")
    result_df = calculate_per_round_accuracy(dataframe, model_dir, max_round_number)

    # Print the results
    logger.info("Accuracy distribution by round:")
    logger.info(f"Result shape: {result_df.shape}")
    logger.info(f"\n{result_df.head().to_string()}")

    # Calculate and print summary statistics by round
    round_stats = {}
    for round_num in range(1, max_round_number + 1):
        round_data = result_df[result_df["round_number"] == round_num]
        logger.info(f"Processing statistics for round {round_num}: {len(round_data)} tasks")
        
        if not round_data.empty:
            # Calculate average correct rate for this round
            bin_columns = [col for col in round_data.columns if "-" in col]
            bin_midpoints = [
                float(bin_name.split("-")[0]) + 0.05
                for bin_name in bin_columns
            ]
            bin_counts = [
                round_data[col].sum() for col in bin_columns
            ]

            weighted_sum = sum(
                midpoint * count for midpoint, count in zip(bin_midpoints, bin_counts)
            )
            total_count = sum(bin_counts)
            
            logger.debug(f"Round {round_num} bin counts: {list(zip(bin_columns, bin_counts))}")
            logger.debug(f"Round {round_num} weighted sum: {weighted_sum}, total count: {total_count}")

            avg_correct_rate = weighted_sum / total_count if total_count > 0 else 0
            round_stats[round_num] = {
                "avg_correct_rate": avg_correct_rate,
                "task_count": len(round_data),
            }

    # Print summary statistics
    logger.info("\nSummary statistics by round:")
    for round_num, stats in round_stats.items():
        logger.info(
            f"Round {round_num}: "
            f"Average Correct Rate: {stats['avg_correct_rate']:.3f}, "
            f"Tasks: {stats['task_count']}"
        )
