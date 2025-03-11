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
) -> Dict[str, Union[float, int]]:
    """Calculate the overall accuracy based on majority answers for a specific round.

    Args:
        dataframe: DataFrame containing the experiment results.
        model_dir: Directory containing the model outputs.
        round_number: The round number to analyze.

    Returns:
        Dictionary with accuracy metrics including:
        - overall_accuracy: The fraction of tasks where the majority answer was correct
        - total_tasks: Number of tasks evaluated
        - correct_tasks: Number of tasks with correct majority answers
    """
    # Track metrics
    total_tasks = 0
    correct_tasks = 0

    # Process each unique task
    task_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    pbar = tqdm(
        task_dirs,
        desc=f"Calculating accuracy for round {round_number}",
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

            # Determine majority answer
            true_count = sum(1 for r in normalized_responses if r is True)
            false_count = sum(1 for r in normalized_responses if r is False)
                
            majority_answer = false_count if false_count >= true_count else true_count
            
            # Check if majority answer is correct
            if majority_answer == correct_answer:
                correct_tasks += 1
                
            total_tasks += 1

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            continue

    # Calculate accuracy
    overall_accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0.0
    
    logger.info(f"Round {round_number} accuracy: {overall_accuracy:.4f} ({correct_tasks}/{total_tasks})")
    
    return {
        "overall_accuracy": overall_accuracy,
        "total_tasks": total_tasks,
        "correct_tasks": correct_tasks,
        "round_number": round_number
    }
