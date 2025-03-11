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
        DataFrame with correct rate distribution.
    """
    # Define the bins for correct rate distribution
    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

    # Create an empty DataFrame to store the distribution
    result_df = pd.DataFrame(columns=bin_labels + ["id", "round_number"])

    # Process each unique task
    task_dirs = list(model_dir.iterdir())
    pbar = tqdm(
        total=len(task_dirs),
        desc="Calculating correct rate distribution for round {}".format(round_number),
    )

    for task_dir in task_dirs:
        id = task_dir.name
        task_df = dataframe[dataframe["id"] == id]

        # Get the correct answer for the task
        answer = task_df["answer"].values[0]
        correct_answer = normalize_boolean_answer(answer)
        if correct_answer is None:
            logging.warning(f"Task {id} has an invalid answer: {answer}")
            continue
