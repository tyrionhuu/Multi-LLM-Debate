import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..llm.parsers import extract_bool_answer
from .utils import get_final_round, normalize_boolean_answer, compare_int_as_str

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
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]

    # Create an empty DataFrame to store the distribution
    result_df = pd.DataFrame(columns=bin_labels + ['id', 'round_number'])

    # Process each unique task
    unique_tasks = dataframe['id'].unique()
    rows = []

    for id in tqdm(unique_tasks, desc=f"Processing round {round_number}"):
        task_df = dataframe[compare_int_as_str(dataframe['id'], id)]
        
        # Count correct answers for this task in this round
        correct_count = 0
        total_count = 0
        
        for _, row in task_df.iterrows():
            output_file = model_dir / f"{row['experiment_id']}.json"
            if not output_file.exists():
                continue
                
            with open(output_file, 'r') as f:
                experiment_data = json.load(f)
            
            # Skip if the round doesn't exist
            if round_number >= len(experiment_data['rounds']):
                continue
                
            round_data = experiment_data['rounds'][round_number]
            predicted_answer = extract_bool_answer(round_data['answer'])
            normalized_answer = normalize_boolean_answer(predicted_answer)
            
            if normalized_answer is not None:
                total_count += 1
                if normalized_answer == row['answer']:
                    correct_count += 1
        
        # Calculate correct rate and determine the bin
        if total_count > 0:
            correct_rate = correct_count / total_count
            distribution = np.zeros(len(bin_labels))
            
            # Find the appropriate bin
            bin_index = min(int(correct_rate * 10), len(bin_labels) - 1)
            
            # Create a row for this task
            row_data = {bin_label: 0 for bin_label in bin_labels}
            row_data[bin_labels[bin_index]] = 1
            row_data['id'] = id
            row_data['round_number'] = round_number
            
            rows.append(row_data)

    # Combine all rows into the result DataFrame
    if rows:
        result_df = pd.DataFrame(rows)
        
    return result_df
