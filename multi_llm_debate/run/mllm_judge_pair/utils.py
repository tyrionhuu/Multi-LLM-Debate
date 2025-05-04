import json
import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

MLLM_JUDGE_PAIR_DATASET_FILE = "datasets/MLLM-Judge/pair_data.tsv"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

def load_mllm_judge_pairs(
    file_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load MLLM-Judge pair dataset from TSV file.
    
    Args:
        file_path: Path to the TSV file. If None, uses the default path.
        sample_size: Optional number of samples to return from the dataset.
            
    Returns:
        DataFrame containing the pair data with columns: id, image, question, 
        and answer.
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist.
    """
    if file_path is None:
        file_path = MLLM_JUDGE_PAIR_DATASET_FILE
    
    logger.info(f"Loading MLLM-Judge pair data from {file_path}")
    
    try:
        # Read TSV file
        df = pd.read_csv(file_path, sep='\t')
        
        # Validate expected columns
        expected_columns = ['id', 'image', 'question', 'answer']
        if not all(col in df.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in df.columns]
            logger.warning(f"Missing expected columns: {missing}")
        
        # Handle image column which contains byte data
        if 'image' in df.columns:
            # If needed, convert string representation of bytes to actual bytes
            if isinstance(df['image'].iloc[0], str):
                # This handles cases where bytes might be represented as strings
                df['image'] = df['image'].apply(
                    lambda x: x.encode() if isinstance(x, str) else x
                )
        
        logger.info(f"Loaded {len(df)} MLLM-Judge pair examples")
        
        if sample_size is not None:
            if sample_size > len(df):
                logger.warning(f"Requested sample size {sample_size} exceeds dataset size. Returning full dataset.")
                sample_size = len(df)
            df = df.head(sample_size)
        
        logger.info(f"Sampled {len(df)} MLLM-Judge pair examples")
        
        return df
    
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
        raise
    except Exception as e:
        logger.exception(f"Error loading dataset: {e}")
        raise
    
if __name__ == "__main__":
    # Example usage
    df = load_mllm_judge_pairs(sample_size=5)
    print(df.head())