import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from multi_llm_debate.utils.download_dataset import load_save_huggingface_dataset_df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_judge_lm_dataset(
    dataset_path: Union[str, Path] = "datasets/JudgeLM",
    dataset_name: str = "BAAI/JudgeLM-100K",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Load the JudgeLM dataset.

    Args:
        dataset_path: Path to the dataset directory. If it exists locally,
            it will be loaded from disk; otherwise, it will be downloaded.
        dataset_name: Name of the dataset to load from Hugging Face.
        random_state: Random seed for shuffling. If None, the dataset will be
            randomized differently each time.

    Returns:
        pd.DataFrame: DataFrame containing the JudgeLM data with randomized order.
    """
    # Initialize empty DataFrame
    df = None
    dataset_path = Path(dataset_path)
    try:
        df = load_save_huggingface_dataset_df(dataset_name, dataset_path)
        logger.info("Loaded JudgeLM dataset from Hugging Face.")
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face: {e}")
        raise e

    # Shuffle the DataFrame if random_state is provided
    if random_state is not None:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df

if __name__ == "__main__":
    # Example usage
    dataset = load_judge_lm_dataset()
    print(dataset.head())