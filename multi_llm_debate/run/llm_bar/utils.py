import logging
import os
import random
import re
from pathlib import Path
from typing import Literal, Union, Optional
import json
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_llm_bar_dataset(
    dataset_path: Union[str, Path] = "datasets/LLMBar",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Load the LLMBar dataset.

    Args:
        dataset_path: Path to the dataset directory. If it exists locally,
            it will be loaded from disk; otherwise, it will be downloaded.
        random_state: Random seed for shuffling. If None, the dataset will be
            randomized differently each time.

    Returns:
        pd.DataFrame: DataFrame containing the LLMBar data with randomized order.
    """
    dataset_path = Path(dataset_path)
    with open(dataset_path / "LLMBar/Natural/dataset.json", "r") as f:
        json_data = json.load(f)
    df = pd.DataFrame(json_data)
    return df

def main():
    dataset = load_llm_bar_dataset()
    print(dataset.head())
if __name__ == "__main__":
    main()
    