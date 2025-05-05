import base64
import json
import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

JUDGE_ANYTHING_PAIR_DATASET_FILE = (
    "datasets/JudgeAnything/X2XRawBenchmark/X2XBenchmark.json"
)


def _load_json_dataset(
    file_path: Union[str, Path] = JUDGE_ANYTHING_PAIR_DATASET_FILE,
) -> pd.DataFrame:
    """Load a JSON dataset and return it as a DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Filter DataFrame to only include rows where task_name is "Image2Text"
    df = df[df["task_name"] == "Image2Text"]
    return df


if __name__ == "__main__":
    # Example usage
    dataset = _load_json_dataset()
    print(dataset.head())
    print(dataset.columns)
