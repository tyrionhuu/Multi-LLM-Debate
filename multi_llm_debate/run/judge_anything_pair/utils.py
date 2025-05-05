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
JUDGE_ANYTHING_PAIR_RESPONSE_FILE = (
    "datasets/JudgeAnything/ResponseCollection/X2XBenchmarkResponse.json"
)
JUDGE_ANYTHING_PAIR_PREFERENCE_FILE = (
    "datasets/JudgeAnything/Preference/Human_Pairing.json"
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
    df = df.drop(columns=["task_name", "audio_path", "video_path"])
    return df


def _load_response_dataset(
    file_path: Union[str, Path] = JUDGE_ANYTHING_PAIR_RESPONSE_FILE,
) -> pd.DataFrame:
    """Load a response dataset and return it as a DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df["task_name"] == "Image2Text"]
    return df


def _load_preference_dataset(
    file_path: Union[str, Path] = JUDGE_ANYTHING_PAIR_PREFERENCE_FILE,
) -> pd.DataFrame:
    """Load a preference dataset and return it as a DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df["task_name"] == "Image2Text"]
    df = df[df["rubric_name"] == "overall_score"]
    df = df.drop(columns=["task_name", "rubric_name", "comment"])
    return df


if __name__ == "__main__":
    # Example usage
    dataset = _load_json_dataset()
    print("Judge Anything Pair Dataset:")
    print(dataset.info())
    print(dataset.columns)

    preference = _load_preference_dataset()
    print("Preference Dataset:")
    print(preference.info())
    print(preference.columns)

    response_dataset = _load_response_dataset()
    print("Response Dataset:")
    print(response_dataset.info())
    print(response_dataset.columns)
