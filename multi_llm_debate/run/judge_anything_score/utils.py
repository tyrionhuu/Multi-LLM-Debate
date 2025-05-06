import json
import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

JUDGE_ANYTHING_SCORE_DATASET_FILE = (
    "datasets/JudgeAnything/X2XRawBenchmark/X2XBenchmark.json"
)

JUDGE_ANYTHING_SCORE_RESPONSE_FILE = (
    "datasets/JudgeAnything/ResponseCollection/X2XBenchmarkResponse.json"
)

JUDGE_ANYTHING_SCORE_ANSWER_FILE = (
    "datasets/JudgeAnything/Preference/Human_Scoring.json"
)


def _load_json_dataset(
    file_path: Union[str, Path] = JUDGE_ANYTHING_SCORE_DATASET_FILE,
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
    file_path: Union[str, Path] = JUDGE_ANYTHING_SCORE_RESPONSE_FILE,
) -> pd.DataFrame:
    """Load a response dataset and return it as a DataFrame."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df["task_name"] == "Image2Text"]
    df = df.drop(columns=["task_name"])
    return df


def _load_answer_dataset(
    file_path: Union[str, Path] = JUDGE_ANYTHING_SCORE_ANSWER_FILE,
) -> pd.DataFrame:
    """
    Load a answer dataset and return it as a DataFrame.

    Filters out entries with 'task_name' different from 'Image2Text',
    entries with 'rubric_name' different from 'overall_score',
    and entries with 'choice' equal to 1.

    Args:
        file_path: Path to the answer dataset JSON file.

    Returns:
        A filtered DataFrame containing answer data.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df["task_name"] == "Image2Text"]
    df = df[df["rubric_name"] == "overall_score"]
    df = df.drop(columns=["task_name", "rubric_name", "comment", "index"])
    return df


if __name__ == "__main__":
    dataset = _load_json_dataset()
    print("Judge Anything Pair Dataset:")
    print(dataset.head())
    print(dataset.info())

    preference = _load_answer_dataset()
    print("Preference Dataset:")
    print(preference.head())
    print(preference.info())
    print("\nUnique values in 'score' column:")
    print(preference["score"].unique())

    print("\nCounts of values in 'score' column:")
    print(preference["score"].value_counts())

    response_dataset = _load_response_dataset()
    print("Response Dataset:")
    print(response_dataset.head())
    print(response_dataset.info())
