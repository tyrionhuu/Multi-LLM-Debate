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


def _merge_dataset(
    dataset: pd.DataFrame,
    response: pd.DataFrame,
    preference: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge dataset, response, and optionally preference DataFrames.

    Args:
        dataset: DataFrame containing the base dataset.
        response: DataFrame containing responses, where uniq_id might not be unique.
        preference: Optional DataFrame containing preference data, where uniq_id
                   is in format like "141_8a093930" while in other dataframes
                   it's like "1732019006_376_7b6a1840". The matching is done on
                   the last two parts (e.g., "376_7b6a1840").

    Returns:
        A merged DataFrame containing data from all input DataFrames.
    """
    # Validate if uniq_id exists in both DataFrames
    if "uniq_id" not in dataset.columns or "uniq_id" not in response.columns:
        raise ValueError("Both DataFrames must contain 'uniq_id' column")

    # Note that duplicate uniq_ids in response DataFrame are allowed
    duplicate_ids = response["uniq_id"].duplicated().sum()
    if duplicate_ids > 0:
        logger.info(
            f"Found {duplicate_ids} duplicate uniq_ids in response DataFrame. "
            f"All duplicates will be included in the merged result."
        )

    # Perform merge operation for dataset and response
    merged_df = pd.merge(
        dataset, response, on="uniq_id", how="inner", suffixes=("", "_response")
    )

    logger.info(
        f"Merged dataset ({len(dataset)} rows) and response "
        f"({len(response)} rows) into a DataFrame with {len(merged_df)} rows"
    )

    # Merge preference if provided
    if preference is not None:
        if "uniq_id" not in preference.columns:
            raise ValueError("Preference DataFrame must contain 'uniq_id' column")

        # Create temporary columns in both dataframes to extract the last two parts of uniq_id
        merged_df["id_suffix"] = merged_df["uniq_id"].apply(
            lambda x: "_".join(x.split("_")[-2:]) if x.count("_") >= 2 else x
        )

        preference_copy = preference.copy()
        preference_copy["id_suffix"] = preference_copy["uniq_id"].apply(
            lambda x: "_".join(x.split("_")[-2:]) if x.count("_") >= 2 else x
        )

        # Merge with preference using the extracted suffixes
        merged_df = pd.merge(
            merged_df,
            preference_copy,
            on="id_suffix",
            how="left",
            suffixes=("", "_preference"),
        )

        # Drop the temporary column
        merged_df = merged_df.drop(columns=["id_suffix"])

        # Rename the preference uniq_id to avoid confusion
        if "uniq_id_preference" not in merged_df.columns:
            merged_df = merged_df.rename(columns={"uniq_id_y": "preference_id"})
        else:
            merged_df = merged_df.rename(
                columns={"uniq_id_preference": "preference_id"}
            )

        logger.info(
            f"Merged with preference dataset ({len(preference)} rows) "
            f"resulting in a DataFrame with {len(merged_df)} rows"
        )

    return merged_df


if __name__ == "__main__":
    # Example usage
    dataset = _load_json_dataset()
    print("Judge Anything Pair Dataset:")
    print(dataset.head())
    print(dataset.columns)

    preference = _load_preference_dataset()
    print("Preference Dataset:")
    print(preference.head())
    print(preference.columns)

    response_dataset = _load_response_dataset()
    print("Response Dataset:")
    print(response_dataset.head())
    print(response_dataset.columns)

    merged_df = _merge_dataset(dataset, response_dataset, preference)
    print("Merged Dataset, Response, and Preference:")
    print(merged_df.head())
    print(merged_df.columns)
