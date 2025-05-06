import base64
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


def _merge_datasets(
    dataset: pd.DataFrame,
    response_dataset: pd.DataFrame,
    answer_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge datasets based on 'uniq_id' and 'response_id' columns.

    Args:
        dataset: The main dataset to merge.
        response_dataset: The response dataset to merge.
        answer_dataset: The answer dataset to merge.

    Returns:
        A merged DataFrame containing the combined data.
    """
    # Validate if uniq_id exists in both DataFrames
    if "uniq_id" not in dataset.columns or "uniq_id" not in response_dataset.columns:
        raise ValueError("Both DataFrames must contain 'uniq_id' column")

    # Note that duplicate uniq_ids in response DataFrame are allowed
    duplicate_ids = response_dataset["uniq_id"].duplicated().sum()
    if duplicate_ids > 0:
        logger.info(
            f"Found {duplicate_ids} duplicate uniq_ids in response DataFrame. "
            f"All duplicates will be included in the merged result."
        )

    # Perform merge operation for dataset and response
    merged_df = pd.merge(
        dataset, response_dataset, on="uniq_id", how="inner", suffixes=("", "_response")
    )

    logger.info(
        f"Merged dataset ({len(dataset)} rows) and response "
        f"({len(response_dataset)} rows) into a DataFrame with {len(merged_df)} rows"
    )
    # return merged_df
    if "uniq_id" not in answer_dataset.columns:
        raise ValueError("Preference DataFrame must contain 'uniq_id' column")

    merged_df["id_suffix"] = merged_df["uniq_id"].apply(
        lambda x: "_".join(x.split("_")[-2:]) if x.count("_") >= 2 else x
    )

    answer_df = answer_dataset.copy()
    answer_df["id_suffix"] = answer_df["uniq_id"].apply(
        lambda x: "_".join(x.split("_")[-2:]) if x.count("_") >= 2 else x
    )

    result_rows = []

    # Group by id_suffix to find matching pairs
    id_suffix_groups = merged_df.groupby("id_suffix")

    for _, answer_row in answer_df.iterrows():
        id_suffix = answer_row["id_suffix"]
        model_name = answer_row["model_name"]

        if id_suffix not in id_suffix_groups.groups:
            logger.warning(f"ID suffix {id_suffix} not found in merged DataFrame.")
            continue

        # Get all entries with matching suffix
        matching_entries = id_suffix_groups.get_group(id_suffix)
        model_entries = matching_entries[
            matching_entries["model_name"].str.contains(model_name, na=False)
        ]

        if len(model_entries) == 0:
            logger.warning(
                f"No matching entries found for model {model_name} with ID suffix {id_suffix}."
            )
            continue

        model_entry = model_entries.iloc[0].to_dict()

        combined_row = {
            "uniq_id": model_entry["uniq_id"],
            "question": model_entry["question"],
            "response": model_entry["response"]["content"],
            "answer": answer_row["score"],
            "image_path": model_entry["image_path"],
        }

        result_rows.append(combined_row)

    if result_rows:
        result_df = pd.DataFrame(result_rows)
        result_df = result_df.drop(columns=["uniq_id"])

        logger.info(
            f"Final merged DataFrame contains {len(result_df)} rows after merging."
        )
    else:
        logger.warning("No matching rows found after merging.")
        result_df = pd.DataFrame(
            columns=["question", "response", "answer", "image_path"]
        )
    return result_df


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


def _image_path_to_bytes(
    image_path: str, base_path: str = "datasets/JudgeAnything/X2XRawBenchmark"
) -> bytes:
    """
    Convert an image path to bytes.

    Args:
        image_path: Path to the image file.

    Returns:
        Bytes representation of the image.
    """
    try:
        full_path = Path(base_path) / image_path
        if not full_path.exists():
            logger.error(f"Image file not found: {full_path}")
            raise FileNotFoundError(f"Image file not found: {full_path}")

        with open(full_path, "rb") as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode("utf-8")
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission denied when accessing file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        raise


if __name__ == "__main__":
    dataset = _load_json_dataset()
    print("Judge Anything Pair Dataset:")
    print(dataset.head())
    print(dataset.info())

    preference = _load_answer_dataset()
    print("Preference Dataset:")
    preference = preference.sort_values(by="uniq_id")
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

    merged_df = _merge_datasets(dataset, response_dataset, preference)
    print("Merged Dataset:")
    print(merged_df.head())
    print(merged_df.info())

    test_bytes = _image_path_to_bytes("images/VisITBench/103_5809ca8d.jpg")
    print("Image bytes:", test_bytes[:10])  # Print first 10 bytes for brevity