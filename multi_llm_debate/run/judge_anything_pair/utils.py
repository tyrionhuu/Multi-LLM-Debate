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


def load_judge_anything_pairs(
    dataset_file: Union[str, Path] = JUDGE_ANYTHING_PAIR_DATASET_FILE,
    response_file: Union[str, Path] = JUDGE_ANYTHING_PAIR_RESPONSE_FILE,
    preference_file: Union[str, Path] = JUDGE_ANYTHING_PAIR_PREFERENCE_FILE,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Judge Anything pair dataset, response, and preference data.

    Args:
        dataset_file: Path to the dataset JSON file.
        response_file: Path to the response JSON file.
        preference_file: Path to the preference JSON file.
        sample_size: Optional number of samples to return from the dataset.

    Returns:
        DataFrame containing the merged data with columns:
        uniq_id, question, image_path, response_A, response_B, answer.
    """
    dataset = _load_json_dataset(dataset_file)
    response = _load_response_dataset(response_file)
    preference = _load_preference_dataset(preference_file)

    merged_df = _merge_dataset(dataset, response, preference)
    df = merged_df.copy()
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df["id"] = range(len(df))

    if sample_size is not None:
        if sample_size > len(df):
            logger.warning(
                f"Requested sample size {sample_size} is larger than dataset size {len(df)}. "
                "Returning the entire dataset."
            )
            sample_size = len(df)
        df = df.head(sample_size)

    logger.info(f"Loaded {len(df)} pairs from dataset, response, and preference files.")
    return df


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
    df = df.drop(columns=["task_name"])
    return df


def _load_preference_dataset(
    file_path: Union[str, Path] = JUDGE_ANYTHING_PAIR_PREFERENCE_FILE,
) -> pd.DataFrame:
    """
    Load a preference dataset and return it as a DataFrame.

    Filters out entries with 'task_name' different from 'Image2Text',
    entries with 'rubric_name' different from 'overall_score',
    and entries with 'choice' equal to 1.

    Args:
        file_path: Path to the preference dataset JSON file.

    Returns:
        A filtered DataFrame containing preference data.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df["task_name"] == "Image2Text"]
    df = df[df["rubric_name"] == "overall_score"]
    # Filter out entries with choice equal to 1 (ensuring we're dealing with strings)
    df = df[df["choice"].astype(str) != "1"]
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

        if "model_pair" not in preference.columns:
            raise ValueError("Preference DataFrame must contain 'model_pair' column")

        # Create temporary columns in both dataframes to extract the last two parts of uniq_id
        merged_df["id_suffix"] = merged_df["uniq_id"].apply(
            lambda x: "_".join(x.split("_")[-2:]) if x.count("_") >= 2 else x
        )

        preference_copy = preference.copy()
        preference_copy["id_suffix"] = preference_copy["uniq_id"].apply(
            lambda x: "_".join(x.split("_")[-2:]) if x.count("_") >= 2 else x
        )

        # Extract model names from model_pair column (format: model1_vs_model2)
        preference_copy[["model1", "model2"]] = preference_copy["model_pair"].str.split(
            "_vs_", expand=True
        )

        # Create result dataframe to store paired responses
        result_rows = []

        # Group by id_suffix to find matching pairs
        id_suffix_groups = merged_df.groupby("id_suffix")

        for idx, pref_row in preference_copy.iterrows():
            suffix = pref_row["id_suffix"]
            model1 = pref_row["model1"]
            model2 = pref_row["model2"]

            # Skip if no matching entries in merged_df
            if suffix not in id_suffix_groups.groups:
                continue

            # Get all entries with matching suffix
            matching_entries = id_suffix_groups.get_group(suffix)

            # Find entries for each model
            model1_entries = matching_entries[
                matching_entries["model_name"].str.contains(model1, case=False)
            ]
            model2_entries = matching_entries[
                matching_entries["model_name"].str.contains(model2, case=False)
            ]

            # Skip if we don't have matches for both models
            if len(model1_entries) == 0 or len(model2_entries) == 0:
                continue

            # Use the first entry for each model
            model1_entry = model1_entries.iloc[0].to_dict()
            model2_entry = model2_entries.iloc[0].to_dict()

            # Create a new row combining both model responses
            combined_row = {
                "uniq_id": model1_entry["uniq_id"],
                "question": model1_entry["question"],
                "id_suffix": suffix,
                "image_path": model1_entry["image_path"],
                "response_A": model1_entry["response"]["content"],
                "response_B": model2_entry["response"]["content"],
                "answer": "A" if pref_row["choice"] == "0" else "B",
            }

            result_rows.append(combined_row)

        # Create new DataFrame from results
        if result_rows:
            merged_df = pd.DataFrame(result_rows)
            logger.info(
                f"Created {len(merged_df)} paired entries from preference dataset "
                f"({len(preference)} rows)"
            )
        else:
            logger.warning("No matching pairs found between datasets")
            merged_df = pd.DataFrame()

        # Drop any temporary columns if they still exist
        if "id_suffix" in merged_df.columns:
            merged_df = merged_df.drop(columns=["id_suffix"])

        merged_df = merged_df.drop(columns=["uniq_id"])
    return merged_df


if __name__ == "__main__":
    # dataset = _load_json_dataset()
    # print("Judge Anything Pair Dataset:")
    # print(dataset.head())
    # print(dataset.info())

    # preference = _load_preference_dataset()
    # print("Preference Dataset:")
    # print(preference.head())
    # print(preference.info())
    # # print("\nUnique values in 'choice' column:")
    # # print(preference["choice"].unique())

    # # print("\nCounts of values in 'choice' column:")
    # # print(preference["choice"].value_counts())

    # response_dataset = _load_response_dataset()
    # print("Response Dataset:")
    # print(response_dataset.head())
    # print(response_dataset.info())

    # merged_df = _merge_dataset(dataset, response_dataset, preference)
    # print("Merged Dataset, Response, and Preference:")
    # print(merged_df.head())
    # print(merged_df.info())
    # # Display all fields for the first few rows
    # # Display the first few rows as JSON for better readability
    # pd.set_option("display.max_colwidth", None)
    # for i in range(min(5, len(merged_df))):
    #     print(f"\n--- Row {i} ---")
    #     print(json.dumps(merged_df.iloc[i].to_dict(), indent=2))
    pairs_df = load_judge_anything_pairs()
    print(pairs_df.head())
    print(pairs_df.info())