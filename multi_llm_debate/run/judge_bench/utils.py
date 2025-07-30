import logging
import os
import random
import re
from typing import Literal, Optional

import pandas as pd

# Try to import datasets library, fallback to direct loading if not available
try:
    from datasets import load_dataset, load_from_disk

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

DATASET_PATH = "datasets/JudgeBench"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
logger = logging.getLogger(__name__)


def load_judge_bench_dataset(
    dataset_path: str = DATASET_PATH,
    base_path: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load the JudgeBench dataset.

    Args:
        dataset_path: Path to the dataset directory. If it exists locally,
            it will be loaded from disk; otherwise, it will be downloaded.
        sample_size: Optional; if provided, the DataFrame will be sampled to this size.

    Returns:
        pd.DataFrame: DataFrame containing the JudgeBench data with randomized order.
    """
    # Initialize empty DataFrames
    df_1, df_2 = None, None
    # Check if base_path is provided
    if base_path is not None:
        dataset_path = os.path.join(base_path, dataset_path)
    else:
        # Use absolute path from the current file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, dataset_path)

    # Try to load local datasets first
    print(f"Checking if dataset path exists: {dataset_path}")
    print(f"Path exists: {os.path.exists(dataset_path)}")
    if os.path.exists(dataset_path):
        try:
            if DATASETS_AVAILABLE:
                # Use datasets library if available
                gpt_path = os.path.join(dataset_path, "gpt")
                claude_path = os.path.join(dataset_path, "claude")

                # Check if both splits exist and are not empty
                if os.path.exists(gpt_path) and os.path.exists(claude_path):
                    dataset_1 = load_from_disk(gpt_path)
                    dataset_2 = load_from_disk(claude_path)
                    df_1 = pd.DataFrame(dataset_1)
                    df_2 = pd.DataFrame(dataset_2)
                    print(
                        "Loaded GPT and Claude splits from local paths using datasets library."
                    )
                else:
                    print(
                        "One or both of the splits are missing. Downloading datasets..."
                    )
                    df_1, df_2 = None, None
            else:
                # Direct loading without datasets library
                dataset_files_path = os.path.join(
                    dataset_path,
                    "ScalerLab___judge_bench/default/0.0.0/57dd5e0b9817d07f05ec8f45a91b2ce1e310e308",
                )
                gpt_file = os.path.join(dataset_files_path, "judge_bench-gpt.arrow")
                claude_file = os.path.join(
                    dataset_files_path, "judge_bench-claude.arrow"
                )

                print(f"Looking for files:")
                print(f"  Dataset path: {dataset_path}")
                print(f"  Dataset files path: {dataset_files_path}")
                print(f"  GPT file: {gpt_file}")
                print(f"  Claude file: {claude_file}")
                print(f"  GPT file exists: {os.path.exists(gpt_file)}")
                print(f"  Claude file exists: {os.path.exists(claude_file)}")

                if os.path.exists(gpt_file) and os.path.exists(claude_file):
                    # Load arrow files directly
                    try:
                        df_1 = pa.ipc.open_file(gpt_file).read_pandas()
                        df_2 = pa.ipc.open_file(claude_file).read_pandas()
                        print("Loaded GPT and Claude splits from local arrow files.")
                    except Exception as e:
                        print(f"Error reading arrow files: {e}")
                        # Try alternative approach
                        try:
                            df_1 = pa.feather.read_feather(gpt_file)
                            df_2 = pa.feather.read_feather(claude_file)
                            print("Loaded GPT and Claude splits using feather format.")
                        except Exception as e2:
                            print(f"Error reading feather files: {e2}")
                            df_1, df_2 = None, None
                else:
                    print("Arrow files not found. Cannot load dataset.")
                    df_1, df_2 = None, None

        except Exception as e:
            print(f"Error loading local dataset: {e}")
            df_1, df_2 = None, None

    # Fall back to downloading if local loading failed
    if df_1 is None:
        if DATASETS_AVAILABLE:
            print("Local GPT split not found, downloading from HuggingFace...")
            dataset_1 = load_dataset(
                "ScalerLab/JudgeBench",
                split="gpt",
                cache_dir=dataset_path,
            )
            if dataset_1 is None:
                raise ValueError("Failed to load the JudgeBench GPT dataset.")
            df_1 = pd.DataFrame(dataset_1)
        else:
            raise ValueError(
                "Datasets library not available and local files not found. Cannot load JudgeBench GPT dataset."
            )

    if df_2 is None:
        if DATASETS_AVAILABLE:
            print("Local Claude split not found, downloading from HuggingFace...")
            dataset_2 = load_dataset(
                "ScalerLab/JudgeBench",
                split="claude",
                cache_dir=dataset_path,
            )
            if dataset_2 is None:
                raise ValueError("Failed to load the JudgeBench Claude dataset.")
            df_2 = pd.DataFrame(dataset_2)
        else:
            raise ValueError(
                "Datasets library not available and local files not found. Cannot load JudgeBench Claude dataset."
            )

    # Concatenate the two DataFrames
    df = pd.concat([df_1, df_2], ignore_index=True)

    # Preprocess the dataframe (inlined from preprocess_dataframe)
    # Rename columns
    df = df.rename(columns={"pair_id": "id", "label": "answer"})
    # Drop unnecessary columns
    df = df.drop(columns=["original_id", "source"], errors="ignore")
    df["id"] = range(len(df))

    if sample_size is not None:
        if sample_size > len(df):
            logger.warning(
                f"Sample size {sample_size} is larger than dataset size {len(df)}. Using full dataset."
            )
            sample_size = len(df)

        df = df.head(sample_size)
    return df


def extract_caption_a_b_answer(response: str) -> Literal["A", "B"]:
    """
    Extract answer from the response string.

    First tries to find "Final Answer: A" or "Final Answer: B" pattern.
    If not found, falls back to finding the last occurrence of A or B.

    Args:
        response: The response string from the LLM.

    Returns:
        Literal["A", "B"]: Answer "A" or "B".
    """
    # Try to find "Final Answer: X" pattern
    match = re.search(r"Final Answer:\s*([AB])", response)
    if match:
        return match.group(1)
    else:
        raise ValueError(
            "No valid answer found in the response. Please ensure the response contains 'Final Answer: A' or 'Final Answer: B'."
        )


def extract_bigger_char(comparison: str) -> str:
    """
    Extract the bigger character from the comparison string.

    Args:
        comparison: The comparison string from the LLM.

    Returns:
        str: The bigger character from the comparison string.
    """
    char1 = comparison[0]
    operator = comparison[1]
    char2 = comparison[2]

    if operator == ">":
        return char1
    elif operator == "<":
        return char2
    else:
        raise ValueError("Invalid comparison operator")


def compare_judge_bench_response(
    response: Literal["A", "B"],
    answer: str,
) -> bool:
    """Compare the responses from the judge bench.

    Args:
        response: The response string from the LLM.
        answer: The correct answer to the question ("A"/"B").

    Returns:
        bool: True if the response matches the answer, False otherwise.
    """
    try:
        gold_answer = extract_bigger_char(answer)
        return response == gold_answer.upper()
    except Exception as e:
        print(f"Error comparing responses: {e}")
        return False


def main() -> None:
    df = load_judge_bench_dataset(sample_size=100)
    print("\nFirst entry in the JudgeBench dataset:")
    # Print the complete first entry as a dictionary for clarity
    print(df.iloc[0].to_dict())


if __name__ == "__main__":
    # Run the main function to load and display the dataset
    main()
