import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

from multi_llm_debate.utils.download_dataset import load_save_huggingface_dataset_df

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mj_bench_dataset(
    dataset_path: Union[str, Path] = "datasets/MJ-Bench",
    dataset_name: str = "MJ-Bench/MJ-Bench",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Load the MJ-Bench dataset.

    Args:
        dataset_path: Path to the dataset directory. If it exists locally,
            it will be loaded from disk; otherwise, it will be downloaded.
        dataset_name: Name of the dataset to load from Hugging Face.
        random_state: Random seed for shuffling. If None, the dataset will be
            randomized differently each time.

    Returns:
        pd.DataFrame: DataFrame containing the MJ-Bench data with randomized order.
    """
    # Initialize empty DataFrame
    df = None
    dataset_path = Path(dataset_path)
    try:
        df = load_save_huggingface_dataset_df(
            dataset_name=dataset_name, 
            dataset_path=dataset_path,
            split="alignment"
        )
        logger.info("Loaded MJ-Bench dataset from Hugging Face.")
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face: {e}")
        raise e

    # Shuffle the DataFrame if random_state is provided
    if random_state is not None:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def preprocess_mj_bench_dataframe(
    dataframe: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Preprocess the MJ Bench DataFrame to ensure it has all required columns.

    Args:
        dataframe: Input DataFrame from MJ Bench dataset.
        random_state: Random seed for shuffling. If None, the dataset will be
            randomized differently each time.

    Returns:
        pd.DataFrame: Processed DataFrame with all required columns and
            multiple-choice format.
    """
    if dataframe is None:
        raise ValueError("Input DataFrame is None. Please load the dataset first.")

    df = dataframe.copy()
    random.seed(random_state)

    # Create ID column if missing
    if "id" not in df.columns:
        df["id"] = df.index + 1

    return df[
        ["caption", "image0", "image1", "label"]
    ].rename(columns={"label": "answer"})


def extract_0_1_answer(
    response: str,
) -> Literal["0", "1"]:
    """Extract the answer from the response string.

    Args:
        response (str): The response string from the LLM.

    Returns:
        Literal["0", "1"]: Answer "0" or "1".

    Raises:
        ValueError: If no valid answer is found in the response.
    """
    match = re.search(r"Final Answer:\s*([01])", response)
    if match:
        return match.group(1)
    raise ValueError(
        "No valid answer found in the response. Please ensure the response "
        "contains 'Final Answer: 0' or 'Final Answer: 1'."
    )


def compare_mj_bench_response(
    response: Literal["1", "0"],
    answer: Union[str, int],
) -> bool:
    """Compare the responses from the MJ Bench dataset.

    Args:
        response: The response string from the LLM.
        answer: The correct answer to the question ("1" or "0").

    Returns:
        bool: True if the response matches the answer, False otherwise.
    """
    if isinstance(answer, int):
        answer = str(answer)
    return response == answer


def main() -> None:
    """Main function for loading and processing the MJ Bench dataset."""
    df = load_mj_bench_dataset(
        "/Users/tyrionhuu/projects/research_projects/Multi-LLM-Debate/datasets/MJ-Bench"
    )
    processed_df = preprocess_mj_bench_dataframe(df)
    print(processed_df.head())


if __name__ == "__main__":
    main()
