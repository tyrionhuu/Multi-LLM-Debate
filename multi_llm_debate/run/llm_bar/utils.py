import json
import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_llm_bar_dataframe(
    dataframe: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Preprocess the LLMBar DataFrame to ensure it has all required columns.

    Args:
        dataframe: Input DataFrame from LLMBar dataset.
        random_state: Random seed for shuffling. If None, the dataset will be
            randomized differently each time.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with required columns.
    """
    # Create a copy to avoid modifying the original dataframe
    processed_df = dataframe.copy()

    # Add an id column
    processed_df["id"] = range(len(processed_df))

    # Rename columns
    column_mapping = {
        "output_1": "response_1",
        "output_2": "response_2",
        "label": "answer",
    }

    processed_df = processed_df.rename(columns=column_mapping)

    # If random_state is provided, shuffle the dataframe
    if random_state is not None:
        processed_df = processed_df.sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)

    return processed_df


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

    if not dataset_path.exists():
        logger.error(f"Dataset path {dataset_path} does not exist.")
        return pd.DataFrame()

    json_data = []

    with open(dataset_path / "LLMBar/Natural/dataset.json", "r", encoding="utf-8") as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "LLMBar/Adversarial/GPTInst/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "LLMBar/Adversarial/GPTOut/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "LLMBar/Adversarial/Manual/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "LLMBar/Adversarial/Neighbor/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "CaseStudy/Base_9/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "CaseStudy/Base_10/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "CaseStudy/Constraint/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "CaseStudy/Negation/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "CaseStudy/Normal/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "Processed/FairEval/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "Processed/LLMEval^2/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))
    with open(
        dataset_path / "Processed/MT-Bench/dataset.json", "r", encoding="utf-8"
    ) as f:
        json_data.extend(json.load(f))

    if random_state is not None:
        random.seed(random_state)
        random.shuffle(json_data)

    df = pd.DataFrame(json_data)
    return df.reset_index(drop=True)


def extract_1_2_answer(
    response: str,
) -> Literal["1", "2"]:
    """Extract the answer from the response string.

    Args:
        response: The response string from the LLM.

    Returns:
        Literal["1", "2"]: Answer "1" or "2".
    """
    # Try to find "Final Answer: X" pattern
    match = re.search(r"Final Answer:\s*([12])", response)
    if match:
        return match.group(1)
    else:
        raise ValueError(
            "No valid answer found in the response. Please ensure the response contains 'Final Answer: 1' or 'Final Answer: 2'."
        )


def main():
    dataset = load_llm_bar_dataset()
    processed_df = preprocess_llm_bar_dataframe(dataset)
    print(processed_df.head())


if __name__ == "__main__":
    main()
