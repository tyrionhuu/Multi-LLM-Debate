import json
import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

LLM_BAR_DATASET_FILES: list[str] = [
    "LLMBar/Natural/dataset.json",
    "LLMBar/Adversarial/GPTInst/dataset.json",
    "LLMBar/Adversarial/GPTOut/dataset.json",
    "LLMBar/Adversarial/Manual/dataset.json",
    "LLMBar/Adversarial/Neighbor/dataset.json",
    "CaseStudy/Base_9/dataset.json",
    "CaseStudy/Base_10/dataset.json",
    "CaseStudy/Constraint/dataset.json",
    "CaseStudy/Negation/dataset.json",
    "CaseStudy/Normal/dataset.json",
    "Processed/FairEval/dataset.json",
    "Processed/LLMEval^2/dataset.json",
    "Processed/MT-Bench/dataset.json",
]
DATASET_PATH = "datasets/LLMBar"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def load_llm_bar_dataset(
    dataset_path: Union[str, Path] = DATASET_PATH,
    base_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load and preprocess the LLMBar dataset.

    Args:
        dataset_path: Path to the dataset directory. If it exists locally,
            it will be loaded from disk; otherwise, it will be downloaded.
        sample_size: Optional; if provided, the DataFrame will be sampled to this size.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with required columns.
    """
    dataset_path = Path(dataset_path)
    if base_path is not None:
        dataset_path = Path(base_path) / dataset_path
    dataset_path = dataset_path.resolve()
    logger.info(f"Dataset path: {dataset_path}")
    if not dataset_path.exists():
        logger.error(f"Dataset path {dataset_path} does not exist.")
        return pd.DataFrame()

    json_data = []

    for rel_path in LLM_BAR_DATASET_FILES:
        with open(dataset_path / rel_path, "r", encoding="utf-8") as f:
            json_data.extend(json.load(f))

    df = pd.DataFrame(json_data)

    # Preprocess: add id column and rename columns
    df = df.copy()
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df["id"] = range(len(df))
    column_mapping = {
        "input": "question",
        "output_1": "response_1",
        "output_2": "response_2",
        "label": "answer",
    }
    df = df.rename(columns=column_mapping)

    if sample_size is not None:
        if sample_size > len(df):
            logger.warning(
                f"Sample size {sample_size} is larger than dataset size {len(df)}. Using full dataset."
            )
            sample_size = len(df)

        df = df.head(sample_size)

    logger.info(f"Loaded LLMBar dataset with {len(df)} samples from {dataset_path}.")
    return df


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


def compare_llm_bar_response(
    response: Literal["1", "2"],
    answer: Union[str, int],
) -> bool:
    """Compare the responses from the LLMBar dataset.

    Args:
        response: The response string from the LLM.
        answer: The correct answer to the question ("1" or "2").

    Returns:
        bool: True if the response matches the answer, False otherwise.
    """
    try:
        return response == str(answer)
    except AttributeError:
        return False


def main():
    dataset = load_llm_bar_dataset()
    print(dataset.iloc[0].to_dict())


if __name__ == "__main__":
    main()
