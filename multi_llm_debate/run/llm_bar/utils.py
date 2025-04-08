import json
import logging
import random
from pathlib import Path
from typing import Optional, Union

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
    # Ensure required columns exist
    required_columns = ["id", "input", "output", "category"]
    for col in required_columns:
        if col not in dataframe.columns:
            logger.error(f"Missing required column: {col}")
            return pd.DataFrame()

    # Shuffle the DataFrame if random_state is provided
    if random_state is not None:
        dataframe = dataframe.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return dataframe

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


def main():
    dataset = load_llm_bar_dataset()
    print(dataset.info())


if __name__ == "__main__":
    main()
