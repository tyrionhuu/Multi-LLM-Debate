import random
import re
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
import json

from multi_llm_debate.utils.download_dataset import load_save_huggingface_dataset_df

def load_prm800k_dataset(
    json_path: Union[str, Path] = "datasets/PRM800K/data/phase2_test.jsonl",
) -> pd.DataFrame:
    """Load the PRM800K dataset from a JSONL file.

    Args:
        json_path: Path to the JSONL file.

    Returns:
        pd.DataFrame: DataFrame containing the PRM800K data.
    """
    json_path = Path(json_path)

    if not json_path.is_file():
        raise FileNotFoundError(f"File not found: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        df = pd.DataFrame(data)
        return df
    except ValueError as e:
        raise ValueError(f"Error reading JSONL file {json_path}: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while processing {json_path}: {e}")
    
if __name__ == "__main__":
    # Example usage
    df = load_prm800k_dataset()
    print(df.describe())