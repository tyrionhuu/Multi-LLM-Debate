import json
import re
from pathlib import Path
from typing import Literal, Union

import pandas as pd

def load_big_bench_dataset(json_path: Union[str, Path]) -> pd.DataFrame:
    """
    Convert a JSON file to a DataFrame with an added 'id' column.

    Args:
        json_path (Union[str, Path]): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the JSON file,
            with columns 'id', 'input', and 'answer'.
    """
    json_path = Path(json_path)

    if not json_path.is_file():
        raise FileNotFoundError(f"File not found: {json_path}")
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)["examples"]
        df = pd.DataFrame(data)
        
        df["answer"] = df["target_scores"].apply(
            lambda x: 1 if x["plausible"] == 1 else 0
        )
        
        # Select and reorder columns
        df = df[["input", "answer"]]
        df.insert(0, "id", range(len(df)))
        return df
    except ValueError as e:
        raise ValueError(f"Error reading JSON file {json_path}: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while processing {json_path}: {e}")
    
df = load_big_bench_dataset("datasets/BIG-Bench/sports_understanding/task.json")
print(df.head())