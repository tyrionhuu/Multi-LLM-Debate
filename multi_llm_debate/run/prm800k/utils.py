import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


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


def preprocess_prm800k_dataset(
    dataframe: pd.DataFrame,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Preprocess the PRM800K DataFrame to extract 'question' and 'answer'.

    Args:
        dataframe: Input DataFrame from PRM800K dataset.
        random_state: Random seed for shuffling. If None, the dataset will be
            randomized differently each time.

    Returns:
        pd.DataFrame: DataFrame with columns ['question', 'answer'].
    """

    def extract_answer(label: Dict) -> List:
        """Extracts the answer list from the label dict."""
        answers = []
        for step in label.get("steps", []):
            used_rating = None
            for text in step.get("completions", []):
                if text.get("rating") is not None:
                    used_rating = text["rating"]
                    break
            if used_rating is None and step.get("completions"):
                used_rating = step["completions"][-1].get("rating")
            if used_rating is None:
                answers.append(None)
            elif used_rating > -1:
                answers.append(1)
            else:
                answers.append(0)
        return answers

    processed = []
    for _, row in dataframe.iterrows():
        question = row["question"]["problem"]
        answer = extract_answer(row["label"])
        processed.append({"question": question, "answer": answer})

    df = pd.DataFrame(processed)

    if random_state is not None:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Example usage
    df = load_prm800k_dataset()
    processed_df = preprocess_prm800k_dataset(df, random_state=42)
    print(processed_df.head())
