import json
import logging
import random
import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

JSON_PATH = "datasets/ICE-Score/conala_grade.json"
RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def load_ice_score_dataset(
    json_path: Union[str, Path] = JSON_PATH,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert a JSON file to a DataFrame with an added 'id' column.

    Args:
        json_path (Union[str, Path]): Path to the JSON file.
        sample_size (Optional[int]): If provided, the DataFrame will be sampled to this size.

    Returns:
        pd.DataFrame: DataFrame containing the data from the JSON file,
            with columns 'id', 'question', 'response', and 'answer'.
    """
    json_path = Path(json_path)

    if not json_path.is_file():
        raise FileNotFoundError(f"File not found: {json_path}")
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        # Compute mean of grade-snippet values for each item
        answers = [
            sum(item["grade-snippet"].values()) / len(item["grade-snippet"])
            for item in data
        ]
        df = pd.DataFrame(data)
        df = df.rename(columns={"intent": "question", "snippet": "response"})
        df = df[["question", "response"]]
        df["answer"] = answers

        df = df.copy()
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        df["id"] = range(len(df))

        if sample_size is not None:
            if sample_size > len(df):
                logger.warning(
                    f"Sample size {sample_size} is larger than the dataset size {len(df)}. "
                    "Using the full dataset instead."
                )
            df = df.head(sample_size)

        logger.info(
            f"Loaded ICE-Score dataset with {len(df)} samples from {json_path}."
        )
        return df

    except ValueError as e:
        raise ValueError(f"Error reading JSON file {json_path}: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while processing {json_path}: {e}")


def extract_0_4_answer(
    response: str,
) -> str:
    """Extract the answer from the LLM response for values 0-4.

    Args:
        response (str): The response string from the LLM.

    Returns:
        str: The extracted answer as a string.
    """
    match = re.search(r"Final Answer:\s*([0-9]+(?:\.[0-9]+)?)", response)
    if match:
        answer = match.group(1)
        try:
            float_answer = float(answer)
            if 0 <= float_answer <= 4:
                return str(float_answer)
            else:
                raise ValueError(
                    "Extracted answer is out of range. Please ensure the "
                    "response contains a valid number between 0 and 4."
                )
        except ValueError:
            raise ValueError(
                "Extracted answer is not a valid float. Please ensure the "
                "response contains a valid number between 0 and 4."
            )
    raise ValueError(
        "No valid answer found in the response. Please ensure the response "
        "contains 'Final Answer: X' where X is a number between 0 and 4."
    )


def compare_ice_score_response(
    response: Union[str, float],
    answer: Union[str, float],
) -> bool:
    """Compare the responses from the ICE-Score dataset for values 0-4.

    Args:
        response (float): The response value extracted from the LLM.
        answer (float): The correct answer to the question.

    Returns:
        bool: True if the response matches the answer (within tolerance),
            False otherwise.
    """
    return abs(float(response) - float(answer)) < 1.5


if __name__ == "__main__":
    # Example usage
    dataframe = load_ice_score_dataset()
    print(dataframe.describe())
