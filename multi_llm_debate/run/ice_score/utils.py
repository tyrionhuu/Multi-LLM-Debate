import json
import re
from pathlib import Path
from typing import Union

import pandas as pd


def load_ice_score_dataset(json_path: Union[str, Path]) -> pd.DataFrame:
    """
    Convert a JSON file to a DataFrame with an added 'id' column.

    Args:
        json_path (Union[str, Path]): Path to the JSON file.

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
        df.insert(0, "id", range(len(df)))
        df["answer"] = answers
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
    json_data = Path("datasets/ICE-Score/conala_grade.json")
    dataframe = load_ice_score_dataset(json_data)
    print(dataframe.describe())