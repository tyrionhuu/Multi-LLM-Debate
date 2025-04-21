import json
import re
from pathlib import Path
from typing import Literal, Union

import pandas as pd


def load_hallu_dial_dataset(json_path: Union[str, Path]) -> pd.DataFrame:
    """
    Convert a JSON file to a DataFrame with an added 'id' column.

    Args:
        json_path (Union[str, Path]): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the JSON file,
            with columns 'id', 'knowledge', 'dialogue_history', and 'response'.
    """
    json_path = Path(json_path)

    if not json_path.is_file():
        raise FileNotFoundError(f"File not found: {json_path}")
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        df = df[["knowledge", "dialogue_history", "response"]]
        df.insert(0, "id", range(len(df)))
        return df
    except ValueError as e:
        raise ValueError(f"Error reading JSON file {json_path}: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while processing {json_path}: {e}")


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


def compare_hallu_dial_response(
    response: Literal["1", "0"],
    answer: Union[str, int],
) -> bool:
    """Compare the responses from the ICE-Score dataset.

    Args:
        response: The response string from the LLM.
        answer: The correct answer to the question ("1" or "0").

    Returns:
        bool: True if the response matches the answer, False otherwise.
    """
    if isinstance(answer, int):
        answer = str(answer)
    return response == answer
