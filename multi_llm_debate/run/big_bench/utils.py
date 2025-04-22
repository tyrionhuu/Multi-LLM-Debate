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


def compare_big_bench_response(
    response: Literal["0", "1"],
    answer: Union[str, int],
) -> bool:
    """Compare the response from the LLM with the expected answer.

    Args:
        response (Literal["0", "1"]): The response from the LLM.
        answer (Union[str, int]): The expected answer.

    Returns:
        bool: True if the response matches the expected answer, False otherwise.
    """
    if isinstance(answer, int):
        answer = str(answer)
    return response == answer
