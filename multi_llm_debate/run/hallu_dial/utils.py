import json
import re
from pathlib import Path
from typing import Literal, Union, Optional
import logging
import random
import pandas as pd
logger = logging.getLogger(__name__)

JSON_PATH = "datasets/HalluDial/all_train.json"
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

def load_hallu_dial_dataset(
    json_path: Union[str, Path] = JSON_PATH,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert a JSON file to a DataFrame with added 'id' and 'answer' columns.
    Filters out entries without valid yes/no answers.

    Args:
        json_path (Union[str, Path]): Path to the JSON file.
        sample_size (Optional[int]): If provided, the DataFrame will be sampled to this size.
        
    Returns:
        pd.DataFrame: DataFrame containing the data from the JSON file,
            with columns 'id', 'knowledge', 'dialogue', 'response',
            and 'answer'. The 'answer' column is derived from the 'target'
            field using the str_to_bool function. Entries without valid yes/no
            answers are filtered out.
    """
    json_path = Path(json_path)

    if not json_path.is_file():
        raise FileNotFoundError(f"File not found: {json_path}")
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        # Apply str_to_bool to the 'target' column to create the 'answer' column
        df["answer"] = df["target"].apply(str_to_bool)
        # Filter out rows with None in the 'answer' column
        df = df.dropna(subset=["answer"])
        # Rename 'dialogue_history' to 'dialogue'
        df = df.rename(columns={"dialogue_history": "dialogue"})
        # Select and reorder columns
        df = df[["knowledge", "dialogue", "response", "answer"]]
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
        logger.info(f"Loaded {len(df)} entries from {json_path}")
        return df
    except ValueError as e:
        raise ValueError(f"Error reading JSON file {json_path}: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while processing {json_path}: {e}")


def str_to_bool(input: str) -> Union[Literal["0", "1"], None]:
    """Convert a string to a boolean value.

    Args:
        input (str): The input string to convert.

    Returns:
        Union[Literal["0", "1"], None]: "0" if the input starts with "no",
            "1" if input starts with "yes", None otherwise.
    """
    input = input.lower()
    if input.startswith("no"):
        return "0"
    elif input.startswith("yes"):
        return "1"
    else:
        return None


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
