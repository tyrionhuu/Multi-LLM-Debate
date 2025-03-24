from typing import Dict, Literal, Union

import pandas as pd

Answer = Literal["true", "false"]
Reasoning = Union[str, Dict[str, str]]


def process_bool_q_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Process the BoolQ DataFrame to ensure it has all required columns.

    Args:
        dataframe: Input DataFrame from BoolQ dataset

    Returns:
        pd.DataFrame: Processed DataFrame with all required columns
    """
    # Create a copy to avoid modifying the original
    processed_df = dataframe.copy()

    # Generate ID from index if 'id' column doesn't exist
    if "id" not in processed_df.columns:
        processed_df["id"] = processed_df.index + 1

    return processed_df


def extract_bool_answer(response: str) -> Answer:
    """
    Extract true or false answer from the response string, using the last occurrence.

    Args:
        response: The response string from the LLM.

    Returns:
        Answer: "true" or "false". Uses the last occurrence of true/false/yes/no.

    Raises:
        ValueError: If no recognizable answer is found.
    """
    response = response.lower()
    last_true = max(response.rfind("true"), response.rfind("yes"))
    last_false = max(response.rfind("false"), response.rfind("no"))

    if last_true == -1 and last_false == -1:
        raise ValueError("Answer not recognized")

    return "true" if last_true > last_false else "false"
