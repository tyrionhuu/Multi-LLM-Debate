import json
import logging
import random
import re
from pathlib import Path
from typing import List, Literal, Optional, Union

import pandas as pd

INPUT_PATHS = [
    "datasets/COMP-Analysis/turn_level_texts/conture-turn_text.txt",
    "datasets/COMP-Analysis/turn_level_texts/fed-turn_text.txt",
    "datasets/COMP-Analysis/turn_level_texts/dailydialog-zhao_text.txt",
    "datasets/COMP-Analysis/turn_level_texts/persona-zhao_text.txt",
    "datasets/COMP-Analysis/turn_level_texts/topical-usr_text.txt",
    "datasets/COMP-Analysis/turn_level_texts/persona-usr_text.txt",
]
ANSWER_PATHS = "datasets/COMP-Analysis/turn_level_texts/turn_overall_ratings.json"
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def load_comp_analysis_dataset(
    input_paths: Union[str, Path, List[Union[str, Path]]] = INPUT_PATHS,
    answer_path: Union[str, Path] = ANSWER_PATHS,
    sample_size: Optional[int] = None,
    template: str = "{}\t{}",
) -> pd.DataFrame:
    """
    Load multiple text files into a DataFrame, processing each line by extracting
    the last two tab-separated fields and formatting them. Optionally, add
    correctness columns for each answer file.

    Args:
        input_paths (Union[str, Path, List[Union[str, Path]]]): Path(s) to the text files.
        answer_path (Union[str, Path]): Path to the ground truth JSON file.
        template (str): Template string to format the extracted fields.
        sample_size (Optional[int]): If provided, the DataFrame will be sampled to this size.

    Returns:
        pd.DataFrame: DataFrame containing the processed data from the text files,
            with columns 'id', 'input', 'response', and optionally correctness columns.
    """
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]

    data = []
    for path in input_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    fields = line.strip().split("\t")
                    if len(fields) < 2:
                        continue  # skip lines that don't have at least two fields
                    formatted = template.format(fields[-2], fields[-1])
                    data.append(formatted)

    df = pd.DataFrame(data, columns=["input_response"])
    df[["input", "response"]] = df["input_response"].str.split("\t", n=1, expand=True)
    df = df.drop(columns=["input_response"])

    # Load ground truth
    ground_truth = []
    with open(answer_path, "r", encoding="utf-8") as f:
        file = json.load(f)
        for key in [
            "conture-turn",
            "fed-turn",
            "dailydialog-zhao",
            "persona-zhao",
            "topical-usr",
            "persona-usr",
        ]:
            ground_truth.extend(file[key])

    # Add ground truth column
    df["answer"] = ground_truth[: len(df)]

    df = df.copy()
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df["id"] = range(len(df))  # Add an ID column

    if sample_size is not None:
        if sample_size > len(df):
            logger.warning(
                f"Sample size {sample_size} is larger than the dataset size {len(df)}. "
                "Using the full dataset instead."
            )
        df = df.head(sample_size)

    logger.info(f"Loaded {len(df)} entries from input files and ground truth.")
    return df


def extract_1_to_5_answer(
    response: str,
) -> Literal["1", "2", "3", "4", "5"]:
    """Extract the answer from the response string.

    Args:
        response (str): The response string from the LLM.

    Returns:
        Literal["1", "2", "3", "4", "5"]: Answer between 1 and 5.

    Raises:
        ValueError: If no valid answer is found in the response.
    """
    match = re.search(r"Final Answer:\s*([1-5])", response)
    if match:
        return match.group(1)
    raise ValueError(
        f"Invalid response format. Expected 'Final Answer: x' where x is between 1 and 5, got: {response}"
    )


def compare_comp_analysis_response(
    response: Union[str, int], answer: Union[str, int, float], threshold: float = 1.5
) -> bool:
    """Compare the response with the answer.

    Args:
        response (Union[str, int]): The response from the LLM.
        answer (Union[str, int, float]): The ground truth answer.
        threshold (float): The threshold for comparison.

    Returns:
        bool: True if the response is within the threshold of the answer, False otherwise.
    """
    try:
        response_value = float(response)
        answer_value = float(answer)
        return abs(response_value - answer_value) <= threshold
    except ValueError:
        raise ValueError(f"Invalid response or answer format: {response}, {answer}")


if __name__ == "__main__":
    df = load_comp_analysis_dataset(sample_size=10)
    print(df.head())
