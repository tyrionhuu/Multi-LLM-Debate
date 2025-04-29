import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

JSON_PATH = "datasets/PRM800K/data/phase2_test.jsonl"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def load_prm800k_dataset(
    json_path: Union[str, Path] = JSON_PATH,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load and preprocess the PRM800K dataset from a JSONL file.

    Args:
        json_path: Path to the JSONL file.
        sample_size: Optional; if provided, the DataFrame will be sampled to this size.

    Returns:
        pd.DataFrame: DataFrame with columns ['question', 'answer', 'steps'].
    """
    json_path = Path(json_path)

    if not json_path.is_file():
        raise FileNotFoundError(f"File not found: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        df = pd.DataFrame(data)
    except ValueError as e:
        raise ValueError(f"Error reading JSONL file {json_path}: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while processing {json_path}: {e}")

    def extract_answer_and_steps(label: Dict) -> Tuple[List, List]:
        """Extracts the answer list and used_texts from the label dict."""
        answers = []
        used_texts = []
        for step in label.get("steps", []):
            used_text = None
            used_rating = None
            for text in step.get("completions", []):
                if text.get("rating") is not None:
                    used_text = text["text"]
                    used_rating = text["rating"]
                    break
            if used_text is None and step.get("completions"):
                used_text = step["completions"][-1]["text"]
                used_rating = step["completions"][-1].get("rating")
            used_texts.append(used_text)
            if used_rating is None:
                answers.append(None)
            elif used_rating > -1:
                answers.append(1)
            else:
                answers.append(0)
        return answers, used_texts

    processed = []
    for idx, (_, row) in enumerate(df.iterrows()):
        question = row["question"]["problem"]
        answer, steps = extract_answer_and_steps(row["label"])
        processed.append(
            {"id": idx, "question": question, "answer": answer, "steps": steps}
        )

    df = pd.DataFrame(processed)
    df = df.copy()
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    df["id"] = range(len(df))  # Add an ID column

    if sample_size is not None:
        if sample_size > len(df):
            logger.warning(
                f"Sample size {sample_size} exceeds dataset size {len(df)}. Using full dataset."
            )
            sample_size = len(df)
        df = df.head(sample_size)
        
    logger.info(f"Loaded PRM800K dataset with {len(df)} samples.")
    return df


def extract_int_list(
    response: str,
) -> List[int]:
    """Extract a list of answers from the response string in the format
    'Final Answer: [x,y,...]', where x,y,... are 1 or -1.

    Args:
        response (str): The response string from the LLM.

    Returns:
        List[int]: List of answers (each must be 1 or -1).

    Raises:
        ValueError: If no valid answer list is found in the response.
    """
    match = re.search(r"Final Answer:\s*\[([-\d\s,]+)\]", response)
    if match:
        numbers = [int(x.strip()) for x in match.group(1).split(",")]
        if all(x in (1, -1) for x in numbers):
            return numbers
        raise ValueError(f"Answer list must only contain 1 or -1, got: {numbers}")
    raise ValueError(
        "Invalid response format. Expected 'Final Answer: [x,y,...]' "
        "where x,y,... are 1 or -1, got: {}".format(response)
    )


def compare_prm800k_response(
    response: Union[str, List[int]],
    answer: Union[str, List[int]],
) -> bool:
    """Compare the response with the answer.

    Args:
        response (Union[str, List[int]]): The response from the LLM.
        answer (Union[str, List[int]]): The ground truth answer.

    Returns:
        bool: True if the response matches the answer, False otherwise.
    """
    if isinstance(response, str):
        response = extract_int_list(response)
    if isinstance(answer, str):
        answer = json.loads(answer)  # Assuming answer is a JSON string

    return response == answer


if __name__ == "__main__":
    # Example usage
    df = load_prm800k_dataset(sample_size=10)
    print(df.head())
