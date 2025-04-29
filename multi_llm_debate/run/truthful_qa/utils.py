import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

from multi_llm_debate.utils.download_dataset import load_save_huggingface_dataset_df

logger = logging.getLogger(__name__)
DATASET_PATH = "datasets/TruthfulQA"
RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def load_truthful_qa_dataset(
    dataset_path: Union[str, Path] = DATASET_PATH,
    dataset_name: str = "domenicrosati/TruthfulQA",
) -> pd.DataFrame:
    """Load and preprocess the TruthfulQA dataset.

    Args:
        dataset_path: Path to the dataset directory. If it exists locally,
            it will be loaded from disk; otherwise, it will be downloaded.
        dataset_name: Name of the dataset to load from Hugging Face.

    Returns:
        pd.DataFrame: Processed DataFrame with all required columns and
            multiple-choice format.
    """
    # Initialize empty DataFrame
    df = None
    dataset_path = Path(dataset_path)
    try:
        df = load_save_huggingface_dataset_df(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            force_download=False,
            split="train",
        )
        logger.info(
            f"Loaded TruthfulQA dataset from Hugging Face. "
            f"Type: {type(df)}, Shape: {getattr(df, 'shape', None)}"
        )
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face: {e}")
        raise e

    if df is None:
        logger.error(
            f"load_save_huggingface_dataset_df returned None for dataset_name={dataset_name}, "
            f"dataset_path={dataset_path}"
        )
        raise ValueError("Input DataFrame is None. Please load the dataset first.")

    if isinstance(df, pd.DataFrame) and df.empty:
        logger.error(
            f"Loaded DataFrame is empty for dataset_name={dataset_name}, "
            f"dataset_path={dataset_path}"
        )
        raise ValueError("Loaded DataFrame is empty. Please check the dataset source.")

    df = df.copy()
    # Shuffle the DataFrame by RANDOM_STATE
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Create ID column if missing
    if "id" not in df.columns:
        df["id"] = df.index + 1

    # Helper to choose random answers
    def _random_answer(answers_str: str) -> Optional[str]:
        answers = [a.strip() for a in answers_str.split(";") if a.strip()]
        return random.choice(answers) if answers else None

    # Select correct and incorrect answers
    df["correct_ans"] = df["Correct Answers"].apply(_random_answer)

    # Select two distinct incorrect answers
    def _two_wrong_answers(answers_str: str) -> tuple[Optional[str], Optional[str]]:
        answers = [a.strip() for a in answers_str.split(";") if a.strip()]
        if len(answers) < 2:
            return (answers[0], answers[0]) if answers else (None, None)
        a, b = random.sample(answers, k=2)
        return (a, b) if a != b else (a, random.choice([x for x in answers if x != a]))

    incorrect_pairs = df["Incorrect Answers"].apply(_two_wrong_answers)
    df[["wrong1", "wrong2"]] = pd.DataFrame(incorrect_pairs.tolist(), index=df.index)

    # Assign options A/B/C randomly
    df["answer"] = random.choices(["A", "B", "C"], k=len(df))

    # Map answers to options
    def _map_options(
        row: pd.Series,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        opts = {"A": None, "B": None, "C": None}
        opts[row["answer"]] = row["correct_ans"]
        others = [o for o in opts if o != row["answer"]]
        opts[others[0]], opts[others[1]] = row["wrong1"], row["wrong2"]
        return opts["A"], opts["B"], opts["C"]

    # Use pd.DataFrame(...) to expand the tuple into columns
    df[["response_A", "response_B", "response_C"]] = pd.DataFrame(
        df.apply(_map_options, axis=1).tolist(), index=df.index
    )

    return df[
        ["id", "Question", "response_A", "response_B", "response_C", "answer"]
    ].rename(columns={"Question": "question"})


def extract_caption_a_b_c_answer(response: str) -> Literal["A", "B", "C"]:
    """
    Extract answer from the response string.

    First tries to find "Final Answer: A", "Final Answer: B", or "Final Answer: C" pattern.
    If not found, falls back to finding the last occurrence of A, B, or C.

    Args:
        response: The response string from the LLM.

    Returns:
        Literal["A", "B", "C"]: Answer "A", "B", or "C".
    """
    # Try to find "Final Answer: X" pattern
    match = re.search(r"Final Answer:\s*([ABC])", response)
    if match:
        return match.group(1)
    else:
        raise ValueError(
            "No valid answer found in the response. Please ensure the response contains 'Final Answer: A', 'Final Answer: B', or 'Final Answer: C'."
        )


def compare_truthful_qa_response(
    response: Literal["A", "B", "C"],
    answer: str,
) -> bool:
    """Compare the responses from the judge bench.

    Args:
        response: The response string from the LLM.
        answer: The correct answer to the question ("A"/"B"/"C").

    Returns:
        bool: True if the response matches the answer, False otherwise.
    """
    try:
        return response.upper() == answer.upper()
    except AttributeError:
        return False


def main() -> None:
    """Main function for loading and processing the TruthfulQA dataset."""
    df = load_truthful_qa_dataset()
    print(df.head())


if __name__ == "__main__":
    main()
