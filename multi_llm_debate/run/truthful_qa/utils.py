import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

from datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)
DATASET_PATH = "datasets/TruthfulQA"
DATASET_NAME = "domenicrosati/TruthfulQA"
RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def load_truthful_qa_dataset(
    dataset_path: Union[str, Path] = DATASET_PATH,
    sample_size: Optional[int] = None,
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

    if dataset_path.exists():
        try:
            # Try to load the dataset from disk
            dataset = load_from_disk(str(dataset_path))
            logger.info(f"Loaded dataset from disk: {dataset_path}")
            df = pd.DataFrame(dataset)
        except Exception as e:
            logger.error(f"Failed to load dataset from disk: {e}")
            df = None

    if df is None:
        # If loading from disk failed, download the dataset from Hugging Face
        logger.info(
            f"Dataset path {dataset_path} does not exist. Downloading from Hugging Face."
        )
        dataset = load_dataset(
            DATASET_NAME,
            cache_dir=str(dataset_path),
            split="train",
        )
        if dataset is None:  # pragma: no cover
            raise ValueError(
                f"Failed to load the TruthfulQA dataset from Hugging Face: {DATASET_NAME}"
            )
        df = pd.DataFrame(dataset)

    df = df.copy()
    # Shuffle the DataFrame by RANDOM_STATE
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Add an ID column
    df["id"] = range(len(df))

    # If the DataFrame has a column of dictionaries, expand it
    first_col = df.columns[0]
    if df[first_col].apply(lambda x: isinstance(x, dict)).all():
        meta_df = pd.json_normalize(df[first_col])
        df = pd.concat([meta_df, df.drop(columns=[first_col])], axis=1)

    # print(df.head())
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

    processed_df = df[
        ["id", "Question", "response_A", "response_B", "response_C", "answer"]
    ].rename(columns={"Question": "question"})

    if sample_size is not None:
        if sample_size > len(processed_df):
            logger.warning(
                f"Sample size {sample_size} exceeds dataset size {len(processed_df)}. Using full dataset."
            )
            sample_size = len(processed_df)

        processed_df = processed_df.head(sample_size)

    logger.info(f"Processed TruthfulQA dataset with {len(processed_df)} samples.")
    return processed_df


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
    df = load_truthful_qa_dataset(sample_size=800)
    print(df.info())
    print(df.head())


if __name__ == "__main__":
    main()
