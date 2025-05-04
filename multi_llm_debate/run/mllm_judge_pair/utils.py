import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

MLLM_JUDGE_PAIR_DATASET_FILE = "datasets/MLLM-Judge/pair_data.tsv"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def parse_question_field(text: str) -> tuple[str, str, str]:
    """Parse the question field to extract question and responses.

    Args:
        text: The content of the question field in format
            "question Assistant A:... Assistant B:..."

    Returns:
        Tuple containing (question, response_a, response_b)
    """
    # Extract question (everything before "Assistant A:")
    question_match = re.search(r"^(.*?)Assistant A:", text, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""

    # Extract Assistant A's response (between "Assistant A:" and "Assistant B:")
    response_a_match = re.search(r"Assistant A:(.*?)Assistant B:", text, re.DOTALL)
    response_a = response_a_match.group(1).strip() if response_a_match else ""

    # Extract Assistant B's response (everything after "Assistant B:")
    response_b_match = re.search(r"Assistant B:(.*?)$", text, re.DOTALL)
    response_b = response_b_match.group(1).strip() if response_b_match else ""

    return question, response_a, response_b


def load_mllm_judge_pairs(
    file_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None,
    parse_question: bool = True,
) -> pd.DataFrame:
    """Load MLLM-Judge pair dataset from TSV file.

    Args:
        file_path: Path to the TSV file. If None, uses the default path.
        sample_size: Optional number of samples to return from the dataset.
        parse_question: Whether to parse the question field into separate columns
            for question, response_a, and response_b.

    Returns:
        DataFrame containing the pair data with columns: id, image, question,
        answer, and if parse_question is True, also includes original_question,
        response_a, and response_b.

    Raises:
        FileNotFoundError: If the specified file doesn't exist.
    """
    if file_path is None:
        file_path = MLLM_JUDGE_PAIR_DATASET_FILE

    logger.info(f"Loading MLLM-Judge pair data from {file_path}")

    try:
        # Read TSV file
        df = pd.read_csv(file_path, sep="\t")

        # Validate expected columns
        expected_columns = ["id", "image", "question", "answer"]
        if not all(col in df.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in df.columns]
            logger.warning(f"Missing expected columns: {missing}")

        # Handle image column which contains byte data
        if "image" in df.columns:
            # If needed, convert string representation of bytes to actual bytes
            if isinstance(df["image"].iloc[0], str):
                # This handles cases where bytes might be represented as strings
                df["image"] = df["image"].apply(
                    lambda x: x.encode() if isinstance(x, str) else x
                )

        # Parse question field if requested
        if parse_question and "question" in df.columns:
            logger.info("Parsing question field into separate columns...")
            # Save original question
            df["original_question"] = df["question"]

            # Apply parsing function
            parsed_results = df["original_question"].apply(parse_question_field)

            # Create new columns
            df["question"] = parsed_results.apply(lambda x: x[0])
            df["response_a"] = parsed_results.apply(lambda x: x[1])
            df["response_b"] = parsed_results.apply(lambda x: x[2])

            logger.info("Question field parsed successfully")

        logger.info(f"Loaded {len(df)} MLLM-Judge pair examples")

        df = df.copy()
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        df["id"] = range(len(df))

        if sample_size is not None:
            if sample_size > len(df):
                logger.warning(
                    f"Requested sample size {sample_size} exceeds dataset size. Returning full dataset."
                )
                sample_size = len(df)
            df = df.head(sample_size)

        logger.info(f"Sampled {len(df)} MLLM-Judge pair examples")

        return df

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
        raise
    except Exception as e:
        logger.exception(f"Error loading dataset: {e}")
        raise


def extract_caption_a_b_answer(response: str) -> Literal["A", "B"]:
    """
    Extract answer from the response string.

    First tries to find "Final Answer: A" or "Final Answer: B" pattern.
    If not found, falls back to finding the last occurrence of A or B.

    Args:
        response: The response string from the LLM.

    Returns:
        Literal["A", "B"]: Answer "A" or "B".
    """
    # Try to find "Final Answer: X" pattern
    match = re.search(r"Final Answer:\s*([AB])", response)
    if match:
        return match.group(1)
    else:
        raise ValueError(
            "No valid answer found in the response. Please ensure the response contains 'Final Answer: A' or 'Final Answer: B'."
        )


def compare_mllm_judge_pairs_response(
    response: Literal["A", "B"],
    answer: str,
) -> bool:
    """Compare the responses from the MLLM-Judge dataset.

    Args:
        response: The response string from the LLM.
        answer: The correct answer to the question ("A"/"B").

    Returns:
        bool: True if the response matches the answer, False otherwise.
    """
    try:
        return response == answer.upper()
    except Exception as e:
        logger.error(f"Error comparing responses: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    df = load_mllm_judge_pairs(sample_size=10)
    print(df["response_a"].head(10))
