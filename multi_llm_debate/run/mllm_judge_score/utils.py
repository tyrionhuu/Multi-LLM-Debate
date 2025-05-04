import base64
import logging
import random
import re
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

MLLM_JUDGE_SCORE_DATASET_FILE = "datasets/MLLM-Judge/score_data.tsv"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)


def parse_question_field(text: str) -> Tuple[str, str]:
    """
    Parse a question field where the question appears before 'Assistant:'

    Args:
        text: The input text to parse.

    Returns:
        A tuple containing the extracted question text, the assistant's response
    """
    # Split by "Assistant:" to get the question part
    parts = text.split("Assistant:")

    if len(parts) < 2:
        logger.warning(f"Could not find 'Assistant:' in text: {text}")
        return text, ""

    question = parts[0].strip()
    assistant_response = parts[1].strip()

    return question, assistant_response


def load_mllm_judge_score_dataset(
    file_path: Optional[Union[str, Path]] = MLLM_JUDGE_SCORE_DATASET_FILE,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load MLLM-Judge score dataset from a TSV file.

    Args:
        file_path: Path to the TSV file. If None, uses the default path.
        sample_size: Optional number of samples to return from the dataset.

    Returns:
        DataFrame containing the score data with columns: id, image, question,
        answer, and if parse_question is True, also includes original_question,
        response_A, and response_B.
    """
    if file_path is None:
        file_path = MLLM_JUDGE_SCORE_DATASET_FILE
    else:
        file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Read TSV file
        df = pd.read_csv(file_path, sep="\t")

        # Validate expected columns
        expected_columns = ["id", "image", "question", "answer"]
        if not all(col in df.columns for col in expected_columns):
            missing = [col for col in expected_columns if col not in df.columns]
            logger.warning(f"Missing expected columns: {missing}")

        # Parse the question field
        df["question"], df["response"] = zip(
            *df["question"].apply(parse_question_field)
        )

        logger.info(f"Loaded {len(df)} rows from {file_path}")

        df = df[["id", "image", "question", "response", "answer"]]

        df = df.copy()
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        df["id"] = range(len(df))

        if sample_size is not None:
            if sample_size > len(df):
                logger.warning(
                    f"Requested sample size {sample_size} exceeds dataset size {len(df)}. Returning full dataset."
                )
                sample_size = len(df)

            df = df.head(sample_size)

        logger.info(f"Sampled {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    from multi_llm_debate.utils.logging_config import setup_logging

    logger = setup_logging(__name__, log_level=logging.INFO)
    # Example usage
    df = load_mllm_judge_score_dataset()
    print(df.info())
    print("Distinct values in answer column:")
    print(df["answer"].unique())
