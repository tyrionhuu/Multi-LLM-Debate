import logging
import os
import random
import re
from typing import List, Literal, Optional

import pandas as pd

from datasets import load_dataset, load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_truthful_qa_dataset(
    dataset_path: str = "datasets/TruthfulQA",
    dataset_name: str = "domenicrosati/TruthfulQA",
    random_state: int = None,
) -> pd.DataFrame:
    """Load the TruthfulQA dataset.

    Args:
        dataset_path: Path to the dataset directory. If it exists locally,
            it will be loaded from disk; otherwise, it will be downloaded.
        dataset_name: Name of the dataset to load from Hugging Face.
        random_state: Random seed for shuffling. If None, the dataset will be
            randomized differently each time.

    Returns:
        pd.DataFrame: DataFrame containing the TruthfulQA data with randomized order.
    """
    # Initialize empty DataFrame
    df = None

    # Try to load local dataset first
    if os.path.exists(dataset_path):
        try:
            dataset = load_from_disk(dataset_path)
            df = pd.DataFrame(dataset)
            print("Loaded TruthfulQA dataset from local path.")
        except Exception as e:
            print(f"Failed to load local dataset: {e}")

    if df is None:
        # Load from Hugging Face datasets
        try:
            dataset = load_dataset(
                path=dataset_name,
                split="train",
                cache_dir=dataset_path,
            )
            df = pd.DataFrame(dataset)
            print("Loaded TruthfulQA dataset from Hugging Face.")
        except Exception as e:
            print(f"Failed to load dataset from Hugging Face: {e}")

    # Shuffle the DataFrame if random_state is provided
    if random_state is not None:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def preprocess_dataframe(
    dataframe: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Preprocess the TruthfulQA DataFrame to ensure it has all required columns.

    Args:
        dataframe: Input DataFrame from TruthfulQA dataset
        random_state: Random seed for shuffling. If None, the dataset will be
            randomized differently each time.

    Returns:
        pd.DataFrame: Processed DataFrame with all required columns and
            multiple-choice format
    """
    if dataframe is None:
        raise ValueError("Input DataFrame is None. Please load the dataset first.")

    df = dataframe.copy()
    random.seed(random_state)

    # Create ID column if missing
    if 'id' not in df.columns:
        df['id'] = df.index + 1

    # Helper to choose random answers
    def _random_answer(answers_str):
        answers = [a.strip() for a in answers_str.split(';') if a.strip()]
        return random.choice(answers) if answers else None

    # Select correct and incorrect answers
    df['correct_ans'] = df['Correct Answers'].apply(_random_answer)
    
    # Select two distinct incorrect answers
    def _two_wrong_answers(answers_str):
        answers = [a.strip() for a in answers_str.split(';') if a.strip()]
        if len(answers) < 2:
            return (answers[0], answers[0]) if answers else (None, None)
        a, b = random.sample(answers, k=2)
        return (a, b) if a != b else (a, random.choice([x for x in answers if x != a]))

    incorrect_pairs = df['Incorrect Answers'].apply(_two_wrong_answers)
    df[['wrong1', 'wrong2']] = pd.DataFrame(incorrect_pairs.tolist(), index=df.index)

    # Assign options A/B/C randomly
    df['answer'] = random.choices(['A', 'B', 'C'], k=len(df))
    
    # Map answers to options
    def _map_options(row):
        opts = {'A': None, 'B': None, 'C': None}
        opts[row['answer']] = row['correct_ans']
        others = [o for o in opts if o != row['answer']]
        opts[others[0]], opts[others[1]] = row['wrong1'], row['wrong2']
        return opts['A'], opts['B'], opts['C']
    
    df[['response_A', 'response_B', 'response_C']] = df.apply(_map_options, axis=1)

    return df[['id', 'Question', 'response_A', 'response_B', 'response_C', 'answer']].rename(
        columns={'Question': 'question'}
    )


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


def main():
    df = load_truthful_qa_dataset(
        "/Users/tyrionhuu/projects/research_projects/Multi-LLM-Debate/datasets/TruthfulQA"
    )
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
