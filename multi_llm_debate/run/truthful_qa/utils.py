import re
from typing import Literal
import pandas as pd
import os
from datasets import load_dataset, load_from_disk

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
                name=dataset_name,
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

def preprocess_dataframe():
    pass





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
