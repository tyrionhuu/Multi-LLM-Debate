import os
import re
from typing import Literal

import pandas as pd

from datasets import load_dataset, load_from_disk


def load_judge_bench_dataset(
    dataset_path: str = "datasets/JudgeBench",
    random_state: int = None,
) -> pd.DataFrame:
    """Load the JudgeBench dataset.

    Args:
        dataset_path: Path to the dataset directory. If it exists locally,
            it will be loaded from disk; otherwise, it will be downloaded.
        random_state: Random seed for shuffling. If None, the dataset will be
            randomized differently each time.

    Returns:
        pd.DataFrame: DataFrame containing the JudgeBench data with randomized order.
    """
    # Initialize empty DataFrames
    df_1, df_2 = None, None

    # Try to load local datasets first
    if os.path.exists(dataset_path):
        try:
            # Define paths for the two splits
            gpt_path = os.path.join(dataset_path, "gpt")
            claude_path = os.path.join(dataset_path, "claude")

            # Check if both splits exist and are not empty
            if os.path.exists(gpt_path) and os.path.exists(claude_path):
                dataset_1 = load_from_disk(gpt_path)
                dataset_2 = load_from_disk(claude_path)
                df_1 = pd.DataFrame(dataset_1)
                df_2 = pd.DataFrame(dataset_2)
                print("Loaded GPT and Claude splits from local paths.")
            else:
                print("One or both of the splits are missing. Downloading datasets...")
                df_1, df_2 = None, None

        except Exception as e:
            print(f"Error loading local dataset: {e}")
            df_1, df_2 = None, None

    # Fall back to downloading if local loading failed
    if df_1 is None:
        print("Local GPT split not found, downloading from HuggingFace...")
        dataset_1 = load_dataset(
            "ScalerLab/JudgeBench",
            split="gpt",
            cache_dir=dataset_path,
        )
        if dataset_1 is None:
            raise ValueError("Failed to load the JudgeBench GPT dataset.")
        df_1 = pd.DataFrame(dataset_1)

    if df_2 is None:
        print("Local Claude split not found, downloading from HuggingFace...")
        dataset_2 = load_dataset(
            "ScalerLab/JudgeBench",
            split="claude",
            cache_dir=dataset_path,
        )
        if dataset_2 is None:
            raise ValueError("Failed to load the JudgeBench Claude dataset.")
        df_2 = pd.DataFrame(dataset_2)

    # Concatenate the two DataFrames
    df = pd.concat([df_1, df_2], ignore_index=True)

    # Preprocess the dataframe (inlined from preprocess_dataframe)
    # Rename columns
    df = df.rename(columns={"pair_id": "id", "label": "answer"})
    # Drop unnecessary columns
    df = df.drop(columns=["original_id", "source"], errors="ignore")

    # Randomize the entries
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


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


def extract_bigger_char(comparison: str) -> str:
    """
    Extract the bigger character from the comparison string.

    Args:
        comparison: The comparison string from the LLM.

    Returns:
        str: The bigger character from the comparison string.
    """
    char1 = comparison[0]
    operator = comparison[1]
    char2 = comparison[2]

    if operator == ">":
        return char1
    elif operator == "<":
        return char2
    else:
        raise ValueError("Invalid comparison operator")


def compare_judge_bench_response(
    response: Literal["A", "B"],
    answer: str,
) -> bool:
    """Compare the responses from the judge bench.

    Args:
        response: The response string from the LLM.
        answer: The correct answer to the question ("A"/"B").

    Returns:
        bool: True if the response matches the answer, False otherwise.
    """
    try:
        gold_answer = extract_bigger_char(answer)
        return response == gold_answer.upper()
    except Exception as e:
        print(f"Error comparing responses: {e}")
        return False


def main() -> None:
    response = "Reasoning:\nStep 1: Both assistants followed the user's instructions and explained the steps to solve the problem.\nStep 2: Both assistants correctly used the ideal gas law and the polytropic relation to calculate the work done.\nStep 3: However, Assistant B provided a more direct and simplified expression for the work done in terms of P\u2081, V\u2081, and k, which made the calculation more straightforward and less prone to errors.\n\nFinal Answer: B"
    extracted_answer = extract_caption_a_b_answer(response)
    print(f"Extracted answer: {extracted_answer}")


if __name__ == "__main__":
    # Run the main function to load and display the dataset
    main()
