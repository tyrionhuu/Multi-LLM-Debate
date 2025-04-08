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

    # Create a copy to avoid modifying the original
    processed_df = dataframe.copy()

    # Set random seed for reproducibility
    random.seed(random_state)

    # Generate ID from index if 'id' column doesn't exist
    if "id" not in processed_df.columns:
        processed_df["id"] = processed_df.index + 1

    # Select a random correct answer for each question
    processed_df["selected_correct_answer"] = processed_df["Correct Answers"].apply(
        lambda x: _choose_random_answer(x, random_state)
    )

    # Select two random incorrect answers for each question
    processed_df["incorrect_answer1"] = processed_df["Incorrect Answers"].apply(
        lambda x: _choose_random_answer(x, random_state)
    )

    processed_df["incorrect_answer2"] = processed_df["Incorrect Answers"].apply(
        lambda x: _choose_random_answer(x, random_state)
    )

    # Make sure the two incorrect answers are different
    for idx, row in processed_df.iterrows():
        incorrect_answers = row["Incorrect Answers"].split(";")
        incorrect_answers = [ans.strip() for ans in incorrect_answers if ans.strip()]

        if (
            row["incorrect_answer1"] == row["incorrect_answer2"]
            and len(incorrect_answers) > 1
        ):
            remaining = [
                ans for ans in incorrect_answers if ans != row["incorrect_answer1"]
            ]
            if remaining:
                processed_df.at[idx, "incorrect_answer2"] = random.choice(remaining)

    # Randomly determine which option (A, B, C) will be the correct answer
    processed_df["correct_option"] = [
        random.choice(["A", "B", "C"]) for _ in range(len(processed_df))
    ]

    # Create the three options based on the correct_option
    for idx, row in processed_df.iterrows():
        options = {"A": None, "B": None, "C": None}

        # Assign the correct answer to the chosen option
        options[row["correct_option"]] = row["selected_correct_answer"]

        # Assign incorrect answers to the other options
        remaining_options = [
            opt for opt in ["A", "B", "C"] if opt != row["correct_option"]
        ]
        options[remaining_options[0]] = row["incorrect_answer1"]
        options[remaining_options[1]] = row["incorrect_answer2"]

        # Store options in the DataFrame
        processed_df.at[idx, "option_A"] = options["A"]
        processed_df.at[idx, "option_B"] = options["B"]
        processed_df.at[idx, "option_C"] = options["C"]

    # Keep original columns that might be useful
    columns_to_keep = [
        "id",
        "Question",
        "Best Answer",
        "Correct Answers",
        "Incorrect Answers",
        "selected_correct_answer",
        "correct_option",
        "option_A",
        "option_B",
        "option_C",
    ]

    return processed_df[columns_to_keep]


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


def _choose_random_answer(input: str, random_state: int = 42) -> Optional[str]:
    """Choose a random answer from the input string.

    Args:
        input: The input string containing possible answers.
        random_state: Random seed for reproducibility.

    Returns:
        str: A randomly chosen answer from the input.
    """

    def _parse_string_to_list(input: str, divider: str = ";") -> List[str]:
        """Parse a string into a list based on a divider.

        Args:
            input: The input string to be parsed.
            divider: The string used to split the input.

        Returns:
            List[str]: A list of parsed strings.
        """
        return [item.strip() for item in input.split(divider) if item.strip()]

    random.seed(random_state)
    answers = _parse_string_to_list(input)
    if not answers:
        logger.warning("No answers found to choose from.")
        return None
    return random.choice(answers)


def main():
    df = load_truthful_qa_dataset(
        "/Users/tyrionhuu/projects/research_projects/Multi-LLM-Debate/datasets/TruthfulQA"
    )
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
