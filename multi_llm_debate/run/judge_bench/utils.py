from typing import Literal

import pandas as pd

from datasets import load_dataset


def load_judge_bench_dataset(
    dataset_path: str = "datasets/JudgeBench",
) -> pd.DataFrame:
    """Load the JudgeBench dataset.

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        pd.DataFrame: DataFrame containing the JudgeBench data.
    """
    dataset_1 = load_dataset(
        "ScalerLab/JudgeBench",
        split="gpt",
        cache_dir=dataset_path,
    )
    if dataset_1 is None:
        raise ValueError("Failed to load the JudgeBench dataset.")
    # Convert to DataFrame
    df_1 = pd.DataFrame(dataset_1)

    dataset_2 = load_dataset(
        "ScalerLab/JudgeBench",
        split="claude",
        cache_dir=dataset_path,
    )
    if dataset_2 is None:
        raise ValueError("Failed to load the JudgeBench dataset.")
    # Convert to DataFrame
    df_2 = pd.DataFrame(dataset_2)

    # Concatenate the two DataFrames
    df = pd.concat([df_1, df_2], ignore_index=True)
    # Rename columns to match expected format

    return df


def extract_caption_a_b_answer(response: str) -> Literal["A", "B"]:
    """
    Extract answer from the response string, using the last occurrence.

    Args:
        response: The response string from the LLM.

    Returns:
        Answer: "A" or "B". Uses the last occurrence of A/B.
    """
    last_a = response.rfind("A")
    last_b = response.rfind("B")

    if last_a == -1 and last_b == -1:
        raise ValueError("Answer not recognized")

    return "A" if last_a > last_b else "B"

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
        return char1 if char1 > char2 else char2
    elif operator == "<":
        return char1 if char1 < char2 else char2
    else:
        raise ValueError("Invalid operator in comparison string")

def main() -> None:
    """Main function to load and display the JudgeBench dataset."""
    # Load the dataset
    dataframe = load_judge_bench_dataset()
    # Display the first few rows
    print(dataframe.head())
    print(f"Total entries: {len(dataframe)}")
    print("Columns:", dataframe.columns.tolist())
    print("Sample entry:", dataframe.iloc[0].to_dict())
    return dataframe


if __name__ == "__main__":
    # Run the main function to load and display the dataset
    main()
