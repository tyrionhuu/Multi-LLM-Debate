from typing import Literal

import pandas as pd

from datasets import load_dataset


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the JudgeBench dataframe by renaming and dropping columns.

    Args:
        df: Raw dataframe loaded from the JudgeBench dataset.

    Returns:
        pd.DataFrame: Processed dataframe with renamed and dropped columns.
    """
    # Rename columns
    df = df.rename(columns={"pair_id": "id", "label": "answer"})

    # Drop unnecessary columns
    df = df.drop(columns=["original_id", "source"], errors="ignore")

    return df


def load_judge_bench_dataset(
    dataset_path: str = "datasets/JudgeBench",
    random_state: int = None,
) -> pd.DataFrame:
    """Load the JudgeBench dataset.

    Args:
        dataset_path: Path to the dataset directory.
        random_state: Random seed for shuffling. If None, the dataset will be
            randomized differently each time.

    Returns:
        pd.DataFrame: DataFrame containing the JudgeBench data with randomized order.
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

    # Preprocess the dataframe
    df = preprocess_dataframe(df)

    # Randomize the entries
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

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
        return char1
    elif operator == "<":
        return char2
    else:
        raise ValueError("Invalid comparison operator")


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
