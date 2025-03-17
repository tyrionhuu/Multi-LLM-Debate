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
    dataset = load_dataset(
        dataset_path,
        split="train",
    )
    if dataset is None:
        raise ValueError("Failed to load the JudgeBench dataset.")
    # Convert to DataFrame
    df = pd.DataFrame(dataset)

    return df


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
