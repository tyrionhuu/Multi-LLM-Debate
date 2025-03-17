import pandas as pd

from datasets import load_dataset


def load_judge_bench_dataset(
    dataset_path: str = "datasets/JudgeBench",
    split: str = None,
) -> pd.DataFrame:
    """Load the JudgeBench dataset.

    Args:
        dataset_path: Path to the dataset directory.
        split: Specific split to load ('claude', 'gpt', etc.). 
               If None, loads all available splits.

    Returns:
        pd.DataFrame: DataFrame containing the JudgeBench data.
    """
    if split is not None:
        # Load specific split
        dataset = load_dataset(
            dataset_path,
            split=split,
        )
        if dataset is None:
            raise ValueError(f"Failed to load the JudgeBench {split} split.")
        return pd.DataFrame(dataset)
    
    # Load all splits
    dataset = load_dataset(dataset_path)
    if dataset is None:
        raise ValueError("Failed to load the JudgeBench dataset.")
    
    # Combine all splits into a single DataFrame
    all_dfs = []
    for split_name, split_data in dataset.items():
        df = pd.DataFrame(split_data)
        df['split'] = split_name  # Add split name as a column
        all_dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def main() -> None:
    """Main function to load and display the JudgeBench dataset."""
    # Load the complete dataset
    dataframe = load_judge_bench_dataset()
    # Display the first few rows
    print(dataframe.head())
    print(f"Total entries: {len(dataframe)}")
    print("Columns:", dataframe.columns.tolist())
    print("Sample entry:", dataframe.iloc[0].to_dict())
    
    # Show the number of entries per split
    if 'split' in dataframe.columns:
        print("\nEntries per split:")
        print(dataframe['split'].value_counts())
    
    return dataframe

if __name__ == "__main__":
    # Run the main function to load and display the dataset
    main()
