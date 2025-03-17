import pandas as pd
import logging
from typing import Optional

from datasets import load_dataset, Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_judge_bench_dataset(
    dataset_path: str = "datasets/JudgeBench",
    split: Optional[str] = None,
) -> pd.DataFrame:
    """Load the JudgeBench dataset with actual content.

    Args:
        dataset_path: Path to the dataset directory.
        split: Specific split to load ('claude', 'gpt', etc.). 
               If None, loads all available splits.

    Returns:
        pd.DataFrame: DataFrame containing the JudgeBench data.
    """
    try:
        # First try loading the dataset to see its structure
        dataset = load_dataset(dataset_path)
        logger.info(f"Available keys in dataset: {list(dataset.keys())}")
        
        # Check the first split to understand the structure
        first_split_name = list(dataset.keys())[0]
        first_split = dataset[first_split_name]
        logger.info(f"First split '{first_split_name}' features: {list(first_split.features)}")
        logger.info(f"First split sample: {first_split[0] if len(first_split) > 0 else 'Empty'}")
        
        all_dfs = []
        for split_name, split_data in dataset.items():
            if split is not None and split_name != split:
                continue
                
            # Convert to DataFrame with all fields
            df = split_data.to_pandas()
            
            # Add source split name as a column
            df['source_split'] = split_name
            all_dfs.append(df)
            
            logger.info(f"Loaded {len(df)} entries from split '{split_name}'")
            if len(df) > 0:
                logger.info(f"Available columns: {df.columns.tolist()}")
        
        if not all_dfs:
            raise ValueError(f"No data found in the dataset for split '{split}'")
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} entries")
        return combined_df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def main() -> None:
    """Main function to load and display the JudgeBench dataset."""
    # Load the complete dataset
    dataframe = load_judge_bench_dataset()
    
    # Display basic information about the dataframe
    print("\n--- Dataset Information ---")
    print(f"Total entries: {len(dataframe)}")
    print(f"Columns: {dataframe.columns.tolist()}")
    
    # Display the first few rows
    print("\n--- First 5 rows ---")
    print(dataframe.head())
    
    # Show distribution by source split
    if 'source_split' in dataframe.columns:
        print("\n--- Entries per source split ---")
        print(dataframe['source_split'].value_counts())
    
    # Display a more detailed look at the first entry
    if len(dataframe) > 0:
        print("\n--- First entry detailed view ---")
        first_entry = dataframe.iloc[0]
        for col, val in first_entry.items():
            # For large text fields, show only beginning
            if isinstance(val, str) and len(val) > 100:
                print(f"{col}: {val[:100]}... (truncated)")
            else:
                print(f"{col}: {val}")
    
    return dataframe


if __name__ == "__main__":
    # Run the main function to load and display the dataset
    main()
