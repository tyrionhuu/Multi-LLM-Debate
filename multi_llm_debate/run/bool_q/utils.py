import pandas as pd


def process_bool_q_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Process the BoolQ DataFrame to ensure it has all required columns.

    Args:
        dataframe: Input DataFrame from BoolQ dataset

    Returns:
        pd.DataFrame: Processed DataFrame with all required columns
    """
    # Create a copy to avoid modifying the original
    processed_df = dataframe.copy()

    # Generate ID from index if 'id' column doesn't exist
    if "id" not in processed_df.columns:
        processed_df["id"] = processed_df.index + 1

    return processed_df
