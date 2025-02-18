import pandas as pd
from ...utils.generate_hash import generate_hash
def process_bool_q_df(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process a DataFrame for the Boolean Question task.

    Args:
        dataframe: Pandas DataFrame containing question, answer, passage and id

    Returns:
        Processed DataFrame with additional columns for model responses.
    """
    dataframe = dataframe.copy()
    dataframe["id"] = dataframe["id"].apply(generate_hash)
    return dataframe