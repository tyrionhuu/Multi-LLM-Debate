from pathlib import Path
from typing import Union
import pandas as pd
def json_to_processed_df(
    json_path: Union[str, Path]
):
    """
    Convert a JSON file to a DataFrame.

    Args:
        json_path (Union[str, Path]): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the JSON file.
    """
    # Ensure the path is a Path object
    json_path = Path(json_path)

    if not json_path.is_file():
        raise FileNotFoundError(f"File not found: {json_path}")
    try:
        # Read the JSON file into a DataFrame
        df = pd.read_json(json_path, lines=True)
        print(df.head())
        return df
    except ValueError as e:
        raise ValueError(f"Error reading JSON file {json_path}: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while processing {json_path}: {e}")

json_to_processed_df("data/ice_score/conala_grade.json")