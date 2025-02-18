from typing import Dict, List, Tuple

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


def model_configs_to_string(model_configs: List[Dict]) -> str:
    """Convert model configs to a string representation.

    Args:
        model_configs: List of model configuration dictionaries

    Returns:
        str: Formatted string representation sorted by model name and quantity

    Example:
        >>> configs = [
        ...     {"name": "llama2", "quantity": 3},
        ...     {"name": "llama3", "quantity": 3}
        ... ]
        >>> model_configs_to_string(configs)
        'llama2(3)+llama3(3)'
    """
    # Sort configs by model name and quantity
    sorted_configs = sorted(model_configs, key=lambda x: (x["name"], x["quantity"]))

    # Join with plus signs, remove spaces for filesystem safety
    return "+".join(
        f"{config['name']}({config['quantity']})" for config in sorted_configs
    )


def format_time(seconds: float) -> Tuple[str, str]:
    """Format time in seconds to human readable format and CSV format.

    Args:
        seconds (float): Time in seconds.

    Returns:
        tuple[str, str]: (human readable format, CSV format)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    if hours > 0:
        display_time = f"{hours}h {minutes}m {remaining_seconds:.2f}s"
        csv_time = f"{hours}:{minutes:02d}:{remaining_seconds:.2f}"
    elif minutes > 0:
        display_time = f"{minutes}m {remaining_seconds:.2f}s"
        csv_time = f"{minutes}:{remaining_seconds:.2f}"
    else:
        display_time = f"{remaining_seconds:.2f}s"
        csv_time = f"{remaining_seconds:.2f}"

    return display_time, csv_time
