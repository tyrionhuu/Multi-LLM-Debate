import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def process_distribution_data(
    result_df: pd.DataFrame,
    round_number: int,
) -> Dict[str, float]:
    """Process distribution data to get percentages for each bin (e.g. '0', '1', '2', ...).
    
    Args:
        result_df: DataFrame with distribution data from calculate_correct_rate_distribution_for_round_n.
        round_number: The round number being processed.
        
    Returns:
        Dictionary mapping bin labels (strings) to percentage of tasks in that bin.
    """
    bin_columns = [col for col in result_df.columns if col.isdigit()]
    bin_columns.sort(key=int)
    
    if not bin_columns or result_df.empty:
        logger.warning(f"No bins found for round {round_number}")
        return {}
    
    task_count = len(result_df)
    bin_sums = result_df[bin_columns].sum()
    bin_percentages = (bin_sums / task_count * 100).to_dict()
    
    return bin_percentages