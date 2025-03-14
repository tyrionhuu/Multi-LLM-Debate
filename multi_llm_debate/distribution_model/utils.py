import pandas as pd
def extract_correct_counts(round_df: pd.DataFrame) -> pd.Series:
    """
    Given the distribution DataFrame with columns "task_id", "round_number", "0", "1", ..., "k",
    this returns a Series of integer counts: how many agents were correct for each task.
    """
    # Identify columns that are numeric bin labels
    bin_cols = [c for c in round_df.columns if c.isdigit()]
    counts = []
    for _, row in round_df.iterrows():
        for b in bin_cols:
            if row[b] == 1:
                counts.append(int(b))
                break
    return pd.Series(counts)