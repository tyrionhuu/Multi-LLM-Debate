from pathlib import Path

import pandas as pd


def analyze_round_number(
    base_dir: Path, max_round_number: int | None = 10, output_csv: Path | None = None
) -> pd.DataFrame:
    """
    Analyzes the convergence of LLMs in a debate setting.

    Args:
        base_dir (Path): Base directory containing the debate data.
        max_round_number (int | None, optional): Maximum number of rounds to analyze.
            Defaults to 10.
        output_csv (Path | None, optional): Path to save the output CSV file.
            If None, no CSV file will be created. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the analysis results with columns for
            model configuration and round numbers.
    """
    directories = [d for d in base_dir.iterdir() if d.is_dir()]
    results: pd.DataFrame = pd.DataFrame()
    for directory in directories:
        model_configuration = directory.name

        # Initialize row data with model configuration
        row_data = {"model_configuration": model_configuration}

        # Get all subdirectories
        subdirs = [d for d in directory.iterdir() if d.is_dir()]

        # Count files in each subdirectory
        for round_num in range(1, max_round_number + 1):
            count = sum(
                1 for subdir in subdirs if sum(1 for _ in subdir.glob("*")) >= round_num
            )
            row_data[str(round_num)] = count

        # Append row to results
        results = pd.concat([results, pd.DataFrame([row_data])], ignore_index=True)

    # Reorder columns to match desired header format
    column_order = ["model_configuration"] + [
        str(i) for i in range(1, max_round_number + 1)
    ]
    results = results[column_order]

    # Save to CSV if path is provided
    if output_csv is not None:
        results.to_csv(output_csv, index=False)
    
    return results
