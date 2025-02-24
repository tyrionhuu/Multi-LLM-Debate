from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def count_rounds_for_model(
    model_dir: Path, max_round_number: int
) -> Dict[str, int | str]:
    """
    Count the number of rounds completed for a specific model configuration.

    Args:
        model_dir (Path): Directory containing the model's debate data.
        max_round_number (int): Maximum number of rounds to analyze.

    Returns:
        Dict[str, int | str]: Dictionary containing round counts with model
            configuration name and counts for each round.
    """
    model_configuration = model_dir.name
    row_data = {"model_configuration": model_configuration}

    # Get all subdirectories
    subdirs = [d for d in model_dir.iterdir() if d.is_dir()]

    # Count files in each subdirectory
    for round_num in range(1, max_round_number + 1):
        count = sum(
            1 for subdir in subdirs if sum(1 for _ in subdir.glob("*")) >= round_num
        )
        row_data[str(round_num)] = count

    return row_data


def analyze_round_number(
    base_dir: Path, max_round_number: int | None = 10, output_csv: Optional[Path] = None
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
    results = []

    # Analyze each model configuration
    for directory in directories:
        row_data = count_rounds_for_model(directory, max_round_number)
        results.append(row_data)

    # Create and format DataFrame
    results_df = pd.DataFrame(results)
    column_order = ["model_configuration"] + [
        str(i) for i in range(1, max_round_number + 1)
    ]
    results_df = results_df[column_order]

    # Save to CSV if path is provided
    if output_csv is not None:
        results_df.to_csv(output_csv, index=False)

    return results_df


def main() -> None:
    """Test the analyze_round_number function with example parameters."""
    base_dir = Path("data/bool_q")

    try:
        df = analyze_round_number(
            base_dir=base_dir,
            max_round_number=10,
        )
        print("Analysis completed successfully!")
        print("\nDataFrame head:")
        print(df.head())
    except Exception as e:
        print(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()
