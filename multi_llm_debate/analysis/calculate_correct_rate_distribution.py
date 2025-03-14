import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..llm.parsers import extract_bool_answer
from .utils import compare_bool, draw_console_histogram, get_final_round

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("analysis.log"),
    ],
)
logger = logging.getLogger(__name__)


def calculate_correct_rate_distribution_for_round_n(
    dataframe: pd.DataFrame,
    model_dir: Path,
    round_number: int,
) -> pd.DataFrame:
    """Calculate the correct rate distribution for a specific round.

    Args:
        dataframe: DataFrame containing the experiment results.
        model_dir: Directory containing the model outputs.
        round_number: The round number to analyze.

    Returns:
        DataFrame with correct rate distribution. Each row represents a task,
        with columns for the number of correct agents (0, 1, 2, etc.),
        task_id, and round_number.
    """
    result_data = []

    # Process each unique task
    task_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    pbar = tqdm(
        task_dirs,
        desc=f"Calculating correct rate distribution for round {round_number}",
    )

    # Track maximum number of agents to create appropriate bins later
    max_agents = 0

    for task_dir in pbar:
        task_id = task_dir.name
        # Convert to string for consistent comparison
        task_id_str = str(task_id)

        # Filter dataframe for this task using string comparison
        task_df = dataframe[dataframe["id"].astype(str) == task_id_str]

        if task_df.empty:
            logger.debug(f"Skipping task {task_id}: Not found in dataframe")
            continue

        # Get the correct answer for the task
        answer = task_df["answer"].iloc[0]

        # Load the debate data for the specified round
        final_round = get_final_round(task_dir)
        if final_round == -1:
            logger.warning(f"No debate data found for task {task_id}")
            continue

        # Use the specified round or the final round if the specified round exceeds it
        actual_round = min(round_number, final_round)
        round_file = task_dir / f"debate_round_{actual_round}.json"

        if not round_file.exists():
            logger.warning(
                f"No debate data found for task {task_id} in round {actual_round}"
            )
            continue

        try:
            # Read responses from the round file
            with open(round_file, "r") as f:
                responses = json.load(f)

            # Extract and normalize responses
            normalized_responses = []
            for response in responses:
                try:
                    extracted = extract_bool_answer(response.get("response", ""))
                    if extracted is not None:
                        normalized_responses.append(extracted)
                except Exception as e:
                    logger.debug(f"Error extracting response: {e}")

            if not normalized_responses:
                logger.debug(
                    f"No valid responses for task {task_id} in round {actual_round}"
                )
                continue

            # Calculate the number of correct agents for this task
            correct_count = sum(
                1 for r in normalized_responses if compare_bool(r, answer)
            )

            # Keep track of maximum number of agents to define bins later
            max_agents = max(max_agents, len(normalized_responses))

            # Create a row for this task
            row = {
                "task_id": task_id,
                "round_number": round_number,
                "correct_count": correct_count,
            }
            result_data.append(row)

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            continue

    # Create result DataFrame
    if result_data:
        result_df = pd.DataFrame(result_data)

        # Create the bins based on the actual number of agents observed (0 to max_agents)
        bin_labels = [str(i) for i in range(max_agents + 1)]

        # Convert from raw counts to one-hot encoding for the bins
        for bin_label in bin_labels:
            result_df[bin_label] = (
                result_df["correct_count"] == int(bin_label)
            ).astype(int)

        # Drop the temporary correct_count column
        result_df = result_df.drop(columns=["correct_count"])

        logger.info(f"Created distribution DataFrame with {len(result_df)} tasks")
    else:
        # We don't know max_agents if there's no data, so assume a reasonable default
        bin_labels = [str(i) for i in range(10)]  # Default to 0-9 agents
        result_df = pd.DataFrame(columns=bin_labels + ["task_id", "round_number"])
        logger.warning("No valid data collected for correct rate distribution")

    return result_df


if __name__ == "__main__":
    import sys

    # Hardcoded configuration
    data_path = "output/bool_q/processed_data.csv"
    model_dir = "data/bool_q/llama3(7)"
    # output_path_pattern = "output/distribution_round_{}.csv"  # Template for output paths

    # Load data
    try:
        dataframe = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    model_dir_path = Path(model_dir)
    if not model_dir_path.exists() or not model_dir_path.is_dir():
        logger.error(f"Model directory does not exist: {model_dir}")
        sys.exit(1)

    # Process rounds 0 through 5
    for round_number in range(6):  # 0 to 5
        logger.info(f"Processing round {round_number}...")

        # Calculate distribution
        result_df = calculate_correct_rate_distribution_for_round_n(
            dataframe=dataframe, model_dir=model_dir_path, round_number=round_number
        )

        # Print raw DataFrame output
        print(f"\nRaw DataFrame for round {round_number}:")
        print(result_df.to_string(max_rows=20))
        print(f"\nDataFrame shape: {result_df.shape}")
        print(f"DataFrame columns: {', '.join(result_df.columns)}")
        
        # Get summary statistics
        if not result_df.empty:
            numeric_cols = [col for col in result_df.columns if col.isdigit()]
            if numeric_cols:
                print("\nSummary statistics for bin columns:")
                print(result_df[numeric_cols].describe())

        # Print summary
        # Update bin column selection to look for numeric columns instead of ones with "-"
        bin_columns = [col for col in result_df.columns if col.isdigit()]
        # Sort the bin columns numerically
        bin_columns.sort(key=int)

        task_count = len(result_df)

        logger.info(f"Results for round {round_number}:")
        logger.info(f"Total tasks analyzed: {task_count}")

        # Calculate bin distribution
        if not result_df.empty and bin_columns:
            bin_sums = result_df[bin_columns].sum()
            bin_percentages = (bin_sums / task_count * 100).to_dict()

            logger.info(f"Correct rate distribution for round {round_number}:")
            for bin_label, percentage in bin_percentages.items():
                logger.info(f"  {bin_label} correct agents: {percentage:.2f}%")

            # Draw a more detailed histogram in the console
            histogram = draw_console_histogram(
                bin_sums.to_dict(),
                title=f"Number of Correct Agents Distribution (Round {round_number})",
                height=20,  # More reasonable height while still showing detail
                bar_char="█",
                fine_grained=True,  # Enable fine-grained display
            )
            print("\n" + histogram + "\n")

        # Save results if needed
        # output_path = output_path_pattern.format(round_number)
        # result_df.to_csv(output_path, index=False)
        # logger.info(f"Results for round {round_number} saved to {output_path}")

        print("\n" + "-" * 80 + "\n")  # Add separator between rounds
