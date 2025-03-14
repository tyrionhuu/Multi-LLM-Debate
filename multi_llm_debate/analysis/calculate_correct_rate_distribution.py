import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm

from ..llm.parsers import extract_bool_answer
from .utils import compare_bool, draw_console_histogram, get_final_round

# Set up logging
logging.basicConfig(
    level=logging.INFO,
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


def calculate_correct_rate_distribution(
    dataframe: pd.DataFrame,
    model_dir: Path,
    max_rounds: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate the correct rate distribution aggregated by round.

    Args:
        dataframe: DataFrame containing the experiment results.
        model_dir: Directory containing the model outputs.
        max_rounds: Maximum number of rounds to process. If None, processes
            all available rounds.

    Returns:
        DataFrame with data aggregated by round. Each row represents a round,
        with columns for the count of tasks having different numbers of 
        correct agents (0, 1, 2, etc.) and the round_number.
    """
    all_results: List[pd.DataFrame] = []
    
    # Sample one task directory to determine the maximum round
    task_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not task_dirs:
        logger.warning(f"No task directories found in {model_dir}")
        return pd.DataFrame()
    
    # Find maximum round available across all tasks
    available_rounds = set()
    for task_dir in task_dirs[:20]:  # Sample a subset for efficiency
        files = list(task_dir.glob("debate_round_*.json"))
        rounds = [int(f.stem.split("_")[-1]) for f in files]
        available_rounds.update(rounds)
    
    if not available_rounds:
        logger.warning(f"No debate round files found in sampled directories")
        return pd.DataFrame()
    
    max_available_round = max(available_rounds)
    rounds_to_process = min(max_available_round + 1, max_rounds or float('inf'))
    rounds_to_process = int(rounds_to_process)
    
    logger.info(f"Processing {rounds_to_process} rounds (0 to {rounds_to_process-1})")
    
    aggregated_results = []
    
    # Process each round
    for round_number in range(rounds_to_process):
        logger.info(f"Processing round {round_number}...")
        
        try:
            result_df = calculate_correct_rate_distribution_for_round_n(
                dataframe=dataframe, 
                model_dir=model_dir, 
                round_number=round_number
            )
            
            if not result_df.empty:
                # Get the columns that represent numbers of correct agents (0, 1, 2, etc.)
                bin_columns = [col for col in result_df.columns if col.isdigit()]
                bin_columns.sort(key=int)
                
                # Drop the task_id column and aggregate by summing across all tasks
                if "task_id" in result_df.columns:
                    result_df = result_df.drop(columns=["task_id"])
                
                # Sum the counts for each bin
                aggregated_row = {"round_number": round_number}
                for bin_col in bin_columns:
                    aggregated_row[bin_col] = result_df[bin_col].sum()
                
                # Add total tasks count for convenience
                aggregated_row["total_tasks"] = len(result_df)
                
                aggregated_results.append(aggregated_row)
                logger.info(f"Aggregated data from {len(result_df)} tasks for round {round_number}")
            else:
                logger.warning(f"No valid data found for round {round_number}")
                
        except Exception as e:
            logger.error(f"Error processing round {round_number}: {e}", exc_info=True)
    
    # Combine all results
    if aggregated_results:
        combined_df = pd.DataFrame(aggregated_results)
        logger.info(f"Created aggregated DataFrame with {len(combined_df)} rounds")
        return combined_df
    else:
        logger.warning("No valid data collected from any round")
        return pd.DataFrame()


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

    # First test the aggregated function for all rounds
    logger.info("Testing calculate_correct_rate_distribution for all rounds...")
    
    try:
        aggregated_df = calculate_correct_rate_distribution(
            dataframe=dataframe, 
            model_dir=model_dir_path
        )
        
        if not aggregated_df.empty:
            print("\nAggregated DataFrame for all rounds:")
            print(aggregated_df.to_string())
            print(f"\nDataFrame shape: {aggregated_df.shape}")
            print(f"DataFrame columns: {', '.join(aggregated_df.columns)}")
            
            # Get numeric columns (bins)
            bin_columns = [col for col in aggregated_df.columns 
                          if col.isdigit()]
            bin_columns.sort(key=int)
            
            if bin_columns:
                # Calculate percentages for each round
                for _, row in aggregated_df.iterrows():
                    round_num = int(row['round_number'])
                    total = row['total_tasks']
                    
                    print(f"\nRound {round_num} distribution:")
                    print(f"Total tasks: {total}")
                    
                    # Create a dictionary for the histogram
                    bin_counts = {bin_col: row[bin_col] for bin_col in bin_columns}
                    bin_percentages = {
                        bin_col: (row[bin_col] / total * 100) 
                        for bin_col in bin_columns
                    }
                    
                    for bin_col in bin_columns:
                        count = row[bin_col]
                        pct = bin_percentages[bin_col]
                        print(f"  {bin_col} correct agents: {count} tasks ({pct:.2f}%)")
                    
                    # Draw histogram
                    histogram = draw_console_histogram(
                        bin_counts,
                        title=f"Number of Correct Agents Distribution (Round {round_num})",
                        height=15,
                        bar_char="█",
                        fine_grained=True,
                    )
                    print("\n" + histogram + "\n")
            
        else:
            logger.warning("No aggregated data available to display")
    
    except Exception as e:
        logger.error(f"Error testing aggregated function: {e}", exc_info=True)
    
    print("\n" + "="*80 + "\n")  # Separator between aggregated and per-round analysis

