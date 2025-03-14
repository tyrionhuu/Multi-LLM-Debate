import concurrent.futures
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


def _read_and_process_file(job_dict) -> Optional[dict]:
    """
    Worker function that runs in parallel.
    Reads the specified 'debate_round_X.json' file, parses it, and determines
    how many correct answers there are for that task in that round.

    job_dict keys:
      'round_file': Path to the debate_round_X.json file
      'task_id': string ID
      'round_number': int
      'answer': bool (correct label)
    """
    round_file = job_dict["round_file"]
    task_id = job_dict["task_id"]
    round_number = job_dict["round_number"]
    answer = job_dict["answer"]

    # Read JSON responses
    try:
        with open(round_file, "r") as f:
            responses = json.load(f)

        # Extract and normalize responses
        normalized_responses = []
        for response in responses:
            try:
                extracted = extract_bool_answer(response.get("response", ""))
                if extracted is not None:
                    normalized_responses.append(extracted)
            except Exception:
                # minimal log or pass
                pass

        if not normalized_responses:
            # No valid responses
            return None

        # Calculate how many are correct
        correct_count = sum(compare_bool(r, answer) for r in normalized_responses)

        # Return a record that we'll put into result_data
        return {
            "task_id": task_id,
            "round_number": round_number,
            "correct_count": correct_count,
            "num_agents": len(normalized_responses),
        }

    except Exception as e:
        logger.error(f"Error reading or processing file {round_file}: {e}")
        return None


def calculate_correct_rate_distribution_for_round_n(
    dataframe: pd.DataFrame,
    model_dir: Path,
    round_number: int,
    max_workers: int = 8,  # how many threads to use
) -> pd.DataFrame:
    """
    Calculate the correct rate distribution for a specific round, but read JSON files in parallel.

    Args:
        dataframe: DataFrame containing the experiment results.
        model_dir: Directory containing the model outputs.
        round_number: The round number to analyze.
        max_workers: Number of threads for reading files in parallel.

    Returns:
        DataFrame with correct rate distribution. Each row represents a task,
        with columns for the number of correct agents (0, 1, 2, etc.),
        task_id, and round_number.
    """
    # We'll gather "jobs" for parallel reading
    jobs_for_threadpool = []

    # Process each unique task
    task_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    pbar = tqdm(
        task_dirs,
        desc=f"Calculating correct rate distribution for round {round_number}",
    )

    for task_dir in pbar:
        task_id = task_dir.name
        task_id_str = str(task_id)  # to match dataframe indexing

        # Filter dataframe for this task
        task_df = dataframe[dataframe["id"].astype(str) == task_id_str]
        if task_df.empty:
            continue  # skip if no matching row

        # get the correct answer
        answer = task_df["answer"].iloc[0]

        # find the final round available
        final_round = get_final_round(task_dir)
        if final_round == -1:
            continue

        actual_round = min(round_number, final_round)
        round_file = task_dir / f"debate_round_{actual_round}.json"
        if not round_file.exists():
            continue

        # We'll process this file in parallel
        job_dict = {
            "round_file": round_file,
            "task_id": task_id,
            "round_number": round_number,
            "answer": answer,
        }
        jobs_for_threadpool.append(job_dict)

    # Now read all the files in parallel
    result_data = []
    max_agents = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(_read_and_process_file, j): j for j in jobs_for_threadpool
        }

        for future in concurrent.futures.as_completed(future_to_job):
            res = future.result()
            if res is not None:
                result_data.append(res)
                max_agents = max(max_agents, res["num_agents"])

    # Now build the DataFrame
    if not result_data:
        # No results
        bin_labels = [str(i) for i in range(10)]
        result_df = pd.DataFrame(columns=bin_labels + ["task_id", "round_number"])
        logger.warning("No valid data collected for correct rate distribution")
        return result_df

    # Convert result_data -> DataFrame
    df = pd.DataFrame(result_data)
    # e.g. columns: task_id, round_number, correct_count, num_agents

    # Create bin columns from 0..max_agents
    bin_labels = [str(i) for i in range(max_agents + 1)]
    for bin_label in bin_labels:
        df[bin_label] = (df["correct_count"] == int(bin_label)).astype(int)

    # drop the "correct_count" and "num_agents" columns
    df.drop(columns=["correct_count", "num_agents"], inplace=True)

    logger.info(f"Created distribution DataFrame with {len(df)} tasks")
    return df


def calculate_correct_rate_distribution(
    dataframe: pd.DataFrame,
    model_dir: Path,
    max_rounds: Optional[int] = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Calculate the correct rate distribution aggregated by round,
    using parallel reading for each round's JSON files.

    Args:
        dataframe: DataFrame containing the experiment results.
        model_dir: Directory containing the model outputs.
        max_rounds: Maximum number of rounds to process. If None, processes all available.
        max_workers: number of threads to use for parallel reading.

    Returns:
        DataFrame with data aggregated by round. Each row represents a round,
        with columns for the count of tasks having different numbers of correct
        agents (0, 1, 2, etc.) and 'round_number'.
    """
    task_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not task_dirs:
        logger.warning(f"No task directories found in {model_dir}")
        return pd.DataFrame()

    # Find maximum available round (sample some tasks)
    available_rounds = set()
    for task_dir in task_dirs[:20]:
        files = list(task_dir.glob("debate_round_*.json"))
        rounds = [int(f.stem.split("_")[-1]) for f in files]
        available_rounds.update(rounds)

    if not available_rounds:
        logger.warning("No debate round files found in sampled directories")
        return pd.DataFrame()

    max_available_round = max(available_rounds)
    rounds_to_process = min(max_available_round + 1, max_rounds or float("inf"))
    rounds_to_process = int(rounds_to_process)

    logger.info(f"Processing {rounds_to_process} rounds (0 to {rounds_to_process - 1})")

    aggregated_results = []

    for round_number in range(rounds_to_process):
        logger.info(f"Processing round {round_number}...")

        # Use the parallel version of reading
        result_df = calculate_correct_rate_distribution_for_round_n(
            dataframe=dataframe,
            model_dir=model_dir,
            round_number=round_number,
            max_workers=max_workers,
        )

        if not result_df.empty:
            bin_columns = [col for col in result_df.columns if col.isdigit()]
            bin_columns.sort(key=int)

            # Summation
            if "task_id" in result_df.columns:
                result_df = result_df.drop(columns=["task_id"])
            if "round_number" in result_df.columns:
                result_df = result_df.drop(columns=["round_number"])

            aggregated_row = {"round_number": round_number}
            for bin_col in bin_columns:
                aggregated_row[bin_col] = result_df[bin_col].sum()

            aggregated_row["total_tasks"] = len(result_df)
            aggregated_results.append(aggregated_row)
            logger.info(
                f"Aggregated data from {len(result_df)} tasks for round {round_number}"
            )
        else:
            logger.warning(f"No valid data found for round {round_number}")

    if not aggregated_results:
        logger.warning("No valid data collected from any round")
        return pd.DataFrame()

    combined_df = pd.DataFrame(aggregated_results)
    logger.info(f"Created aggregated DataFrame with {len(combined_df)} rounds")
    return combined_df


if __name__ == "__main__":
    import sys

    data_path = "output/bool_q/processed_data.csv"
    model_dir = "data/bool_q/llama3(7)"

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

    logger.info("Testing calculate_correct_rate_distribution in parallel...")

    try:
        aggregated_df = calculate_correct_rate_distribution(
            dataframe=dataframe,
            model_dir=model_dir_path,
            max_rounds=None,  # or some integer
            max_workers=8,  # adjust # of threads
        )

        if not aggregated_df.empty:
            print("\nAggregated DataFrame for all rounds:")
            print(aggregated_df.to_string())
            print(f"\nDataFrame shape: {aggregated_df.shape}")
            print(f"DataFrame columns: {', '.join(aggregated_df.columns)}")

        else:
            logger.warning("No aggregated data available to display")

    except Exception as e:
        logger.error(
            f"Error during parallel reading or distribution calculation: {e}",
            exc_info=True,
        )

    print("\n" + "=" * 80 + "\n")
