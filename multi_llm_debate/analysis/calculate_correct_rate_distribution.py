import logging
from typing import Optional

import pandas as pd
from tqdm import tqdm

from ..llm.parsers import extract_bool_answer
from .utils import compare_bool

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
    df_answers: pd.DataFrame,
    df_debates: pd.DataFrame,
    round_number: int,
) -> pd.DataFrame:
    """
    Compute correct-rate distribution for a specific round, using the merged CSV data
    (df_debates) instead of reading individual JSON files.

    Args:
        df_answers: DataFrame with columns ["id", "answer"].
                    `id` is numeric, `answer` is the correct boolean.
        df_debates: DataFrame from debate_rounds.csv, containing columns:
                    [task_id, round_number, agent_index, agent_id, model, response]
        round_number: which round to analyze

    Returns:
        DataFrame with columns [task_id, round_number, 0, 1, 2, ...]
        where each row is a single task, and there's exactly one "1" in
        the bin column that matches how many agents were correct.
    """
    # 1) Filter debate_df to this round
    df_this_round = df_debates[df_debates["round_number"] == round_number]
    if df_this_round.empty:
        return pd.DataFrame()  # No data for this round

    # 2) We'll need to group by task_id. For each task, figure out how many
    #    agents answered correctly.
    merged_rows = []
    # We do a group-by on "task_id" so we can handle each task's set of agent responses
    grouped = df_this_round.groupby("task_id")

    max_agents = 0

    for task_id_val, group_df in tqdm(
        grouped, desc=f"Round {round_number}", unit="task"
    ):
        # Convert to int or str for consistent merges
        # The df_answers has "id" that lines up with "task_id"
        # We'll find the correct label by matching "id == task_id_val"
        ans_row = df_answers[df_answers["id"] == task_id_val]
        if ans_row.empty:
            # no known correct label
            continue
        correct_label = ans_row["answer"].iloc[0]

        # For each agent response, parse True/False if possible
        normalized_responses = []
        for _, row in group_df.iterrows():
            try:
                extracted = extract_bool_answer(row["response"])
                if extracted is not None:
                    normalized_responses.append(extracted)
            except ValueError:
                # Handle the case where answer is not recognized
                logger.warning(f"Could not extract boolean answer for task {task_id_val}")
                continue

        if not normalized_responses:
            continue  # no valid responses

        correct_count = sum(
            compare_bool(r, correct_label) for r in normalized_responses
        )
        num_agents = len(normalized_responses)
        max_agents = max(max_agents, num_agents)

        merged_rows.append(
            {
                "task_id": task_id_val,
                "round_number": round_number,
                "correct_count": correct_count,
                "num_agents": num_agents,
            }
        )

    if not merged_rows:
        return pd.DataFrame()

    # Build DataFrame
    df_result = pd.DataFrame(merged_rows)
    if df_result.empty:
        return df_result

    # Now we create bin columns [0..max_agents]
    bin_labels = [str(i) for i in range(max_agents + 1)]
    for bin_label in bin_labels:
        df_result[bin_label] = (df_result["correct_count"] == int(bin_label)).astype(
            int
        )

    # Drop the raw counts
    df_result.drop(columns=["correct_count", "num_agents"], inplace=True)
    return df_result


def calculate_correct_rate_distribution(
    df_answers: pd.DataFrame,
    df_debates: pd.DataFrame,
    max_rounds: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aggregate correct-rate distribution across all rounds found in df_debates.

    This replicates your old logic of summing bin columns, but now we do it by
    reading from the single debate_rounds CSV DataFrame (df_debates).

    Args:
        df_answers: DataFrame with columns ["id", "answer"] (the correct labels).
        df_debates: DataFrame from debate_rounds.csv
                    columns: ["task_id", "round_number", "agent_index", "agent_id", "model", "response"]
        max_rounds: if provided, limit to [0..max_rounds-1], else use all found

    Returns:
        DataFrame aggregated by round, with columns:
           [round_number, 0, 1, 2, ..., total_tasks]
    """
    # 1) Identify all round_numbers in df_debates
    unique_rounds = sorted(df_debates["round_number"].unique())
    if max_rounds is not None:
        unique_rounds = [r for r in unique_rounds if r < max_rounds]

    aggregated_rows = []

    for rnum in unique_rounds:
        df_round = calculate_correct_rate_distribution_for_round_n(
            df_answers, df_debates, rnum
        )
        if df_round.empty:
            continue

        # Identify bin columns
        bin_cols = [c for c in df_round.columns if c.isdigit()]
        bin_cols.sort(key=int)

        # We drop "task_id" and "round_number" before summing
        tmp = df_round.drop(columns=["task_id", "round_number"])

        # Sum the bins across tasks
        aggregated_row = {"round_number": rnum}
        for bc in bin_cols:
            aggregated_row[bc] = tmp[bc].sum()

        aggregated_row["total_tasks"] = len(df_round)
        aggregated_rows.append(aggregated_row)

    if not aggregated_rows:
        return pd.DataFrame()

    df_combined = pd.DataFrame(aggregated_rows)
    return df_combined


def main():
    import sys

    # Hardcoded paths for this example
    answers_csv = "output/bool_q/processed_data.csv"  # your "id" -> "answer" file
    debates_csv = "data/bool_q/llama3(11)/debate_rounds.csv"  # the merged CSV
    max_rounds = None  # or an int

    # 1) Load the "answer" DataFrame
    try:
        df_answers = pd.read_csv(answers_csv)
        # df_answers has columns like ["id", "answer"]
        # Make sure "id" is numeric
        df_answers["id"] = pd.to_numeric(df_answers["id"], errors="coerce")
        df_answers.dropna(subset=["id"], inplace=True)
        df_answers["id"] = df_answers["id"].astype(int)
        logger.info(f"Loaded answers from {answers_csv}")
    except Exception as e:
        logger.error(f"Failed to load {answers_csv}: {e}")
        sys.exit(1)

    # 2) Load the merged debate CSV
    try:
        df_debates = pd.read_csv(debates_csv)
        # columns: "task_id","round_number","agent_index","agent_id","model","response"
        # ensure task_id, round_number are int
        df_debates["task_id"] = pd.to_numeric(df_debates["task_id"], errors="coerce")
        df_debates["round_number"] = pd.to_numeric(
            df_debates["round_number"], errors="coerce"
        )
        df_debates.dropna(subset=["task_id", "round_number"], inplace=True)
        df_debates["task_id"] = df_debates["task_id"].astype(int)
        df_debates["round_number"] = df_debates["round_number"].astype(int)
        logger.info(f"Loaded debates from {debates_csv}")
    except Exception as e:
        logger.error(f"Failed to load {debates_csv}: {e}")
        sys.exit(1)

    # 3) Calculate the distribution
    logger.info(
        f"Calculating distribution from merged CSV, max_rounds={max_rounds} ..."
    )
    df_distribution = calculate_correct_rate_distribution(
        df_answers, df_debates, max_rounds=max_rounds
    )

    if df_distribution.empty:
        logger.warning("No distribution data produced.")
    else:
        print("\nAggregated distribution across rounds:")
        print(df_distribution)
        # for each row, you can do further analysis or write to CSV, etc.


if __name__ == "__main__":
    main()
