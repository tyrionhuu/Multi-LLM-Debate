import logging
from typing import Optional, Callable

import pandas as pd
from tqdm import tqdm


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
    extract_func: Callable = None,
    compare_func: Callable = None,
) -> pd.DataFrame:
    """
    Compute correct-rate distribution for a *requested* round_number.
    For each task, if that round is not present, we *fallback* to the
    highest round found for that task.

    Args:
        df_answers: DataFrame with columns ["id", "answer"].
                    `id` is numeric, `answer` is the correct boolean.
        df_debates: DataFrame from debate_rounds.csv, containing columns:
                    [task_id, round_number, agent_index, agent_id, model, response]
        round_number: The round the user wants to analyze

    Returns:
        DataFrame with columns [task_id, round_number, 0, 1, 2, ...]
        where each row is a single task, and there's exactly one "1" in
        the bin column that matches how many agents were correct.
        Note: If a task didn't have the requested round, we use its
              highest available round. The 'round_number' column in
              the result still shows the *requested* round, but that
              row's data is actually from whichever round was used.
    """
    if extract_func is None:
        raise ValueError("extract_func is required")
    if compare_func is None:
        raise ValueError("compare_func is required")    
    
    # We'll gather results per task
    merged_rows = []

    # All unique tasks in df_debates
    all_task_ids = df_debates["task_id"].unique()

    max_agents = 0

    # Process each task individually
    for task_id_val in tqdm(all_task_ids, desc=f"Round {round_number}", unit="task"):
        # All debate rows for this task
        df_task = df_debates[df_debates["task_id"] == task_id_val]
        if df_task.empty:
            continue  # no data for this task at all

        # Attempt to find the requested round
        df_requested = df_task[df_task["round_number"] == round_number]
        if not df_requested.empty:
            # We can use the requested round
            df_use = df_requested
        else:
            # Fallback: use the maximum round for that task
            max_round_for_this_task = df_task["round_number"].max()
            df_use = df_task[df_task["round_number"] == max_round_for_this_task]

        # Find the correct label for this task
        ans_row = df_answers[df_answers["id"] == task_id_val]
        if ans_row.empty:
            correct_label = None
        else:
            correct_label = ans_row["answer"].iloc[0]

        # Extract booleans from each response
        normalized_responses = []
        for _, row in df_use.iterrows():
            try:
                extracted = extract_func(row["response"])
                if extracted is not None:
                    normalized_responses.append(extracted)
            except ValueError:
                logger.debug(f"Could not extract boolean answer for task {task_id_val}")
                continue

        # Count how many are correct
        if correct_label is None or not normalized_responses:
            correct_count = 0
            num_agents = 0
        else:
            correct_count = sum(
                compare_func(r, correct_label) for r in normalized_responses
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

    # Create bin columns [0..max_agents]
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
    extract_func: Callable = None,
    compare_func: Callable = None,
) -> pd.DataFrame:
    """
    Aggregate correct-rate distribution across all rounds found in df_debates.
    For each round r in [0..], we call `calculate_correct_rate_distribution_for_round_n`.
    If a task doesn't have round r, we fallback to that task's max round.

    Args:
        df_answers: DataFrame with columns ["id", "answer"] (the correct labels).
        df_debates: DataFrame from debate_rounds.csv
                    columns: ["task_id", "round_number", "agent_index", "agent_id", "model", "response"]
        max_rounds: if provided, limit to [0..max_rounds-1], else use all found

    Returns:
        DataFrame aggregated by round, with columns:
           [round_number, 0, 1, 2, ..., total_tasks]
        The `round_number` column indicates which round was *requested*,
        though some tasks might have used their fallback if that round wasn't available.
    """
    if extract_func is None:
        raise ValueError("extract_func is required")
    if compare_func is None:
        raise ValueError("compare_func is required")
    
    # 1) Identify all round_numbers in df_debates
    unique_rounds = sorted(df_debates["round_number"].unique())
    if max_rounds is not None:
        unique_rounds = [r for r in unique_rounds if r < max_rounds]

    aggregated_rows = []

    for rnum in unique_rounds:
        df_round = calculate_correct_rate_distribution_for_round_n(
            df_answers=df_answers,
            df_debates=df_debates,
            round_number=rnum,
            extract_func=extract_func,
            compare_func=compare_func,
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
    from .utils import compare_bool
    from ..run.bool_q.utils import extract_bool_answer

    # Hardcoded paths for this example
    answers_csv = "output/bool_q/processed_data.csv"  # your "id" -> "answer" file
    debates_csv = "data/bool_q/llama3(11)/debate_rounds.csv"  # the merged CSV
    max_rounds = 6

    # 1) Load the "answer" DataFrame
    try:
        df_answers = pd.read_csv(answers_csv)
        # df_answers has columns like ["id", "answer"]
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
        f"Calculating distribution from merged CSV with fallback, max_rounds={max_rounds} ..."
    )
    df_distribution = calculate_correct_rate_distribution(
        df_answers, df_debates, max_rounds=max_rounds
    )

    if df_distribution.empty:
        logger.warning("No distribution data produced.")
    else:
        print(
            "\nAggregated distribution across requested rounds (fallback used if missing):"
        )
        print(df_distribution)
        # e.g., you could write to CSV here:
        # df_distribution.to_csv("output/bool_q/correct_rate_distribution.csv", index=False)


if __name__ == "__main__":
    main()
