import csv
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ...debate.agents_ensemble import AgentsEnsemble
from ...debate.debate import debate
from ...llm.prompt_builder import PromptBuilder
from ...utils.model_config import ModelConfig
from ...utils.progress import progress
from .evaluate import EvaluationResults
from .utils import format_time, model_configs_to_string

logger = logging.getLogger(__name__)


def execute_debate_workflow(
    dataframe: pd.DataFrame,
    run_debate_fn: Callable,
    evaluate_fn: Callable,
    task_name: str = "default_task",
    report_path: Path = Path("data"),
    model_configs: List[ModelConfig] = [
        {
            "name": "llama3",
            "quantity": 6,
        }
    ],
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Callable = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
) -> Dict[str, Any]:
    """Execute debate evaluation with the given configuration.

    Args:
        dataframe: The input dataframe containing the dataset
        run_debate_fn: Function to run the debate (e.g., run_debate_bool_q)
        evaluate_fn: Function to evaluate results
        process_df_fn: Optional function to preprocess the dataframe
        task_name: Name of the task for logging
        report_path: Path to save results
        model_configs: List of model configurations
        temperature: Temperature for model responses
        max_tokens: Maximum tokens for model responses
        batch: Whether to run in batch mode
        batch_size: Size of the batch.
        quality_pruning_func: Optional function for quality pruning
        quality_pruning_amount: int = 5,
        diversity_pruning_func: Optional function for diversity pruning
        diversity_pruning_amount: int = 5,
        api_key: Optional API key for LLM providers

    Returns:
        Dict containing execution results and evaluation metrics
    """
    start_time = time.time()

    model_config_str = model_configs_to_string(model_configs)
    output_path = report_path / model_config_str.replace(" ", "_").replace(
        ".", "_"
    ).replace("/", "_")
    logger.info(f"Starting {task_name} task with {model_config_str}")

    # Run the debate task
    logger.info(f"Executing debate function for {task_name}")

    execution_report = run_debate_fn(
        dataframe=dataframe,
        base_dir=output_path,
        model_configs=model_configs,
        temperature=temperature,
        max_tokens=max_tokens,
        batch=batch,
        batch_size=batch_size,
        quality_pruning_func=quality_pruning_func,
        quality_pruning_amount=quality_pruning_amount,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
    )

    # Print execution summary
    print(f"\nExecution Summary for {task_name}:")
    print("-" * 50)
    print(f"Total entries processed: {execution_report['total_entries']}")
    print(f"Successfully processed: {execution_report['processed_count']}")
    print(f"Failed entries: {len(execution_report['failed_entries'])}")
    print(f"Success rate: {execution_report['success_rate']:.2f}%")

    # Evaluate using provided evaluation function
    logger.info("Running evaluation")
    try:
        results: EvaluationResults = evaluate_fn(output_path, dataframe)

        # Calculate running time
        running_time = time.time() - start_time
        display_time, csv_time = format_time(running_time)
        print(f"\nTotal running time: {display_time}")

        # Save results to CSV
        report_path.mkdir(parents=True, exist_ok=True)
        csv_path = report_path / "results.csv"
        logger.info(f"Saving results to {csv_path}")

        # Read existing data if file exists
        existing_data = []
        if csv_path.exists():
            try:
                with open(csv_path, "r", newline="") as f:
                    reader = csv.reader(f)
                    existing_data = list(reader)
            except Exception as e:
                logger.error(f"Error reading existing CSV: {str(e)}")
                existing_data = []

        current_config = model_configs_to_string(model_configs)
        new_row = [
            current_config,
            task_name,  # Add task name to the results row
            f"{results.single_llm_accuracy:.4f}",
            f"{results.single_llm_error:.4f}",
            f"{results.ensemble_accuracy:.4f}",
            f"{results.ensemble_error:.4f}",
            f"{results.debate_accuracy:.4f}",
            f"{results.debate_error:.4f}",
            csv_time,
        ]

        if not existing_data:
            existing_data = [
                [
                    "Model Configuration",
                    "Task Name",  # Add Task Name column header
                    "Single LLM Accuracy",
                    "Single LLM Error Margin",
                    "Ensemble Accuracy",
                    "Ensemble Error Margin",
                    "Debate Accuracy",
                    "Debate Error Margin",
                    "Running Time",
                ]
            ]

        # Update existing entry or append new one
        found = False
        for i, row in enumerate(existing_data[1:], 1):
            if row and row[0] == current_config:
                existing_data[i] = new_row
                found = True
                break
        if not found:
            existing_data.append(new_row)

        # Write all data back to CSV
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(existing_data)
            print(f"\nResults saved to {csv_path}")
        except Exception as e:
            logger.error(f"Error writing results to CSV: {str(e)}")
            print(f"\nFailed to save results: {str(e)}")

        return {
            "execution_report": execution_report,
            "evaluation_results": results,
            "running_time": running_time,
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        print(f"\nEvaluation failed: {str(e)}")
        running_time = time.time() - start_time
        display_time, _ = format_time(running_time)
        print(f"\nTotal running time: {display_time}")

        return {
            "execution_report": execution_report,
            "evaluation_results": None,
            "error": str(e),
            "running_time": running_time,
        }


def process_debate_dataset(
    dataframe: pd.DataFrame,
    process_entry_fn: Callable,
    required_columns: List[str],
    base_dir: Path,
    max_rounds: int = 10,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    task_name: str = "default_task",
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Callable = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
) -> Dict[str, Any]:
    """Run a debate task on all entries in a dataframe.

    Args:
        dataframe: Pandas DataFrame containing entries to process
        process_entry_fn: Function to process a single entry
        required_columns: List of column names required in the dataframe
        base_dir: Base directory for output files
        max_rounds: Maximum number of debate rounds
        model_configs: Optional list of model configurations
        overwrite: Whether to overwrite existing debate results
        task_name: Name of the task for logging purposes
        temperature: Temperature for model responses
        max_tokens: Maximum tokens for model responses
        batch: bool = False,
        batch_size: int = 11,
        quality_pruning_func: Optional function for quality pruning
        quality_pruning_amount: int = 5,
        diversity_pruning_func: Optional function for diversity pruning
        diversity_pruning_amount: int = 5,
        api_key: Optional API key for LLM providers

    Returns:
        Dict containing summary of execution including failed entries

    Raises:
        ValueError: If DataFrame format is invalid
    """
    failed_entries = []
    processed_count = 0

    try:
        logger.info(f"Starting debate for {task_name}")

        # Check if the DataFrame is valid
        if not isinstance(dataframe, pd.DataFrame):
            logger.error("Invalid DataFrame type")
            raise ValueError("Dataframe must be a pandas DataFrame.")

        missing_columns = [
            col for col in required_columns if col not in dataframe.columns
        ]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        if dataframe.empty:
            logger.error("DataFrame is empty")
            raise ValueError("DataFrame is empty. Please provide valid data.")

        # Using progress manager for the main progress bar
        with progress.main_bar(
            total=len(dataframe), desc=f"Running {task_name}", unit="task"
        ) as pbar:
            # Always process sequentially
            for _, entry in dataframe.iterrows():
                try:
                    process_entry_fn(
                        entry=entry,
                        max_rounds=max_rounds,
                        base_dir=base_dir,
                        model_configs=model_configs,
                        overwrite=overwrite,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        batch=batch,
                        batch_size=batch_size,
                        quality_pruning_func=quality_pruning_func,
                        quality_pruning_amount=quality_pruning_amount,
                        diversity_pruning_func=diversity_pruning_func,
                        diversity_pruning_amount=diversity_pruning_amount,
                    )
                    processed_count += 1
                except Exception as e:
                    entry_id = entry.get("id", "unknown")
                    logger.error(
                        f"Error processing entry {entry_id}: {str(e)}", exc_info=True
                    )
                    failed_entries.append(
                        {
                            "id": entry_id,
                            "error": str(e),
                            "question": entry.get("question", ""),
                        }
                    )
                finally:
                    pbar.update(1)

    except Exception as e:
        logger.error(f"Global execution error: {str(e)}", exc_info=True)
        raise RuntimeError(f"Global execution error: {str(e)}") from e

    finally:
        # Log summary
        total_entries = len(dataframe)
        failed_count = len(failed_entries)
        success_rate = (
            (processed_count / total_entries) * 100 if total_entries > 0 else 0
        )

        logger.info("Debate execution completed")
        logger.info(f"Total entries processed: {total_entries}")
        logger.info(f"Successful: {processed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Success rate: {success_rate:.2f}%")

        if failed_entries:
            logger.warning("Failed entries:")
            for entry in failed_entries:
                logger.warning(f"ID: {entry['id']}, Error: {entry['error']}")

        if len(failed_entries) == total_entries and total_entries > 0:
            logger.error(f"All {total_entries} entries failed. Check logs for details.")

    return {
        "total_entries": total_entries,
        "processed_count": processed_count,
        "failed_entries": failed_entries,
        "success_rate": success_rate,
    }


def process_single_debate_entry(
    entry: pd.Series,
    required_columns: List[str],
    base_dir: Path,
    max_rounds: int,
    model_configs: Optional[List[ModelConfig]],
    overwrite: bool,
    prompt_builder_fn: Callable[..., PromptBuilder],
    prompt_params: Dict[str, Any],
    extract_func: Optional[Callable[..., Any]] = None,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Callable = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
) -> None:
    """Run a single entry's debate logic in a generic way.

    Args:
        entry: A single row from the dataframe.
        required_columns: Columns required to run the debate.
        base_dir: Base directory for output files.
        max_rounds: Maximum debate rounds.
        use_cot: Whether to use chain-of-thought prompting.
        model_configs: List of model configs or None for defaults.
        overwrite: Whether to overwrite existing files.
        prompt_builder_fn: Function returning a prompt builder.
        prompt_params: Parameters used to build prompts.
        extract_func: Optional function for post-processing responses.
        temperature: Temperature for model responses.
        max_tokens: Maximum tokens for model responses.
        batch: Whether to run in batch mode.
        batch_size: Size of the batch.
        quality_pruning_func: Optional function for quality pruning.
        quality_pruning_amount: Amount for pruning quality.
        diversity_pruning_func: Optional function for diversity pruning.
        diversity_pruning_amount: Amount for pruning diversity.
        api_key: Optional API key for LLM providers.

    Raises:
        ValueError: If required columns are missing.
        RuntimeError: If debate execution fails.
    """

    def is_missing(val):
        # Check for missing scalar or empty/fully-NA array-like
        if isinstance(val, (list, np.ndarray, pd.Series)):
            if len(val) == 0:
                return True
            return all(pd.isna(x) for x in val)
        return pd.isna(val)

    missing_cols = [
        c for c in required_columns if c not in entry or is_missing(entry[c])
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    entry_id = str(entry.get("id", "unknown"))
    output_dir = base_dir / entry_id

    # Check if directory exists and has content
    directory_has_content = (
        output_dir.exists() and any(output_dir.iterdir())
        if output_dir.exists()
        else False
    )

    # Skip if directory has content and overwrite is False
    if directory_has_content and not overwrite:
        logger.info(f"Skipping entry {entry_id} as data already exists.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_builder = prompt_builder_fn(prompt_params=prompt_params)
    agents_ensemble = AgentsEnsemble(config_list=model_configs)

    debate(
        max_rounds=max_rounds,
        prompt_builder=prompt_builder,
        agents_ensemble=agents_ensemble,
        output_dir=output_dir,
        extract_func=extract_func,
        temperature=temperature,
        max_tokens=max_tokens,
        batch=batch,
        batch_size=batch_size,
        quality_pruning_func=quality_pruning_func,
        quality_pruning_amount=quality_pruning_amount,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
    )
