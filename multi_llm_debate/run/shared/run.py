import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from .evaluate import EvaluationResults
from .utils import format_time, model_configs_to_string

logger = setup_logging(__name__)


def run(
    dataframe: pd.DataFrame,
    run_debate_fn: Callable[[pd.DataFrame, Path, List[ModelConfig], Any], Dict],
    evaluate_fn: Callable[[Path, pd.DataFrame, bool], Any],
    process_df_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    task_name: str = "debate",
    sample_size: Optional[int] = None,
    report_path: Path = Path("data"),
    model_configs: List[ModelConfig] = [
        {
            "provider": "ollama",
            "name": "llama3",
            "quantity": 6,
        }
    ],
    random_seed: int = 42,
    max_workers: Optional[int] = 4,
    **debate_kwargs: Any,
) -> Dict[str, Any]:
    """Execute debate evaluation with the given configuration.

    Args:
        dataframe: The input dataframe containing the dataset
        run_debate_fn: Function to run the debate (e.g., run_debate_bool_q)
        evaluate_fn: Function to evaluate results
        process_df_fn: Optional function to preprocess the dataframe
        task_name: Name of the task for logging
        sample_size: Optional number of samples to use
        report_path: Path to save results
        model_configs: List of model configurations
        random_seed: Random seed for sampling
        max_workers: Maximum number of concurrent workers
        **debate_kwargs: Additional arguments to pass to run_debate_fn

    Returns:
        Dict containing execution results and evaluation metrics
    """
    start_time = time.time()

    model_config_str = model_configs_to_string(model_configs)
    output_path = report_path / model_config_str.replace(" ", "_")
    logger.info(f"Starting {task_name} task with {model_config_str}")

    # Process the DataFrame if needed
    if process_df_fn:
        logger.info("Preprocessing input dataframe")
        processed_dataframe = process_df_fn(dataframe)
    else:
        processed_dataframe = dataframe

    if sample_size and len(processed_dataframe) > sample_size:
        logger.info(
            f"Sampling {sample_size} entries from dataset (random seed: {random_seed})"
        )
        processed_dataframe = processed_dataframe.sample(
            sample_size, random_state=random_seed
        )

    # Run the debate task
    logger.info(f"Executing debate function for {task_name}")
    execution_report = run_debate_fn(
        dataframe=processed_dataframe,
        base_dir=output_path,
        model_configs=model_configs,
        max_workers=max_workers,
        **debate_kwargs,
    )

    # Print execution summary
    print(f"\nExecution Summary for {task_name}:")
    print("-" * 50)
    print(f"Total entries processed: {execution_report['total_entries']}")
    print(f"Successfully processed: {execution_report['processed_count']}")
    print(f"Failed entries: {len(execution_report['failed_entries'])}")
    print(f"Success rate: {execution_report['success_rate']:.2f}%")

    # Check if we have multiple model types
    model_types = {(config["provider"], config["name"]) for config in model_configs}
    multiple_models = len(model_types) > 1
    logger.info(f"Multiple model types detected: {multiple_models}")

    # Evaluate using provided evaluation function
    logger.info("Running evaluation")
    try:
        results: EvaluationResults = evaluate_fn(
            output_path, processed_dataframe, multiple_models=multiple_models
        )

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
            "N/A" if multiple_models else f"{results.single_llm_accuracy:.4f}",
            (
                f"{results.ensemble_accuracy:.4f}"
                if hasattr(results, "ensemble_accuracy")
                else "N/A"
            ),
            (
                f"{results.debate_accuracy:.4f}"
                if hasattr(results, "debate_accuracy")
                else "N/A"
            ),
            csv_time,
        ]

        if not existing_data:
            # Create new file with headers
            existing_data = [
                [
                    "Model Configuration",
                    "Single LLM Accuracy",
                    "Ensemble Accuracy",
                    "Debate Accuracy",
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


def run_debate_task(
    dataframe: pd.DataFrame,
    process_entry_fn: Callable[[pd.Series, int, Path, bool, Optional[List[ModelConfig]], 
                               bool, Optional[int]], None],
    required_columns: List[str],
    base_dir: Path,
    max_rounds: int = 10,
    use_cot: bool = True,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
    task_name: str = "debate task",
) -> Dict[str, Any]:
    """Run a debate task on all entries in a dataframe.

    Args:
        dataframe: Pandas DataFrame containing entries to process
        process_entry_fn: Function to process a single entry
        required_columns: List of column names required in the dataframe
        base_dir: Base directory for output files
        max_rounds: Maximum number of debate rounds
        use_cot: Whether to use chain-of-thought prompting
        model_configs: Optional list of model configurations
        overwrite: Whether to overwrite existing debate results
        max_workers: Maximum number of concurrent workers
        task_name: Name of the task for logging purposes

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
        from ...utils.progress import progress
        with progress.main_bar(
            total=len(dataframe), desc=f"Running {task_name}", unit="debate"
        ) as pbar:
            for _, entry in dataframe.iterrows():
                try:
                    process_entry_fn(
                        entry,
                        max_rounds=max_rounds,
                        base_dir=base_dir,
                        use_cot=use_cot,
                        model_configs=model_configs,
                        overwrite=overwrite,
                        max_workers=max_workers,
                    )
                    processed_count += 1
                    pbar.update(1)
                except Exception as e:
                    entry_id = entry.get("id", "unknown")
                    logger.error(f"Error processing entry {entry_id}: {str(e)}")
                    failed_entries.append({
                        "id": entry_id,
                        "error": str(e),
                        "question": entry.get("question", ""),
                    })
                    pbar.update(1)  # Update progress even for failures
                    continue

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
