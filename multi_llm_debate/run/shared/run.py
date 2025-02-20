import csv
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ...utils.model_config import ModelConfig
from .evaluate import EvaluationResults
from .utils import format_time, model_configs_to_string


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
) -> None:
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
        None: Results are saved to files and printed to console
    """
    start_time = time.time()

    model_config_str = model_configs_to_string(model_configs)
    output_path = report_path / model_config_str.replace(" ", "_")

    # Process the DataFrame if needed
    if process_df_fn:
        processed_dataframe = process_df_fn(dataframe)
    else:
        processed_dataframe = dataframe

    if sample_size:
        processed_dataframe = processed_dataframe.sample(
            sample_size, random_state=random_seed
        )

    # Run the debate task
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

    # Evaluate using provided evaluation function
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
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Task",
                    "Model Configuration",
                    "Single LLM Accuracy",
                    "Ensemble Accuracy",
                    "Debate Accuracy",
                    "Running Time",
                ]
            )
        writer.writerow(
            [
                task_name,
                model_configs_to_string(model_configs),
                "N/A" if multiple_models else f"{results.single_llm_accuracy:.4f}",
                f"{results.ensemble_accuracy:.4f}",
                f"{results.debate_accuracy:.4f}",
                csv_time,
            ]
        )

    print(f"\nResults saved to {csv_path}")


