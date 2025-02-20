import csv
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ...utils.download_dataset import load_save_dataset_df
from ...utils.model_config import ModelConfig
from ...utils.progress import progress
from .evaluate import evaluate_df, evaluate_ensemble_df, evaluate_single_llm_df
from .run import run_bool_q
from .utils import model_configs_to_string, process_bool_q_df
from ..utils import format_time

def run(
    dataframe: pd.DataFrame,
    sample_size: Optional[int] = None,
    report_path: Path = Path("data/bool_q"),
    model_configs: List[ModelConfig] = [
        {
            "provider": "ollama",
            "name": "llama3",
            "quantity": 6,
        }
    ],
    random_seed: int = 42,
    max_workers: Optional[int] = 4,
) -> None:
    """Execute boolean question evaluation with the given configuration.

    Args:
        dataframe: The input dataframe containing boolean questions dataset.
        test (bool, optional): Whether to run in test mode. Defaults to False.
        sample_size (int, optional): Number of samples to use in test mode.
            Defaults to 20.
        report_path (Path, optional): Path to save results.
            Defaults to Path("data/bool_q").
        model_configs (List[ModelConfig], optional): List of model configurations.
            Defaults to single Ollama config.
        random_seed (int, optional): Random seed for sampling. Defaults to 42.
        max_workers (int, optional): Maximum number of concurrent workers.
            Defaults to 4.

    Returns:
        None: Results are saved to files and printed to console.
    """
    start_time = time.time()

    model_config_str = model_configs_to_string(model_configs)
    output_path = report_path / model_config_str.replace(" ", "_")

    # Process the DataFrame
    processed_dataframe = process_bool_q_df(dataframe)
    if sample_size:
        processed_dataframe = processed_dataframe.sample(
            sample_size, random_state=random_seed
        )

    # Run the Boolean Question task
    execution_report = run_bool_q(
        dataframe=processed_dataframe,
        base_dir=output_path,
        model_configs=model_configs,
        max_workers=max_workers,
    )

    # Print execution summary
    print("\nExecution Summary:")
    print("-" * 50)
    print(f"Total entries processed: {execution_report['total_entries']}")
    print(f"Successfully processed: {execution_report['processed_count']}")
    print(f"Failed entries: {len(execution_report['failed_entries'])}")
    print(f"Success rate: {execution_report['success_rate']:.2f}%")

    # Evaluate the results
    accuracy = evaluate_df(output_path, processed_dataframe)

    # Only calculate single LLM accuracy when there's one type of model
    single_llm_accuracy = None
    if len(model_configs) == 1:
        single_llm_accuracy = evaluate_single_llm_df(output_path, processed_dataframe)

    ensemble_accuracy = evaluate_ensemble_df(output_path, processed_dataframe)
    print(f"\nAccuracy: {accuracy:.2f}")
    if single_llm_accuracy is not None:
        print(f"Single LLM Accuracy: {single_llm_accuracy:.2f}")
    print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}")

    # Calculate running time
    running_time = time.time() - start_time
    display_time, csv_time = format_time(running_time)
    print(f"\nTotal running time: {display_time}")

    # Save the execution report
    report_path.mkdir(parents=True, exist_ok=True)

    # Save results to CSV
    csv_path = report_path / "results.csv"
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Model Configuration",
                    "Single LLM Accuracy",
                    "Ensemble Accuracy",
                    "Debate Accuracy",
                    "Running Time",
                ]
            )
        writer.writerow(
            [
                model_configs_to_string(model_configs),
                (
                    f"{single_llm_accuracy:.4f}"
                    if single_llm_accuracy is not None
                    else "N/A"
                ),
                f"{ensemble_accuracy:.4f}",
                f"{accuracy:.4f}",
                csv_time,
            ]
        )

    print(f"\nResults saved to {csv_path}")


def _format_config_overview(model_configs_list: List[List[ModelConfig]]) -> str:
    """Format model configurations for display in progress bar.

    Args:
        model_configs_list: List of model configuration lists

    Returns:
        str: Formatted string showing number of configs and total models
    """
    total_configs = len(model_configs_list)
    total_models = sum(
        sum(config["quantity"] for config in configs) for configs in model_configs_list
    )
    return f"Running {total_configs} configs ({total_models} total models)"


def main(
    sample_size: Optional[int] = None,
    max_workers: Optional[int] = 4,
    config_path: Optional[Path] = None,
) -> None:
    """Run boolean question evaluation with configured models.

    This function loads the dataset and model configurations from a JSON file,
    then runs the evaluation for each model configuration. Progress is tracked
    with a progress bar.

    Args:
        sample_size (Optional[int], optional): Number of samples to use.
            Defaults to None.
        max_workers (Optional[int], optional): Maximum number of concurrent
            workers. Defaults to 4.
        config_path (Optional[Path], optional): Path to JSON config file.
            Defaults to config.json in same directory as script.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    import json

    try:
        # Load the dataset first
        dataset_path = Path("datasets/boolq")
        dataframe = load_save_dataset_df(
            dataset_name="google/boolq",
            dataset_path=dataset_path,
            force_download=False,
        )

        # Use provided config path or default to config.json in script directory
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"

        with open(config_path) as f:
            model_configs_list = json.load(f)

        config_overview = _format_config_overview(model_configs_list)
        # Add progress bar for model configurations
        with progress.main_bar(
            total=len(model_configs_list),
            desc=config_overview,
            unit="config",
        ) as pbar:
            for model_configs in model_configs_list:
                run(
                    dataframe=dataframe,  # Pass the loaded dataframe
                    sample_size=sample_size,
                    report_path=Path("data/bool_q"),
                    model_configs=model_configs,
                    max_workers=max_workers,
                )
                pbar.update(1)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run boolean question evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config JSON file",
        default=None,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of samples to process",
        default=2000,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of concurrent workers",
        default=16,
    )

    args = parser.parse_args()
    main(
        sample_size=args.sample_size,
        max_workers=args.max_workers,
        config_path=args.config,
    )
