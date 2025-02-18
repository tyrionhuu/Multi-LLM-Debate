import csv
import time
from pathlib import Path
from typing import List

from ...utils.download_dataset import load_save_dataset_df
from ...utils.model_config import ModelConfig
from ...utils.progress import progress
from .evaluate import evaluate_baseline_df, evaluate_df
from .run import run_bool_q
from .utils import format_time, model_configs_to_string, process_bool_q_df


def run(
    test: bool = False,
    sample_size: int = 20,
    report_path: Path = Path("data/bool_q"),
    model_configs: List[ModelConfig] = [
        {
            "provider": "ollama",
            "name": "llama3",
            "quantity": 6,
        }
    ],
    show_progress: bool = True,  # Add this parameter
) -> None:
    start_time = time.time()

    # Load the dataset
    dataset_path = Path("datasets/boolq")
    model_config_str = model_configs_to_string(model_configs)
    output_path = report_path / model_config_str.replace(" ", "_")

    dataframe = load_save_dataset_df(
        dataset_name="google/boolq",
        dataset_path=dataset_path,
        force_download=False,
    )

    # Process the DataFrame
    processed_dataframe = process_bool_q_df(dataframe)
    if test:
        # processed_dataframe = processed_dataframe.iloc[[3]]
        processed_dataframe = processed_dataframe.sample(sample_size)

    # Run the Boolean Question task
    execution_report = run_bool_q(
        dataframe=processed_dataframe,
        base_dir=output_path,
        model_configs=model_configs,
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
    baseline_accuracy = evaluate_baseline_df(output_path, processed_dataframe)
    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}")

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
                    "Baseline Accuracy",
                    "Debate Accuracy",
                    "Running Time",
                ]
            )
        writer.writerow(
            [
                model_configs_to_string(model_configs),
                f"{baseline_accuracy:.4f}",
                f"{accuracy:.4f}",
                csv_time,
            ]
        )

    print(f"\nResults saved to {csv_path}")


def main(test: bool = False) -> None:
    """Run the boolean question evaluation with configured models.

    Args:
        test (bool): Whether to run in test mode. Defaults to False.
    """
    import json

    try:
        config_path = Path(__file__).parent / "config.json"
        with open(config_path) as f:
            model_configs_list = json.load(f)

        # Add progress bar for model configurations
        with progress.main_bar(
            total=len(model_configs_list),
            desc="Testing model configurations",
            unit="config"
        ) as pbar:
            for model_configs in model_configs_list:
                run(
                    test=test,
                    sample_size=1,
                    report_path=Path("data/bool_q"),
                    model_configs=model_configs,
                    show_progress=True,  # Enable progress tracking
                )
                pbar.update(1)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")


if __name__ == "__main__":
    main(test=False)
