from pathlib import Path
from typing import List

from ...utils.download_dataset import load_save_dataset_df
from ...utils.model_config import ModelConfig
from .evaluate import evaluate_baseline_df, evaluate_df
from .run import run_bool_q
from .utils import process_bool_q_df

MODEL_CONFIGS_LIST = [
    [
        {
            "provider": "ollama",
            "name": "llama3",
            "quantity": 6,
        }
    ],
    [
        {
            "provider": "ollama",
            "name": "llama2",
            "quantity": 6,
        }
    ],
    [
        {
            "provider": "ollama",
            "name": "mistral",
            "quantity": 6,
        }
    ],
    [
        {
            "provider": "ollama",
            "name": "llama3",
            "quantity": 3,
        },
        {
            "provider": "ollama",
            "name": "llama2",
            "quantity": 3,
        },
    ],
    [
        {
            "provider": "ollama",
            "name": "llama3",
            "quantity": 3,
        },
        {
            "provider": "ollama",
            "name": "mistral",
            "quantity": 3,
        },
    ],
    [
        {
            "provider": "ollama",
            "name": "llama2",
            "quantity": 3,
        },
        {
            "provider": "ollama",
            "name": "mistral",
            "quantity": 3,
        },
    ],
]


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
) -> None:
    # Load the dataset
    dataset_path = Path("datasets/boolq")
    output_path = Path("data/bool_q/llama3")
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

    # Save the execution report
    report_path.mkdir(parents=True, exist_ok=True)
