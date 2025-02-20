from pathlib import Path
from typing import List, Optional

import pandas as pd

from ...utils.download_dataset import load_save_dataset_df
from ...utils.model_config import ModelConfig
from ...utils.progress import progress
from ..shared.utils import format_config_overview
from .evaluate import evaluate_all_bool_q
from .run_debate import run_debate_bool_q
from .utils import process_bool_q_df
from ..shared.run import run as shared_run


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
    """Execute boolean question evaluation with the given configuration."""
    shared_run(
        dataframe=dataframe,
        run_debate_fn=run_debate_bool_q,
        evaluate_fn=evaluate_all_bool_q,
        process_df_fn=process_bool_q_df,
        task_name="bool_q",
        sample_size=sample_size,
        report_path=report_path,
        model_configs=model_configs,
        random_seed=random_seed,
        max_workers=max_workers,
    )




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

        config_overview = format_config_overview(model_configs_list)
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
