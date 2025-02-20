from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from .run import run


def main(
    dataframe: pd.DataFrame,
    run_debate_fn: Callable[..., Any],
    evaluate_fn: Callable[..., Any],
    process_df_fn: Callable[..., pd.DataFrame],
    task_name: str,
    sample_size: Optional[int] = None,
    max_workers: Optional[int] = 4,
    config_path: Optional[Path] = None,
) -> None:
    """Run debate evaluation with configured models.

    Args:
        dataframe: Input DataFrame containing the debate data
        run_debate_fn: Function to run debates
        evaluate_fn: Function to evaluate debate results
        process_df_fn: Function to preprocess DataFrame
        task_name: Name of the debate task
        sample_size: Optional number of samples to process
        max_workers: Maximum number of concurrent workers
        config_path: Path to JSON config file
    """
    import json

    try:
        # Use provided config path or default to config.json in task directory
        if config_path is None:
            config_path = Path(f"multi_llm_debate/run/{task_name}/config.json")

        with open(config_path) as f:
            model_configs_list = json.load(f)

        # Process all configurations
        processed_df = process_df_fn(dataframe)
        for model_configs in model_configs_list:
            run(
                dataframe=processed_df,
                run_debate_fn=run_debate_fn,
                evaluate_fn=evaluate_fn,
                task_name=task_name,
                sample_size=sample_size,
                report_path=Path(f"data/{task_name}"),
                model_configs=model_configs,
                max_workers=max_workers,
            )

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
