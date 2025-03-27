import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from .run import execute_debate_workflow
from .utils import setup_logging

logger = setup_logging(__name__)


def main(
    dataframe: pd.DataFrame,
    run_debate_fn: Callable[..., Dict],
    evaluate_fn: Callable[..., Any],
    process_df_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    task_name: str = "debate",
    sample_size: Optional[int] = None,
    max_workers: Optional[int] = 4,
    config_path: Optional[Path] = None,
    random_seed: Optional[int] = None,
    run_debate: bool = True,
) -> None:
    """Run debate evaluation with configured models.

    Args:
        dataframe: Input DataFrame containing the debate data
        run_debate_fn: Function to run debates
        evaluate_fn: Function to evaluate debate results
        process_df_fn: Optional function to preprocess DataFrame. If None,
            the dataframe will be used without preprocessing.
        task_name: Name of the debate task
        sample_size: Optional number of samples to process
        max_workers: Maximum number of concurrent workers
        config_path: Path to JSON config file
        random_seed: Random seed for sampling
        run_debate: Whether to run the debate or just evaluate existing results
        **debate_kwargs: Additional arguments to pass to run_debate_fn
    """

    try:
        # Use provided config path or default to config.json in task directory
        if config_path is None:
            config_path = Path(f"multi_llm_debate/run/{task_name}/config.json")

        with open(config_path) as f:
            model_configs_list = json.load(f)

        # Adjust sample size if it exceeds dataset size
        if sample_size is not None and sample_size > len(dataframe):
            logger.warning(
                f"Sample size {sample_size} is larger than dataset size {len(dataframe)}. "
                "Using entire dataset."
            )
            sample_size = None

        # Create output directory if it doesn't exist
        output_dir = Path(f"output/{task_name}")
        # output_dir.mkdir(parents=True, exist_ok=True)

        if run_debate:
            for model_configs in model_configs_list:
                execute_debate_workflow(
                    dataframe=dataframe,
                    run_debate_fn=run_debate_fn,
                    evaluate_fn=evaluate_fn,
                    process_df_fn=process_df_fn,
                    task_name=task_name,
                    sample_size=sample_size,
                    report_path=Path(f"data/{task_name}"),
                    model_configs=model_configs,
                    random_seed=random_seed,
                    max_workers=max_workers,
                )

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
