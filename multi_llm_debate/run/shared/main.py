import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .run import execute_debate_workflow
from .utils import setup_logging

logger = setup_logging(__name__)
logger.setLevel("INFO")


def main(
    dataframe: pd.DataFrame,
    run_debate_fn: Callable[..., Dict],
    evaluate_fn: Callable[..., Any],
    process_df_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    task_name: str = "debate",
    sample_size: Optional[int] = None,
    config: Optional[Union[Path, List[Dict]]] = None,
    config_json: Optional[str] = None,
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
        config: Path to JSON config file or list of model configurations
        config_json: JSON string containing model configurations
        random_seed: Random seed for sampling
        run_debate: Whether to run the debate or just evaluate existing results
    """

    try:
        # Priority: 1. config_json, 2. config as list, 3. config as file path
        if config_json is not None:
            try:
                model_configs_list = json.loads(config_json)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string provided in config_json")
        # Check if config is a list (direct configuration)
        elif isinstance(config, list):
            model_configs_list = config
        else:
            # Use provided config path or default to config.json in task directory
            if config is None:
                config = Path(f"multi_llm_debate/run/{task_name}/config.json")

            # Load configuration from file
            with open(config) as f:
                model_configs_list = json.load(f)

        # Adjust sample size if it exceeds dataset size
        if sample_size is not None and sample_size > len(dataframe):
            logger.warning(
                f"Sample size {sample_size} is larger than dataset size {len(dataframe)}. "
                "Using entire dataset."
            )
            sample_size = None

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
                )

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in configuration file at {config}")
