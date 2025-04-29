import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .run import execute_debate_workflow

logger = logging.getLogger(__name__)


def main(
    dataframe: pd.DataFrame,
    run_debate_fn: Callable[..., Dict],
    evaluate_fn: Callable[..., Any],
    process_df_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    task_name: str = "debate",
    config: Optional[Union[Path, List[Dict]]] = None,
    config_json: Optional[str] = None,
    run_debate: bool = True,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Callable = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
) -> None:
    """Run debate evaluation with configured models.

    Args:
        dataframe: Input DataFrame containing the debate data
        run_debate_fn: Function to run debates
        evaluate_fn: Function to evaluate debate results
        process_df_fn: Optional function to preprocess DataFrame. If None,
            the dataframe will be used without preprocessing.
        task_name: Name of the debate task
        config: Path to JSON config file or list of model configurations
        config_json: JSON string containing model configurations
        run_debate: Whether to run the debate or just evaluate existing results
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        batch: Whether to run in batch mode
        batch_size: Size of the batch.
        quality_pruning_func: Function for quality pruning
        quality_pruning_amount: Amount of pruning to apply
        diversity_pruning_func: Function for diversity pruning
        diversity_pruning_amount: Amount of pruning to apply
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

        if run_debate:
            for model_configs in model_configs_list:
                execute_debate_workflow(
                    dataframe=dataframe,
                    run_debate_fn=run_debate_fn,
                    evaluate_fn=evaluate_fn,
                    task_name=task_name,
                    report_path=Path(f"data/{task_name}"),
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

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in configuration file at {config}")
