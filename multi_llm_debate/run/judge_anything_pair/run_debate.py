import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ...llm.prompt_builder import PromptBuilder
from ...utils.model_config import ModelConfig
from ..shared.run import process_debate_dataset, process_single_debate_entry
from .prompts import (
    build_judge_anything_pair_round_n_prompt,
    build_judge_anything_pair_round_zero_prompt,
)
from .utils import extract_caption_a_b_answer

# Fix the setup_logging call by removing the level parameter
logger = logging.getLogger(__name__)


def process_judge_anything_pair_dataset(
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "judge_anything_pair",
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Callable = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
) -> Dict[str, Any]:
    """Run the JudgeAnythingPair task on a DataFrame.

    Args:
        dataframe: Pandas DataFrame containing question, answer, passage and id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting (default: True)
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        overwrite: Whether to overwrite existing debate results (default: False)
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        batch: Whether to run in batch mode (default: False)
        batch_size: Size of the batch (default: 11)
        quality_pruning_func: Optional function for quality pruning
        quality_pruning_amount: Amount for pruning quality
        diversity_pruning_func: Optional function for diversity pruning
        diversity_pruning_amount: Amount for pruning diversity

    Returns:
        Dict containing summary of execution including failed entries

    Raises:
        ValueError: If DataFrame format is invalid
    """
    required_columns = ["question", "response_A", "response_B", "id"]

    return process_debate_dataset(
        dataframe=dataframe,
        process_entry_fn=process_judge_anything_pair_entry,
        required_columns=required_columns,
        base_dir=base_dir,
        max_rounds=max_rounds,
        model_configs=model_configs,
        overwrite=overwrite,
        task_name="JudgeAnything pair task",
        temperature=temperature,
        max_tokens=max_tokens,
        batch=batch,
        batch_size=batch_size,
        quality_pruning_func=quality_pruning_func,
        quality_pruning_amount=quality_pruning_amount,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
    )


def process_judge_anything_pair_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "judge_anything_pair",
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Callable = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
) -> None:
    """Process a single entry for the JudgeAnythingPair task.

    Args:
        entry: A Pandas Series containing the entry data
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        overwrite: Whether to overwrite existing debate results (default: False)
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        batch: Whether to run in batch mode (default: False)
        batch_size: Size of the batch (default: 11)
        quality_pruning_func: Optional function for quality pruning
        quality_pruning_amount: Amount for pruning quality
        diversity_pruning_func: Optional function for diversity pruning
        diversity_pruning_amount: Amount for pruning diversity

    Returns:
        None
    """
    process_single_debate_entry(
        entry=entry,
        required_columns=["question", "response_A", "response_B", "id"],
        base_dir=base_dir,
        max_rounds=max_rounds,
        model_configs=model_configs,
        overwrite=overwrite,
        prompt_builder_fn=lambda prompt_params: PromptBuilder(
            round_zero_fn=build_judge_anything_pair_round_zero_prompt,
            round_n_fn=build_judge_anything_pair_round_n_prompt,
            prompt_params=prompt_params,
            query=entry["question"],
        ),
        prompt_params={
            "question": entry["question"],
            "response_a": entry["response_A"],
            "response_b": entry["response_B"],
        },
        extract_func=extract_caption_a_b_answer,
        temperature=temperature,
        max_tokens=max_tokens,
        batch=batch,
        batch_size=batch_size,
        quality_pruning_func=quality_pruning_func,
        quality_pruning_amount=quality_pruning_amount,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
    )
