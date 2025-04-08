import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...llm.prompt_builder import PromptBuilder
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from ..shared.run import process_debate_dataset, process_single_debate_entry
from .prompts import (
    build_llm_bar_round_n_prompt,
    build_llm_bar_round_zero_prompt,
)
from .utils import extract_1_2_answer

# Fix the setup_logging call by removing the level parameter
logger = setup_logging(__name__)
logger.setLevel(logging.INFO)

def process_llm_bar_dataset(
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "llm_bar",
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    parallel: bool = False,
) -> Dict[str, Any]:
    """Run the LLMBar task on a DataFrame.

    Args:
        dataframe: Pandas DataFrame containing question, answer, response_1 and response_2
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        overwrite: Whether to overwrite existing debate results (default: False)
        max_workers: Maximum number of concurrent workers (default: 4)
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        parallel: Whether to run in parallel (default: False)

    Returns:
        Dict containing summary of execution including failed entries

    Raises:
        ValueError: If DataFrame format is invalid
    """
    required_columns = [
        "question",
        "response_1",
        "response_2",
        "id",
        "answer",
    ]

    # Process the dataset for debates
    return process_debate_dataset(
        dataframe=dataframe,
        process_entry_fn=process_llm_bar_entry,
        required_columns=required_columns,
        base_dir=base_dir,
        max_rounds=max_rounds,
        model_configs=model_configs,
        overwrite=overwrite,
        max_workers=max_workers,
        task_name="LLMBar task",
        temperature=temperature,
        max_tokens=max_tokens,
        parallel=parallel,
    )
    
def process_llm_bar_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    model_configs: Optional[List[ModelConfig]] = None,
    base_dir: Path = Path("data") / "llm_bar",
    overwrite: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    parallel: bool = False,
) -> Dict[str, Any]:
    """Process a single entry for the LLMBar task.

    Args:
        entry: Pandas Series containing the entry data
        round_number: Current round number
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        base_dir: Base directory for output files
        overwrite: Whether to overwrite existing debate results (default: False)
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        parallel: Whether to run in parallel (default: False)
        
    Returns:
        Dict containing execution summary including failed entries

    Raises:
        ValueError: If entry format is invalid
    """
    logger.info(f"Processing entry ID: {entry['id']} for LLMBar task")
    
    process_single_debate_entry(
        entry=entry,
        required_columns=[
            "question",
            "response_1",
            "response_2",
            "id",
            "answer",
        ],
        base_dir=base_dir,
        max_rounds=max_rounds,
        model_configs=model_configs,
        overwrite=overwrite,
        prompt_builder_fn=lambda prompt_params: PromptBuilder(
            prompt_params=prompt_params,
            build_round_zero_prompt=build_llm_bar_round_zero_prompt,
            build_round_n_prompt=build_llm_bar_round_n_prompt,
        ),
        prompt_params={
            "question": entry["question"],
            "response_1": entry["response_1"],
            "response_2": entry["response_2"],
        },
        process_answer_fn=extract_1_2_answer,
        temperature=temperature,
        max_tokens=max_tokens,
        parallel=parallel,
    )