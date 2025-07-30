import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..shared.mad_debate_runner import run_mad_debate_workflow

logger = logging.getLogger(__name__)


def convert_llm_bar_to_mad_format(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert LLMBar format to MAD debate format.

    LLMBar format: question, response_1, response_2, answer
    MAD format: debate_topic, id

    Args:
        dataframe: LLMBar DataFrame with columns [question, response_1, response_2, answer, id]

    Returns:
        DataFrame in MAD format with columns [debate_topic, id]
    """
    mad_data = []

    for _, row in dataframe.iterrows():
        question = row["question"]
        response_1 = row["response_1"]
        response_2 = row["response_2"]
        answer = row["answer"]
        entry_id = row["id"]

        # Create debate topic that includes the question and both responses
        debate_topic = f"""Question: {question}

Response 1: {response_1}

Response 2: {response_2}

Please debate which response (Response 1 or Response 2) better answers the question. 
Consider factors such as accuracy, completeness, relevance, and helpfulness."""

        mad_data.append(
            {
                "debate_topic": debate_topic,
                "id": entry_id,
                "original_question": question,
                "response_1": response_1,
                "response_2": response_2,
                "correct_answer": answer,
            }
        )

    return pd.DataFrame(mad_data)


def process_llm_bar_mad_dataset(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "llm_bar_mad",
    model_configs: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Optional[Callable] = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Optional[Callable] = None,
    diversity_pruning_amount: int = 5,
    num_debaters: int = 2,  # Changed from num_players to num_debaters, default to 2 for practical use
    provider: str = "google",  # Changed from ollama to google
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_rounds: int = 10,  # Increased default from 3 to 10
    verbose: bool = False,  # Add verbose mode
    task_name: str = "llm_bar_mad",  # Add task_name parameter
) -> Dict[str, Any]:
    """Process LLMBar dataset using MAD framework.

    Args:
        dataframe: LLMBar DataFrame with columns [question, response_1, response_2, answer, id]
        base_dir: Output directory
        model_configs: Model configurations
        temperature: Temperature for model responses
        max_tokens: Maximum tokens for responses
        batch: Whether to run in batch mode
        batch_size: Batch size
        quality_pruning_func: Quality pruning function
        quality_pruning_amount: Quality pruning amount
        diversity_pruning_func: Diversity pruning function
        diversity_pruning_amount: Diversity pruning amount
        num_debaters: Number of debaters in debate
        provider: LLM provider
        base_url: Base URL for API calls
        api_key: API key for the provider
        max_rounds: Maximum debate rounds

    Returns:
        Execution results dictionary
    """
    # Validate required columns
    required_columns = ["question", "response_1", "response_2", "answer", "id"]
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info("Converting LLMBar format to MAD debate format...")
    mad_dataframe = convert_llm_bar_to_mad_format(dataframe)

    return run_mad_debate_workflow(
        dataframe=mad_dataframe,
        base_dir=base_dir,
        model_configs=model_configs,
        temperature=temperature,
        max_tokens=max_tokens,
        batch=batch,
        batch_size=batch_size,
        quality_pruning_func=quality_pruning_func,
        quality_pruning_amount=quality_pruning_amount,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
        num_debaters=num_debaters,  # Changed from num_players to num_debaters
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_rounds=max_rounds,
        task_name=task_name,  # Pass the actual task name
        verbose=verbose,  # Pass verbose setting
    )


def run_llm_bar_mad_debate(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "llm_bar_mad",
    model_configs: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Optional[Callable] = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Optional[Callable] = None,
    diversity_pruning_amount: int = 5,
    num_debaters: int = 2,  # Changed from num_players to num_debaters, default to 2 for practical use
    provider: str = "google",  # Changed from ollama to google
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_rounds: int = 10,  # Increased default from 3 to 10
    verbose: bool = False,  # Add verbose mode
    task_name: str = "llm_bar_mad",  # Add task_name parameter
) -> Dict[str, Any]:
    """Run LLMBar MAD debate workflow.

    Wrapper function for process_llm_bar_mad_dataset.

    Args:
        dataframe: LLMBar DataFrame
        base_dir: Output directory
        model_configs: Model configurations
        temperature: Temperature for model responses
        max_tokens: Maximum tokens for responses
        batch: Whether to run in batch mode
        batch_size: Batch size
        quality_pruning_func: Quality pruning function
        quality_pruning_amount: Quality pruning amount
        diversity_pruning_func: Diversity pruning function
        diversity_pruning_amount: Diversity pruning amount
        num_debaters: Number of debaters in debate
        provider: LLM provider
        base_url: Base URL for API calls
        api_key: API key for the provider
        max_rounds: Maximum debate rounds

    Returns:
        Execution results dictionary
    """
    return process_llm_bar_mad_dataset(
        dataframe=dataframe,
        base_dir=base_dir,
        model_configs=model_configs,
        temperature=temperature,
        max_tokens=max_tokens,
        batch=batch,
        batch_size=batch_size,
        quality_pruning_func=quality_pruning_func,
        quality_pruning_amount=quality_pruning_amount,
        diversity_pruning_func=diversity_pruning_func,
        diversity_pruning_amount=diversity_pruning_amount,
        num_debaters=num_debaters,  # Changed from num_players to num_debaters
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_rounds=max_rounds,
        verbose=verbose,  # Pass verbose setting
        task_name=task_name,  # Pass task_name parameter
    )
