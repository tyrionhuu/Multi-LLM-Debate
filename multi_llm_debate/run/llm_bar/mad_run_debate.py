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
    max_rounds: int = 3,
    base_dir: Path = Path("data") / "llm_bar_mad",
    model_configs: Optional[List[Dict[str, Any]]] = None,
    overwrite: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Optional[Callable] = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Optional[Callable] = None,
    diversity_pruning_amount: int = 5,
    num_players: int = 3,
    provider: str = "ollama",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run MAD debate on LLMBar dataset.

    Args:
        dataframe: LLMBar DataFrame with columns [question, response_1, response_2, answer, id]
        max_rounds: Maximum number of debate rounds (default: 3 for MAD)
        base_dir: Base directory for output files
        model_configs: Optional list of model configurations
        overwrite: Whether to overwrite existing debate results
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        batch: Whether to run in batch mode
        batch_size: Number of entries to process in a single batch
        quality_pruning_func: Optional function for quality pruning
        quality_pruning_amount: Amount for pruning quality
        diversity_pruning_func: Optional function for diversity pruning
        diversity_pruning_amount: Amount for pruning diversity
        num_players: Number of players in the debate (default: 3)
        provider: LLM provider (default: "ollama")
        base_url: Base URL for API calls
        api_key: API key for the provider

    Returns:
        Dict containing summary of execution including failed entries

    Raises:
        ValueError: If DataFrame format is invalid
    """
    # Validate required columns
    required_columns = ["question", "response_1", "response_2", "answer", "id"]
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert to MAD format
    mad_dataframe = convert_llm_bar_to_mad_format(dataframe)

    # Run MAD debate workflow
    results = run_mad_debate_workflow(
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
        num_players=num_players,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_rounds=max_rounds,
        task_name="llm_bar",
    )

    return results


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
    num_players: int = 3,
    provider: str = "ollama",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """Run MAD debate on LLMBar dataset.

    Args:
        dataframe: LLMBar DataFrame
        base_dir: Base directory for output files
        model_configs: List of model configurations
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        batch: Whether to run in batch mode
        batch_size: Number of entries to process in a single batch
        quality_pruning_func: Optional function for quality pruning
        quality_pruning_amount: Amount for pruning quality
        diversity_pruning_func: Optional function for diversity pruning
        diversity_pruning_amount: Amount for pruning diversity
        num_players: Number of players in the debate
        provider: LLM provider
        base_url: Base URL for API calls
        api_key: API key for the provider
        max_rounds: Maximum number of debate rounds

    Returns:
        Dict containing execution results
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
        num_players=num_players,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_rounds=max_rounds,
    )
