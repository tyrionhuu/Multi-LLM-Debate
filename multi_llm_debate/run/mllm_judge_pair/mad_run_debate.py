import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..shared.mad_debate_runner import run_mad_debate_workflow

logger = logging.getLogger(__name__)


def convert_mllm_judge_pair_to_mad_format(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert MLLM-Judge-pairs format to MAD debate format.

    MLLM-Judge-pairs format: question, response_A, response_B, answer, id, image
    MAD format: debate_topic, id

    Args:
        dataframe: MLLM-Judge-pairs DataFrame with columns [question, response_A, response_B, answer, id, image]

    Returns:
        DataFrame in MAD format with columns [debate_topic, id]
    """
    mad_data = []

    for _, row in dataframe.iterrows():
        question = row["question"]
        response_A = row["response_A"]
        response_B = row["response_B"]
        answer = row["answer"]
        entry_id = row["id"]
        image = row.get("image", None)  # Get image (base64 or path), use None for empty

        # Create debate topic without embedding image as text
        debate_topic = f"""Question: {question}

Response A: {response_A}

Response B: {response_B}

Please debate which response (Response A or Response B) better answers the question about the image. 
Consider factors such as accuracy, completeness, relevance, and helpfulness."""

        mad_data.append(
            {
                "debate_topic": debate_topic,
                "id": entry_id,
                "original_question": question,
                "response_A": response_A,
                "response_B": response_B,
                "correct_answer": answer,
                "image": image,  # Keep image in the row for reference
            }
        )

    return pd.DataFrame(mad_data)


def process_mllm_judge_pair_mad_dataset(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "mllm_judge_pair_mad",
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
    task_name: str = "mllm_judge_pair_mad",  # Add task_name parameter
) -> Dict[str, Any]:
    """Process MLLM-Judge-pairs dataset using MAD framework.

    Args:
        dataframe: MLLM-Judge-pairs DataFrame with columns [question, response_A, response_B, answer, id]
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
    logger.info("Converting MLLM-Judge-pairs format to MAD debate format...")
    mad_dataframe = convert_mllm_judge_pair_to_mad_format(dataframe)

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


def run_mllm_judge_pair_mad_debate(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "mllm_judge_pair_mad",
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
    task_name: str = "mllm_judge_pair_mad",  # Add task_name parameter
) -> Dict[str, Any]:
    """Run MLLM-Judge-pairs MAD debate workflow.

    Wrapper function for process_mllm_judge_pair_mad_dataset.

    Args:
        dataframe: MLLM-Judge-pairs DataFrame
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
    return process_mllm_judge_pair_mad_dataset(
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
