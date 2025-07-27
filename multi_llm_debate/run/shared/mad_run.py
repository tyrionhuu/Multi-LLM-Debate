import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ...mad.debate import Debate
from ...utils.model_config import ModelConfig
from .run import process_debate_dataset, process_single_debate_entry
from .utils import format_time, model_configs_to_string

logger = logging.getLogger(__name__)


def process_mad_dataset(
    dataframe: pd.DataFrame,
    max_rounds: int = 3,
    base_dir: Path = Path("data") / "mad",
    model_configs: Optional[List[ModelConfig]] = None,
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
    """Run the MAD (Multi-Agent Debate) task on a DataFrame.

    Args:
        dataframe: Pandas DataFrame containing debate topics and related data
        max_rounds: Maximum number of debate rounds (default: 3 for MAD)
        base_dir: Base directory for output files
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        overwrite: Whether to overwrite existing debate results (default: False)
        temperature: Temperature for model responses
        max_tokens: Maximum number of tokens for model responses
        batch: Whether to run in batch mode (default: False)
        batch_size: Number of entries to process in a single batch (default: 11)
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
    required_columns = ["debate_topic", "id"]

    # For MAD, we'll use the direct workflow instead of the generic process_debate_dataset
    # since MAD has specific requirements that don't fit the generic pattern
    from .mad_debate_runner import run_mad_debate_workflow
    
    return run_mad_debate_workflow(
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


# Note: process_mad_entry is not used since MAD uses its own workflow
# This function is kept for compatibility but not implemented


def build_mad_prompt_builder(
    debate_topic: str,
    max_rounds: int = 3,
    mad_params: Optional[Dict[str, Any]] = None,
) -> "MADPromptBuilder":
    """Build a MAD prompt builder for the given debate topic.

    Args:
        debate_topic: The topic for the debate
        max_rounds: Maximum number of debate rounds
        mad_params: Additional parameters for MAD framework

    Returns:
        MADPromptBuilder instance
    """
    return MADPromptBuilder(debate_topic, max_rounds, mad_params)


def extract_mad_answer(response: str) -> str:
    """Extract the final answer from MAD debate response.

    Args:
        response: The response from the MAD debate

    Returns:
        Extracted answer string
    """
    try:
        # Try to parse JSON response
        if isinstance(response, str):
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "debate_answer" in parsed:
                return parsed["debate_answer"]
        return response
    except (json.JSONDecodeError, TypeError):
        # If not JSON, return as is
        return response


class MADPromptBuilder:
    """Prompt builder for MAD framework debates."""

    def __init__(
        self,
        debate_topic: str,
        max_rounds: int = 3,
        mad_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the MAD prompt builder.

        Args:
            debate_topic: The topic for the debate
            max_rounds: Maximum number of debate rounds
            mad_params: Additional parameters for MAD framework
        """
        self.debate_topic = debate_topic
        self.max_rounds = max_rounds
        self.mad_params = mad_params or {}
        self.images = None  # MAD doesn't support images

    def build_round_zero(self) -> str:
        """Build the initial prompt for round zero.

        Returns:
            Prompt string for round zero
        """
        return f"Debate Topic: {self.debate_topic}\n\nPlease provide your initial position on this topic."

    def build_round_n(self, round_num: int, previous_responses: List[Dict]) -> str:
        """Build prompt for round n.

        Args:
            round_num: Current round number
            previous_responses: Responses from previous rounds

        Returns:
            Prompt string for round n
        """
        if round_num == 0:
            return self.build_round_zero()
        
        # Build context from previous responses
        context = self._build_context_from_previous_responses(previous_responses)
        
        return f"Debate Topic: {self.debate_topic}\n\nPrevious round responses:\n{context}\n\nPlease provide your response for round {round_num + 1}."

    def _build_context_from_previous_responses(self, previous_responses: List[Dict]) -> str:
        """Build context string from previous responses.

        Args:
            previous_responses: List of previous round responses

        Returns:
            Context string
        """
        context_parts = []
        for i, round_responses in enumerate(previous_responses):
            context_parts.append(f"Round {i + 1}:")
            for j, response in enumerate(round_responses):
                agent_name = response.get("agent_name", f"Agent {j + 1}")
                content = response.get("content", "")
                context_parts.append(f"  {agent_name}: {content}")
        
        return "\n".join(context_parts)


def run_mad_debate(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "mad",
    model_configs: Optional[List[ModelConfig]] = None,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
    quality_pruning_func: Callable = None,
    quality_pruning_amount: int = 5,
    diversity_pruning_func: Callable = None,
    diversity_pruning_amount: int = 5,
    num_players: int = 3,
    provider: str = "ollama",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run MAD debate on the given dataframe.

    Args:
        dataframe: Input DataFrame containing debate data
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

    Returns:
        Dict containing execution results
    """
    return process_mad_dataset(
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
    ) 