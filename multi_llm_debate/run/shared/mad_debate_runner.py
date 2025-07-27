import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ...mad.debate import Debate
from ...mad.prompts import (
    AFFIRMATIVE_PROMPT,
    MODERATOR_META_PROMPT,
    MODERATOR_PROMPT,
    NEGATIVE_PROMPT,
    PLAYER_META_PROMPT,
)
from ...utils.model_config import ModelConfig
from ..shared.utils import format_time, model_configs_to_string

logger = logging.getLogger(__name__)


class MADDebateRunner:
    """Runner for MAD (Multi-Agent Debate) framework."""

    def __init__(
        self,
        model_configs: List[ModelConfig],
        temperature: float = 1.0,
        max_tokens: int = 6400,
        num_players: int = 3,
        provider: str = "ollama",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_rounds: int = 3,
    ):
        """Initialize the MAD debate runner.

        Args:
            model_configs: List of model configurations
            temperature: Temperature for model responses
            max_tokens: Maximum number of tokens for model responses
            num_players: Number of players in the debate
            provider: LLM provider
            base_url: Base URL for API calls
            api_key: API key for the provider
            max_rounds: Maximum number of debate rounds
        """
        self.model_configs = model_configs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_players = num_players
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.max_rounds = max_rounds

    def run_debate(
        self,
        debate_topic: str,
        output_dir: Path,
        entry_id: str,
    ) -> Dict[str, Any]:
        """Run a MAD debate for the given topic.

        Args:
            debate_topic: The topic for the debate
            output_dir: Directory to save results
            entry_id: Unique identifier for the entry

        Returns:
            Dict containing debate results
        """
        logger.info(f"Running MAD debate for topic: {debate_topic[:100]}...")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare MAD configuration
        mad_config = self._prepare_mad_config(debate_topic)

        # Get model name from configs (use first one for simplicity)
        model_name = (
            self.model_configs[0]["name"] if self.model_configs else "gpt-3.5-turbo"
        )

        try:
            # Create and run MAD debate
            debate = Debate(
                model_name=model_name,
                temperature=self.temperature,
                num_players=self.num_players,
                provider=self.provider,
                config=mad_config,
                max_round=self.max_rounds,
                base_url=self.base_url,
                api_key=self.api_key,
            )

            # Run the debate
            debate_results = debate.run()

            # Save results
            results = self._save_debate_results(
                debate_results, output_dir, entry_id, debate_topic
            )

            logger.info(f"MAD debate completed for entry {entry_id}")
            return results

        except Exception as e:
            logger.error(f"Error running MAD debate for entry {entry_id}: {str(e)}")
            raise

    def _prepare_mad_config(self, debate_topic: str) -> Dict[str, Any]:
        """Prepare configuration for MAD debate.

        Args:
            debate_topic: The topic for the debate

        Returns:
            MAD configuration dictionary
        """
        return {
            "debate_topic": debate_topic,
            "player_meta_prompt": PLAYER_META_PROMPT.replace(
                "##debate_topic##", debate_topic
            ),
            "moderator_meta_prompt": MODERATOR_META_PROMPT.replace(
                "##debate_topic##", debate_topic
            ),
            "affirmative_prompt": AFFIRMATIVE_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "moderator_prompt": MODERATOR_PROMPT,
        }

    def _save_debate_results(
        self,
        debate_results: Any,
        output_dir: Path,
        entry_id: str,
        debate_topic: str,
    ) -> Dict[str, Any]:
        """Save debate results to files.

        Args:
            debate_results: Results from the MAD debate
            output_dir: Directory to save results
            entry_id: Unique identifier for the entry
            debate_topic: The debate topic

        Returns:
            Dict containing saved results and metadata
        """
        # Create results structure
        results = {
            "entry_id": entry_id,
            "debate_topic": debate_topic,
            "model_configs": self.model_configs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "num_players": self.num_players,
            "provider": self.provider,
            "max_rounds": self.max_rounds,
            "debate_results": debate_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Save results to JSON file
        results_file = output_dir / f"{entry_id}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Extract final answer if available
        final_answer = self._extract_final_answer(debate_results)
        results["final_answer"] = final_answer

        # Save final answer separately
        answer_file = output_dir / f"{entry_id}_answer.txt"
        with open(answer_file, "w") as f:
            f.write(str(final_answer))

        return results

    def _extract_final_answer(self, debate_results: Any) -> str:
        """Extract the final answer from debate results.

        Args:
            debate_results: Results from the MAD debate

        Returns:
            Extracted final answer
        """
        try:
            # Try to extract from moderator's final decision
            if hasattr(debate_results, "moderator_decision"):
                decision = debate_results.moderator_decision
                if isinstance(decision, dict) and "debate_answer" in decision:
                    return decision["debate_answer"]
                return str(decision)

            # Try to extract from the last round
            if hasattr(debate_results, "rounds") and debate_results.rounds:
                last_round = debate_results.rounds[-1]
                if hasattr(last_round, "moderator_response"):
                    response = last_round.moderator_response
                    if isinstance(response, str):
                        try:
                            parsed = json.loads(response)
                            if isinstance(parsed, dict) and "debate_answer" in parsed:
                                return parsed["debate_answer"]
                        except json.JSONDecodeError:
                            pass
                    return str(response)

            # Fallback to string representation
            return str(debate_results)

        except Exception as e:
            logger.warning(f"Could not extract final answer: {str(e)}")
            return str(debate_results)


def run_mad_debate_workflow(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "mad",
    model_configs: Optional[List[ModelConfig]] = None,
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
    """Run MAD debate workflow on the given dataframe.

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
        max_rounds: Maximum number of debate rounds

    Returns:
        Dict containing execution results
    """
    # Validate required columns
    required_columns = ["debate_topic", "id"]
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Use default model configs if none provided
    if model_configs is None:
        model_configs = [{"name": "gpt-3.5-turbo", "quantity": 1, "provider": "ollama"}]

    # Create MAD debate runner
    runner = MADDebateRunner(
        model_configs=model_configs,
        temperature=temperature,
        max_tokens=max_tokens,
        num_players=num_players,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_rounds=max_rounds,
    )

    # Process entries
    total_entries = len(dataframe)
    processed_count = 0
    failed_entries = []

    logger.info(f"Starting MAD debate workflow with {total_entries} entries")

    for idx, row in dataframe.iterrows():
        entry_id = str(row["id"])
        debate_topic = row["debate_topic"]

        # Create output directory for this entry
        entry_output_dir = base_dir / entry_id

        try:
            logger.info(f"Processing entry {entry_id} ({int(idx) + 1}/{total_entries})")

            # Run debate for this entry
            runner.run_debate(
                debate_topic=str(debate_topic),
                output_dir=entry_output_dir,
                entry_id=entry_id,
            )

            processed_count += 1
            logger.info(f"Successfully processed entry {entry_id}")

        except Exception as e:
            logger.error(f"Failed to process entry {entry_id}: {str(e)}")
            failed_entries.append(
                {
                    "entry_id": entry_id,
                    "error": str(e),
                    "index": idx,
                }
            )

    # Prepare execution report
    execution_report = {
        "total_entries": total_entries,
        "processed_count": processed_count,
        "failed_entries": failed_entries,
        "success_rate": (
            (processed_count / total_entries * 100) if total_entries > 0 else 0
        ),
        "model_configs": model_configs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_players": num_players,
        "provider": provider,
        "max_rounds": max_rounds,
    }

    logger.info(
        f"MAD debate workflow completed. Success rate: {execution_report['success_rate']:.2f}%"
    )

    return execution_report
