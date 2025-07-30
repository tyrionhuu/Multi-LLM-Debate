import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from ...mad.debate import Debate
from ...mad.prompts import (
    DEBATER_A_INITIAL_PROMPT,
    DEBATER_B_DISAGREE_PROMPT,
    DEBATER_A_REBUTTAL_PROMPT,
    JUDGE_DECISION_PROMPT,
    JUDGE_META_PROMPT,
    PLAYER_META_PROMPT,
    build_mad_prompts_for_task,
)
from ..shared.utils import format_time, model_configs_to_string

logger = logging.getLogger(__name__)


def save_mad_results_to_csv(
    evaluation_results: Dict[str, Any],
    task_name: str,
    model_configs: List[Dict[str, Any]],
    report_path: Path,
    running_time: float,
) -> None:
    """Save MAD evaluation results to CSV file.

    Args:
        evaluation_results: Results from MAD evaluation
        task_name: Name of the task
        model_configs: Model configurations used
        report_path: Path to save the CSV file
        running_time: Total running time in seconds
    """
    csv_path = report_path / "results.csv"
    logger.info(f"Saving MAD results to {csv_path}")

    # Calculate error margin (using standard error for now)
    accuracy = evaluation_results.get("accuracy", 0.0)
    total_entries = evaluation_results.get("total_entries", 0)
    processed_entries = evaluation_results.get("processed_entries", 0)

    # Calculate standard error of proportion
    if processed_entries > 0:
        # Standard error = sqrt(p * (1-p) / n)
        error_margin = ((accuracy * (1 - accuracy)) / processed_entries) ** 0.5
    else:
        error_margin = 0.0

    # Format running time
    display_time, csv_time = format_time(running_time)

    # Create model configuration string
    current_config = model_configs_to_string(model_configs)

    # Create new row for MAD results
    new_row = [
        current_config,
        task_name,
        f"{accuracy:.4f}",
        f"{error_margin:.4f}",
        csv_time,
    ]

    # Read existing data if file exists
    existing_data = []
    if csv_path.exists():
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                existing_data = list(reader)
        except Exception as e:
            logger.error(f"Error reading existing CSV: {str(e)}")
            existing_data = []

    # Create header if file doesn't exist
    if not existing_data:
        existing_data = [
            [
                "Model Configuration",
                "Task Name",
                "MAD Accuracy",
                "MAD Error Margin",
                "Running Time",
            ]
        ]

    # Update existing entry or append new one
    found = False
    for i, row in enumerate(existing_data[1:], 1):
        if row and row[0] == current_config and row[1] == task_name:
            existing_data[i] = new_row
            found = True
            break
    if not found:
        existing_data.append(new_row)

    # Create directory if it doesn't exist
    report_path.mkdir(parents=True, exist_ok=True)

    # Write all data back to CSV
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(existing_data)
        print(f"\nMAD results saved to {csv_path}")
    except Exception as e:
        logger.error(f"Error writing MAD results to CSV: {str(e)}")
        print(f"\nFailed to save MAD results: {str(e)}")


class MADDebateRunner:
    """Runner for MAD (Multi-Agent Debate) framework."""

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: int = 6400,
        num_debaters: int = 2,  # Changed from num_players to num_debaters, default to 2 for practical use
        provider: str = "ollama",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_rounds: int = 10,  # Increased default from 3 to 10
        verbose: bool = False,  # Add verbose mode
    ):
        """Initialize the MAD debate runner.

        Args:
            model_configs: List of model configurations
            temperature: Temperature for model responses
            max_tokens: Maximum number of tokens for model responses
            num_debaters: Number of debaters in the debate
            provider: LLM provider
            base_url: Base URL for API calls
            api_key: API key for the provider
            max_rounds: Maximum number of debate rounds
        """
        self.model_configs = model_configs
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_debaters = num_debaters  # Changed from num_players to num_debaters
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.max_rounds = max_rounds
        self.verbose = verbose  # Store verbose setting

    def run_debate(
        self,
        debate_topic: str,
        output_dir: Path,
        entry_id: str,
        task_name: str = "default",
    ) -> Dict[str, Any]:
        """Run a MAD debate for the given topic.

        Args:
            debate_topic: The topic for the debate
            output_dir: Directory to save results
            entry_id: Unique identifier for the entry

        Returns:
            Dict containing debate results
        """
        # Running MAD debate silently

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare MAD configuration
        mad_config = self._prepare_mad_config(debate_topic, task_name)

        # Get model name from configs (use first one for simplicity)
        model_name = (
            self.model_configs[0]["name"] if self.model_configs else "gpt-3.5-turbo"
        )

        try:
            # Create and run MAD debate
            debate = Debate(
                model_name=model_name,
                temperature=self.temperature,
                num_debaters=self.num_debaters,  # Using num_debaters
                provider=self.provider,
                config=mad_config,
                max_round=self.max_rounds,
                base_url=self.base_url,
                api_key=self.api_key,
                verbose=self.verbose,  # Pass verbose setting
            )

            # Run the debate
            debate_results = debate.run()

            # Save results
            results = self._save_debate_results(
                debate_results, output_dir, entry_id, debate_topic
            )

            return results

        except Exception as e:
            logger.error(f"Error running MAD debate for entry {entry_id}: {str(e)}")
            raise

    def _prepare_mad_config(self, debate_topic: str, task_name: str = "default") -> Dict[str, Any]:
        """Prepare configuration for MAD debate.

        Args:
            debate_topic: The topic for the debate
            task_name: Name of the task for task-specific prompts

        Returns:
            MAD configuration dictionary
        """
        # Import task-specific prompts
        if task_name == "judge_anything_pair":
            from ..judge_anything_pair.mad_prompts import build_judge_anything_pair_mad_prompts
            task_prompts = build_judge_anything_pair_mad_prompts(debate_topic)
        elif task_name == "big_bench":
            from ..big_bench.mad_prompts import build_big_bench_mad_prompts
            task_prompts = build_big_bench_mad_prompts(debate_topic)
        elif task_name == "judge_bench":
            from ..judge_bench.mad_prompts import build_judge_bench_mad_prompts
            task_prompts = build_judge_bench_mad_prompts(debate_topic)
        elif task_name == "llm_bar":
            from ..llm_bar.mad_prompts import build_llm_bar_mad_prompts
            task_prompts = build_llm_bar_mad_prompts(debate_topic)
        elif task_name == "mllm_judge_pair":
            from ..mllm_judge_pair.mad_prompts import build_mllm_judge_pair_mad_prompts
            task_prompts = build_mllm_judge_pair_mad_prompts(debate_topic)
        elif task_name == "truthful_qa":
            from ..truthful_qa.mad_prompts import build_truthful_qa_mad_prompts
            task_prompts = build_truthful_qa_mad_prompts(debate_topic)
        else:
            # Fallback to generic prompts
            task_prompts = build_mad_prompts_for_task(task_name)
            # Replace debate topic in meta prompts
            task_prompts["player_meta_prompt"] = task_prompts["player_meta_prompt"].replace(
                "##debate_topic##", debate_topic
            )
            task_prompts["moderator_meta_prompt"] = task_prompts["moderator_meta_prompt"].replace(
                "##debate_topic##", debate_topic
            )
        
        return {
            "debate_topic": debate_topic,
            "player_meta_prompt": task_prompts["player_meta_prompt"],
            "judge_meta_prompt": task_prompts["judge_meta_prompt"],
            "debater_prompt": task_prompts["debater_prompt"],
            "judge_discriminative_prompt": task_prompts["judge_discriminative_prompt"],
            "judge_extractive_prompt": task_prompts["judge_extractive_prompt"],
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
            "num_debaters": self.num_debaters,  # Changed from num_players to num_debaters
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

        # Save final answer separately as JSON for better structure
        answer_file = output_dir / f"{entry_id}_answer.json"
        answer_data = {
            "final_answer": final_answer,
            "extraction_method": "from_debate_results",
            "timestamp": datetime.now().isoformat()
        }
        with open(answer_file, "w") as f:
            json.dump(answer_data, f, indent=2, default=str)

        return results

    def _extract_final_answer(self, debate_results: Any) -> str:
        """Extract the final answer from debate results.

        Args:
            debate_results: Results from the MAD debate

        Returns:
            Extracted final answer
        """
        try:
            # NEW: Handle N-debater MAD framework format
            # The new framework stores results in config field
            if hasattr(debate_results, "config"):
                config = debate_results.config
                if isinstance(config, dict):
                    # Look for Final Answer in config (new N-debater format)
                    if "Final Answer" in config:
                        return config["Final Answer"]
                    elif "final_answer" in config:
                        return config["final_answer"]
                    # Also check for solution_obtained and reasoning
                    elif config.get("solution_obtained", False) and "Final Answer" in config:
                        return config["Final Answer"]

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
                            if isinstance(parsed, dict) and "Final Answer" in parsed:
                                return parsed["Final Answer"]
                        except json.JSONDecodeError:
                            pass
                    return str(response)

            # Try to extract from base_answer structure (unified format)
            if hasattr(debate_results, "base_answer"):
                base_answer = debate_results.base_answer
                
                # Look for final_choice directly in base_answer (highest priority)
                if hasattr(base_answer, "final_choice"):
                    return base_answer.final_choice
                
                # Look for conclusion directly in base_answer
                if hasattr(base_answer, "conclusion"):
                    return base_answer.conclusion
                
                # Look for winner directly in base_answer
                if hasattr(base_answer, "winner"):
                    return base_answer.winner
                
                # Look in debate structure
                if hasattr(base_answer, "debate"):
                    debate = base_answer.debate
                    
                    # Handle debate as object
                    if hasattr(debate, "final_choice"):
                        return debate.final_choice
                    elif hasattr(debate, "conclusion"):
                        return debate.conclusion
                    elif hasattr(debate, "verdict"):
                        return debate.verdict
                    
                    # Handle debate as list
                    if isinstance(debate, list) and debate:
                        # First, look for final_choice in any element (highest priority)
                        for item in debate:
                            if hasattr(item, "final_choice"):
                                return item.final_choice
                        
                        # Then look for choice in any element
                        for item in debate:
                            if hasattr(item, "choice"):
                                return item.choice
                        
                        # Finally look for conclusion in any element (lowest priority)
                        for item in debate:
                            if hasattr(item, "conclusion"):
                                return item.conclusion

            # Try to extract from debate_answer field
            if hasattr(debate_results, "debate_answer"):
                return debate_results.debate_answer

            # Try to extract from Final Answer field
            if hasattr(debate_results, "Final Answer"):
                return debate_results["Final Answer"]

            # Fallback: return the entire results as string
            return str(debate_results)

        except Exception as e:
            logger.warning(f"Error extracting final answer: {str(e)}")
            return str(debate_results)


def run_mad_debate_workflow(
    dataframe: pd.DataFrame,
    base_dir: Path = Path("data") / "mad",
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
    provider: str = "ollama",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_rounds: int = 10,  # Increased default from 3 to 10
    task_name: str = "default",
    verbose: bool = False,  # Add verbose mode
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

    # Extract provider, base_url, and api_key from model configs if not explicitly provided
    if model_configs:
        first_config = model_configs[0]
        if provider == "ollama":
            provider = first_config.get("provider", "ollama")
        if base_url is None:
            base_url = first_config.get("base_url")
        if api_key is None:
            api_key = first_config.get("api_key")

    # Create model configuration directory name
    model_config_str = model_configs_to_string(model_configs)
    model_dir_name = (
        model_config_str.replace(" ", "_").replace(".", "_").replace("/", "_")
    )
    model_output_dir = base_dir / model_dir_name

    logger.info(f"Creating MAD debate directory: {model_output_dir}")
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Create MAD debate runner
    runner = MADDebateRunner(
        model_configs=model_configs,
        temperature=temperature,
        max_tokens=max_tokens,
        num_debaters=num_debaters,  # Changed from num_players to num_debaters
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        max_rounds=max_rounds,
        verbose=verbose,  # Pass verbose setting
    )

    # Process entries with progress bar
    total_entries = len(dataframe)
    processed_count = 0
    failed_entries = []

    logger.info(f"Starting MAD debate workflow with {total_entries} entries")

    # Create progress bar
    with tqdm(total=total_entries, desc="MAD Debates", unit="entry") as pbar:
        for idx, row in dataframe.iterrows():
            entry_id = str(row["id"])
            debate_topic = row["debate_topic"]

            # Create output directory for this entry within the model-specific directory
            entry_output_dir = model_output_dir / entry_id
            results_file = entry_output_dir / f"{entry_id}_results.json"

            # Check if results already exist
            if results_file.exists():
                pbar.set_postfix({"status": "skipped", "entry": entry_id})
                processed_count += 1
                pbar.update(1)
                continue

            try:
                pbar.set_postfix({"status": "processing", "entry": entry_id})

                # Run debate for this entry
                runner.run_debate(
                    debate_topic=str(debate_topic),
                    output_dir=entry_output_dir,
                    entry_id=entry_id,
                    task_name=task_name,
                )

                processed_count += 1
                pbar.set_postfix({"status": "completed", "entry": entry_id})

            except Exception as e:
                logger.error(f"Failed to process entry {entry_id}: {str(e)}")
                failed_entries.append(
                    {
                        "entry_id": entry_id,
                        "error": str(e),
                        "index": idx,
                    }
                )
                pbar.set_postfix({"status": "failed", "entry": entry_id})

            pbar.update(1)

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
        "num_debaters": num_debaters,  # Changed from num_players to num_debaters
        "provider": provider,
        "max_rounds": max_rounds,
    }

    logger.info(
        f"MAD debate workflow completed. Success rate: {execution_report['success_rate']:.2f}%"
    )

    return execution_report
