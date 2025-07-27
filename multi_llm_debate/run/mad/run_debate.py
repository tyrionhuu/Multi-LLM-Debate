import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from multi_llm_debate.mad.debate import Debate

from .utils import create_mad_config, save_mad_results

logger = logging.getLogger(__name__)


def process_mad_dataset(
    dataframe: pd.DataFrame,
    model_configs: List[Dict[str, Any]],
    output_dir: Path,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    max_rounds: int = 3,
    **kwargs,
) -> Dict[str, Any]:
    """Process the MAD dataset using the MAD framework.

    Args:
        dataframe (pd.DataFrame): DataFrame containing debate topics
        model_configs (List[Dict[str, Any]]): List of model configurations
        output_dir (Path): Directory to save results
        temperature (float): Temperature for model responses
        max_tokens (int): Maximum tokens for responses
        max_rounds (int): Maximum debate rounds
        **kwargs: Additional arguments

    Returns:
        Dict[str, Any]: Results from the MAD debate
    """
    logger.info(f"Processing MAD dataset with {len(dataframe)} samples")

    results = {
        "task": "mad",
        "model_configs": model_configs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_rounds": max_rounds,
        "debates": [],
    }

    # Get the first model config for the debate (MAD uses the same model for all agents)
    if not model_configs:
        raise ValueError("No model configurations provided")

    model_config = model_configs[0]
    model_name = model_config.get("name", "gpt-3.5-turbo")
    base_url = model_config.get("base_url")
    api_key = model_config.get("api_key")

    for idx, row in dataframe.iterrows():
        try:
            logger.info(
                f"Processing debate {idx + 1}/{len(dataframe)}: {row['question']}"
            )

            # Create MAD configuration
            config = create_mad_config(
                debate_topic=row["question"],
                model_configs=model_configs,
                max_rounds=max_rounds,
            )

            # Create and run debate
            debate = Debate(
                model_name=model_name,
                temperature=temperature,
                num_players=3,
                provider="ollama",  # Default provider
                config=config,
                max_round=max_rounds,
                sleep_time=0,
                base_url=base_url,
                api_key=api_key,
            )

            # Run the debate
            debate_result = debate.run()

            # Store results
            debate_output = {
                "id": row.get("id", idx),
                "question": row["question"],
                "category": row.get("category", "debate"),
                "difficulty": row.get("difficulty", "medium"),
                "base_answer": debate_result.get("base_answer", ""),
                "debate_answer": debate_result.get("debate_answer", ""),
                "reason": debate_result.get("Reason", ""),
                "success": debate_result.get("success", False),
                "rounds_completed": len(debate.players[0].memory_lst)
                // 2,  # Rough estimate
                "affirmative_response": debate_result.get("aff_ans", ""),
                "negative_response": debate_result.get("neg_ans", ""),
                "moderator_response": debate_result.get("mod_ans", ""),
            }

            results["debates"].append(debate_output)

            # Save individual debate result
            debate_file = output_dir / f"debate_{idx:04d}.json"
            save_mad_results(debate_output, debate_file)

            logger.info(f"Completed debate {idx + 1}: {debate_output['success']}")

        except Exception as e:
            logger.error(f"Error processing debate {idx + 1}: {str(e)}")

            # Store error result
            error_output = {
                "id": row.get("id", idx),
                "question": row["question"],
                "category": row.get("category", "debate"),
                "difficulty": row.get("difficulty", "medium"),
                "error": str(e),
                "success": False,
            }

            results["debates"].append(error_output)

    # Save overall results
    results_file = output_dir / "mad_results.json"
    save_mad_results(results, results_file)

    logger.info(f"MAD processing completed. Results saved to {results_file}")
    return results


def extract_mad_answer(response: Dict[str, Any]) -> str:
    """Extract the final answer from MAD debate results.

    Args:
        response (Dict[str, Any]): MAD debate response

    Returns:
        str: Extracted answer
    """
    if isinstance(response, dict):
        return response.get("debate_answer", response.get("base_answer", ""))
    return str(response)


def extract_mad_success(response: Dict[str, Any]) -> bool:
    """Extract success status from MAD debate results.

    Args:
        response (Dict[str, Any]): MAD debate response

    Returns:
        bool: Success status
    """
    if isinstance(response, dict):
        return response.get("success", False)
    return False
