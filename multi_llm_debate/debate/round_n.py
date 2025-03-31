import json
import logging
import time
from pathlib import Path
from typing import List

from ..utils.logging_config import setup_logging
from .agents_ensemble import AgentsEnsemble

logger = setup_logging(__name__)
logger.setLevel(logging.INFO)


def run_debate_round_n(
    prompt: str,
    agents_ensemble: AgentsEnsemble,
    output_dir: str | Path,
    round_num: int,
    json_mode: bool = False,
) -> List[dict]:
    """Run a subsequent round of debate with the given prompt and agents.

    Args:
        prompt (str): The debate prompt including previous context.
        agents_ensemble (AgentsEnsemble): Collection of LLM agents participating in
            the debate.
        output_dir (str | Path): Directory path where debate responses will be saved.
        round_num (int): Current round number.
        json_mode (bool): Whether to expect JSON responses from agents.

    Returns:
        List[dict]: List of agent responses, where each response is a dictionary.

    Raises:
        OSError: If unable to create output directory or save results.
        json.JSONDecodeError: If unable to serialize responses to JSON.
        Exception: If any error occurs when requesting agent responses.
    """
    logger.info(
        f"Starting debate round {round_num} with "
        f"{len(agents_ensemble.agents)} agents"
    )
    logger.debug(
        f"Round {round_num} prompt: "
        f"{prompt[:100]}{'...' if len(prompt) > 100 else ''}"
    )

    start_time = time.time()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.debug(f"Output directory created/verified: {output_dir}")

    try:
        logger.debug(f"Requesting responses from all agents for round {round_num}...")
        response_start_time = time.time()
        responses = agents_ensemble.get_responses(prompt=prompt, json_mode=json_mode)
        response_time = time.time() - response_start_time
        logger.info(
            f"All agent responses for round {round_num} received "
            f"in {response_time:.2f} seconds"
        )
    except Exception as e:
        logger.error(f"Error in round {round_num}: {str(e)}", exc_info=True)
        logger.error(
            f"Failed prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}"
        )
        raise

    for i, response in enumerate(responses):
        agent_id = response.get("agent_id", f"Unknown-{i}")
        response_text = response.get("response", "")
        logger.debug(
            f"Agent {agent_id} responded (#{i+1}/{len(responses)}) "
            f"in round {round_num}"
        )
        logger.debug(f"Agent {agent_id} response length: {len(response_text)} chars")

        # Log a preview of each response
        if response_text:
            preview = response_text[:100] + ("..." if len(response_text) > 100 else "")
            logger.debug(f"Agent {agent_id} response preview: {preview}")

    output_file = output_dir / f"debate_round_{round_num}.json"
    try:
        with open(output_file, "w") as f:
            json.dump(responses, f, indent=2)
        logger.info(f"Round {round_num} responses saved to {output_file}")
    except (OSError, json.JSONDecodeError) as e:
        logger.error(
            f"Failed to save responses to {output_file}: {str(e)}", exc_info=True
        )
        raise

    total_time = time.time() - start_time
    logger.info(f"Debate round {round_num} completed in {total_time:.2f} seconds")
    return responses
