import json
from pathlib import Path
from typing import List

from ..utils.logging_config import setup_logging
from .agent import LLMConnectionError
from .agents_ensemble import AgentsEnsemble

logger = setup_logging(__name__)


def run_debate_round_n(
    prompt: str,
    agents_ensemble: AgentsEnsemble,
    output_dir: str | Path,
    round_num: int,
    json_mode: bool = False,
) -> List[dict]:
    """
    Run a subsequent round of debate with the given prompt and agents.

    Args:
        prompt (str): The debate prompt including previous context
        agents_ensemble (AgentsEnsemble): Collection of LLM agents participating in the debate
        output_dir (str | Path): Directory path where debate responses will be saved
        round_num (int): Current round number
        json_mode (bool): Whether to expect JSON responses from agents

    Returns:
        List[dict]: List of agent responses, where each response is a dictionary

    Raises:
        LLMConnectionError: If there are connection issues with the LLM services
        OSError: If unable to create output directory or save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Running debate round {round_num}")

    try:
        responses = agents_ensemble.get_responses(prompt=prompt, json_mode=json_mode)
    except LLMConnectionError as e:
        logger.error(f"Connection error in round {round_num}: {str(e)}")
        raise

    for i, response in enumerate(responses):
        logger.info(f"Agent {i} response: {response}")

    output_file = output_dir / f"debate_round_{round_num}.json"
    with open(output_file, "w") as f:
        json.dump(responses, f, indent=2)

    logger.info(f"Debate data saved to {output_file}")
    logger.info(f"Debate round {round_num} finished")
    return responses
