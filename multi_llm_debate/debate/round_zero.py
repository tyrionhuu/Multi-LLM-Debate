import json
from pathlib import Path
from typing import List

from ..utils.logging_config import setup_logging
from .agent import LLMConnectionError
from .agents_ensemble import AgentsEnsemble

logger = setup_logging(__name__)


def run_debate_round_zero(
    prompt: str,
    agents_ensemble: AgentsEnsemble,
    output_dir: str | Path,
    json_mode: bool = False,
) -> List[dict]:
    """Run the initial round (round zero) of a debate.

    Gets responses from all agents for the initial prompt and saves them to a JSON file.
    Logs the process and response metadata.

    Args:
        prompt: The initial prompt/question to start the debate.
        agents_ensemble: Collection of LLM agents participating in the debate.
        output_dir: Directory path where debate responses will be saved.
        json_mode: Whether to expect JSON responses from agents.

    Returns:
        List[dict]: List of agent responses, where each response is a dictionary.

    Raises:
        LLMConnectionError: If there are connection issues with the LLM services.
        OSError: If unable to create output directory or save results file.
        json.JSONDecodeError: If unable to serialize responses to JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    try:
        responses = agents_ensemble.get_responses(
            prompt=prompt,
            json_mode=json_mode,
        )
    except LLMConnectionError as e:
        logger.error(f"Connection error in round zero: {str(e)}")
        raise

    for response in responses:
        logger.info(f"Agent {response['agent_id']} responded")

    output_file = output_dir / "debate_round_0.json"
    with open(output_file, "w") as f:
        json.dump(responses, f, indent=2)

    logger.info(f"Round zero responses saved to {output_file}")
    return responses
