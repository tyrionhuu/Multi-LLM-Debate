import json
from pathlib import Path
from typing import List

from ..utils.logging_config import setup_logging
from .agents_ensemble import AgentsEnsemble

logger = setup_logging(__name__)


def run_debate_round_n(
    prompt: str,
    agents_ensemble: AgentsEnsemble,
    output_dir: str | Path,
    round_num: int,
) -> List[dict]:
    """
    Run a subsequent round of debate with the given prompt and agents.

    Args:
        prompt (str): The debate prompt including previous context
        agents_ensemble (AgentsEnsemble): Collection of LLM agents participating in the debate
        output_dir (str | Path): Directory path where debate responses will be saved
        round_num (int): Current round number

    Returns:
        List[dict]: List of agent responses, where each response is a dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Running debate round {round_num}")
    # logger.info(f"Debate prompt: {prompt}")

    responses = agents_ensemble.get_responses(prompt)
    for i, response in enumerate(responses):
        logger.info(f"Agent {i} response: {response}")

    output_file = output_dir / f"debate_round_{round_num}.json"
    with open(output_file, "w") as f:
        json.dump(responses, f, indent=2)

    logger.info(f"Debate data saved to {output_file}")
    logger.info(f"Debate round {round_num} finished")
    return responses
