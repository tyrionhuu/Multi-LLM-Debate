from pathlib import Path
from typing import List

from ..llm.prompts import PromptBuilder
from ..utils.logging_config import setup_logging
from .agents_ensemble import AgentsEnsemble
from .round_n import run_debate_round_n
from .round_zero import run_debate_round_zero

logger = setup_logging(__name__)


def run_debate(
    max_rounds: int,
    prompt_builder: PromptBuilder,
    agents_ensemble: AgentsEnsemble,
    output_dir: str | Path,
) -> List[List[dict]]:
    """Run a full debate with multiple rounds using the given prompts and agents.

    Coordinates multiple rounds of debate between agents, starting with round zero
    and continuing through subsequent rounds. Logs progress and saves results.

    Args:
        max_rounds: Maximum number of debate rounds to run.
        prompt_builder: PromptBuilder instance to generate prompts for each round.
        agents_ensemble: Collection of LLM agents participating in the debate.
        output_dir: Directory path where debate responses will be saved.

    Returns:
        List[List[dict]]: List of responses from each round, where each round's
            responses is a list of dictionaries containing agent responses.

    Raises:
        Exception: If any error occurs during the debate process.
            Original exception is logged and re-raised.
    """
    logger.info(f"Starting debate with {len(agents_ensemble)} agents")
    logger.info(f"Maximum rounds: {max_rounds}")
    logger.info(f"Output directory: {output_dir}")

    all_responses = []

    try:
        for i in range(max_rounds):
            logger.info(f"Starting debate round {i}")
            if i == 0:
                prompt = prompt_builder.build_round_zero()
                round_responses = run_debate_round_zero(
                    prompt, agents_ensemble, output_dir
                )
            else:
                prompt = prompt_builder.build_round_n(all_responses[i - 1])
                round_responses = run_debate_round_n(
                    prompt, agents_ensemble, output_dir, i
                )
            all_responses.append(round_responses)
            logger.info(f"Completed debate round {i}")

        logger.info("Debate completed successfully")
        return all_responses
    except Exception as e:
        logger.error(f"Error during debate: {str(e)}", exc_info=True)
        raise
