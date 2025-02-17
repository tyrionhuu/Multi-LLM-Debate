import json
from pathlib import Path

from ..utils.logging_config import setup_logging
from .agents_ensemble import AgentsEnsemble

logger = setup_logging(__name__)

def run_debate_round_zero(
    prompt: str, agents_ensemble: AgentsEnsemble, output_dir: str | Path
) -> None:
    """Run the initial round (round zero) of a debate."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Starting round zero with prompt: {prompt}")
    responses = agents_ensemble.get_responses(prompt)
    
    for response in responses:
        logger.info(f"Agent {response['agent_id']} responded")
        logger.debug(f"Response content: {response}")

    output_file = output_dir / "debate_round_0.json"
    with open(output_file, "w") as f:
        json.dump(responses, f, indent=2)

    logger.info(f"Round zero responses saved to {output_file}")
