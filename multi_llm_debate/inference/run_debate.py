from pathlib import Path

from ..utils.logging_config import setup_logging
from .agents_ensemble import AgentsEnsemble
from .round_n import run_debate_round_n
from .round_zero import run_debate_round_zero

logger = setup_logging(__name__)

def run_debate(
    max_rounds: int,
    round_zero_prompt: str,
    round_n_prompt: str,
    agents_ensemble: AgentsEnsemble,
    output_dir: str | Path,
) -> None:
    """Run a full debate with multiple rounds using the given prompts and agents."""
    logger.info(f"Starting debate with {len(agents_ensemble)} agents")
    logger.info(f"Maximum rounds: {max_rounds}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        for i in range(max_rounds):
            logger.info(f"Starting debate round {i}")
            if i == 0:
                run_debate_round_zero(round_zero_prompt, agents_ensemble, output_dir)
            else:
                run_debate_round_n(round_n_prompt, agents_ensemble, output_dir, i)
            logger.info(f"Completed debate round {i}")
        
        logger.info("Debate completed successfully")
    except Exception as e:
        logger.error(f"Error during debate: {str(e)}", exc_info=True)
        raise