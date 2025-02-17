import logging
from datetime import datetime
from pathlib import Path

from .agents_ensemble import AgentsEnsemble

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Configure logging
log_file = log_dir / f'debate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

def run_debate_round_zero(prompt: str, agents_ensemble: AgentsEnsemble) -> None:
    logger.info(f"Running debate round with prompt: {prompt}")
    responses = agents_ensemble.get_responses(prompt)
    for i, response in enumerate(responses):
        logger.info(f"Agent {i} response: {response}")

    logger.info("Debate round finished")