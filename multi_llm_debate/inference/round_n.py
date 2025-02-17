import json
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

def run_debate_round_zero(prompt: str, agents_ensemble: AgentsEnsemble, data_dir: str | Path) -> None:
    """
    Run the initial round (round zero) of a debate with the given prompt and agents.
    
    Args:
        prompt (str): The initial prompt/question to start the debate
        agents_ensemble (AgentsEnsemble): Collection of LLM agents participating in the debate
        data_dir (str | Path): Directory path where debate responses will be saved
    
    Returns:
        None: Results are saved to file and logged
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    logger.info(f"Running debate round with prompt: {prompt}")
    responses = agents_ensemble.get_responses(prompt)
    for i, response in enumerate(responses):
        logger.info(f"Agent {i} response: {response}")

    output_file = data_dir / f'debate_round_0.json'
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    logger.info(f"Debate data saved to {output_file}")
    logger.info("Debate round finished")