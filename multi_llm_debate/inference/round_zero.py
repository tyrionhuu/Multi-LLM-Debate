import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..llm.llm import call_model

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

# def round_zero_debate(prompt: str, data_dir: Path) -> List[Dict[str, str]]:
#     """
#     Run the zeroth round of the debate.

#     Args:
#         prompt (str): The debate prompt.
#         data_dir (Path): Path to the data directory.

#     Returns:
#         List[Dict[str, str]]: A list of responses from the LLMs.
#     """
