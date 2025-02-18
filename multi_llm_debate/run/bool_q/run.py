from pathlib import Path

import pandas as pd

from ...debate.agents_ensemble import AgentsEnsemble
from ...debate.run_debate import run_debate
from ...llm.prompts import (
    PromptBuilder,
    build_bool_q_round_n_prompt,
    build_bool_q_round_zero_prompt,
)
from ...utils.logging_config import setup_logging

logger = setup_logging(__name__)


def run_bool_q_single_entry(
    entry: pd.Series, max_rounds: int = 10, base_dir: Path = Path("data" / "bool_q")
) -> None:
    """Run a single entry for the Boolean Question task.

    Args:
        entry: Pandas Series containing question, answer, passage and id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files

    Raises:
        ValueError: If entry format is invalid
        RuntimeError: If debate execution fails
    """
    try:
        logger.info(f"Starting debate for entry ID: {entry.get('id', 'unknown')}")

        # Check if the entry is valid
        if not isinstance(entry, pd.Series):
            logger.error("Invalid entry type")
            raise ValueError("Entry must be a pandas Series.")

        required_fields = ["question", "answer", "passage", "id"]
        missing_fields = [field for field in required_fields if field not in entry]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            raise ValueError(
                "Entry must contain 'question', 'answer', 'passage', and 'id'."
            )

        # Extract values from the entry
        question = entry["question"]
        answer = entry["answer"]
        passage = entry["passage"]
        id = entry["id"]

        output_dir = base_dir / id
        logger.debug(f"Output directory set to: {output_dir}")

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory: {e}")
            raise RuntimeError(f"Failed to create output directory: {e}")

        # Initialize components
        logger.debug("Initializing prompt builder and agents ensemble")
        prompt_builder = PromptBuilder(
            round_zero_fn=build_bool_q_round_zero_prompt,
            round_n_fn=build_bool_q_round_n_prompt,
        )
        agents_ensemble = AgentsEnsemble()

        # Run the debate
        logger.info("Starting debate execution")
        run_debate(
            agents_ensemble,
            prompt_builder,
            max_rounds=max_rounds,
            output_dir=output_dir,
        )
        logger.info("Debate completed successfully")

    except Exception as e:
        logger.error(f"Debate execution failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Debate execution failed: {str(e)}") from e
