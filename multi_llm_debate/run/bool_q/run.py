from pathlib import Path
from typing import List, Optional

import pandas as pd

from ...debate.agent import Agent
from ...debate.agents_ensemble import AgentsEnsemble
from ...debate.run_debate import run_debate
from ...llm.prompts import (
    PromptBuilder,
    build_bool_q_round_n_prompt,
    build_bool_q_round_zero_prompt,
)
from ...utils.logging_config import setup_logging

logger = setup_logging(__name__)


def run_bool_q(
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
    base_dir: Path = Path("data" / "bool_q"),
    use_cot: bool = True,
    agents: Optional[List[Agent]] = None,
) -> None:
    """Run the Boolean Question task on a DataFrame.

    Args:
        dataframe: Pandas DataFrame containing question, answer, passage and id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting (default: True)
        agents: Optional list of Agent instances to use in debate. If None,
               default agents will be used.

    Raises:
        ValueError: If DataFrame format is invalid
        RuntimeError: If debate execution fails
    """
    try:
        logger.info("Starting debate for Boolean Question task")

        # Check if the DataFrame is valid
        if not isinstance(dataframe, pd.DataFrame):
            logger.error("Invalid DataFrame type")
            raise ValueError("Dataframe must be a pandas DataFrame.")

        required_columns = ["question", "answer", "passage", "id"]
        missing_columns = [
            col for col in required_columns if col not in dataframe.columns
        ]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(
                "DataFrame must contain 'question', 'answer', 'passage', and 'id' columns."
            )

        # Iterate over each entry in the DataFrame
        for _, entry in dataframe.iterrows():
            run_bool_q_single_entry(entry, max_rounds, base_dir, use_cot, agents)

    except Exception as e:
        logger.error(f"Debate execution failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Debate execution failed: {str(e)}") from e


def run_bool_q_single_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    base_dir: Path = Path("data" / "bool_q"),
    use_cot: bool = True,
    agents: Optional[List[Agent]] = None,
) -> None:
    """Run a single entry for the Boolean Question task.

    Args:
        entry: Pandas Series containing question, answer, passage and id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting (default: True)
        agents: Optional list of Agent instances to use in debate. If None,
               default agents will be used.

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
        passage = entry["passage"]
        _id = entry["id"]

        output_dir = base_dir / _id
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
            prompt_params={
                "question": question,
                "passage": passage,
                "use_cot": use_cot,
            },
        )
        agents_ensemble = AgentsEnsemble(agents=agents)

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
