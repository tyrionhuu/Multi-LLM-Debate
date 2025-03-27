from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...debate.agents_ensemble import AgentsEnsemble
from ...debate.debate import debate
from ...llm.prompt_builder import PromptBuilder
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from ..shared.run import run_debate_task
from .prompts import build_bool_q_round_n_prompt, build_bool_q_round_zero_prompt
from .utils import extract_bool_answer

logger = setup_logging(__name__)


def run_bool_q(
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "bool_q",
    use_cot: bool = True,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
) -> Dict[str, Any]:
    """Run the Boolean Question task on a DataFrame.

    Args:
        dataframe: Pandas DataFrame containing question, answer, passage and id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting (default: True)
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        overwrite: Whether to overwrite existing debate results (default: False)
        max_workers: Maximum number of concurrent workers (default: 4)

    Returns:
        Dict containing summary of execution including failed entries

    Raises:
        ValueError: If DataFrame format is invalid
    """
    required_columns = ["question", "answer", "passage", "id"]
    
    return run_debate_task(
        dataframe=dataframe,
        process_entry_fn=run_bool_q_single_entry,
        required_columns=required_columns,
        base_dir=base_dir,
        max_rounds=max_rounds,
        use_cot=use_cot,
        model_configs=model_configs,
        overwrite=overwrite,
        max_workers=max_workers,
        task_name="Boolean Question task",
    )


def run_bool_q_single_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "bool_q",
    use_cot: bool = True,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
) -> None:
    """Run a single entry for the Boolean Question task.

    Args:
        entry: Pandas Series containing question, answer, passage and id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting (default: True)
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        overwrite: Whether to overwrite existing debate results (default: False)
        max_workers: Maximum number of concurrent workers (default: 4)

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
        id_ = str(entry["id"])

        output_dir = base_dir / id_
        logger.debug(f"Output directory set to: {output_dir}")

        # Check if response already exists
        if output_dir.exists() and not overwrite:
            debate_files = [
                output_dir / f"debate_round_{i}.json" for i in range(max_rounds)
            ]
            debate_files_exist = any(f.exists() for f in debate_files)

            if debate_files_exist:
                logger.info(f"Skipping entry {id_} - debate results exist")
                return
            else:
                logger.debug(f"Directory exists but no debate files found for {id_}")

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
        agents_ensemble = AgentsEnsemble(
            config_list=model_configs, max_workers=max_workers
        )

        # Run the debate
        logger.info("Starting debate execution")
        debate(
            max_rounds=max_rounds,
            prompt_builder=prompt_builder,
            agents_ensemble=agents_ensemble,
            output_dir=output_dir,
            process_answer=extract_bool_answer,
        )
        logger.info("Debate completed successfully")

    except Exception as e:
        logger.error(f"Debate execution failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Debate execution failed: {str(e)}") from e
