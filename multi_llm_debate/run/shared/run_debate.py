from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ...debate.agents_ensemble import AgentsEnsemble
from ...debate.debate import debate
from ...llm.prompt_builder import PromptBuilder
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig

logger = setup_logging(__name__)


def run_debate_single_entry(
    entry: pd.Series,
    round_zero_fn: Callable,
    round_n_fn: Callable,
    prompt_params: Dict[str, Any],
    max_rounds: int = 10,
    base_dir: Path = Path("data"),
    use_cot: bool = True,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
) -> None:
    """Run a single debate entry with configurable prompt functions.

    Args:
        entry: Pandas Series containing the debate entry data
        round_zero_fn: Function to build the initial round prompt
        round_n_fn: Function to build subsequent round prompts
        prompt_params: Parameters to pass to the prompt builder
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting
        model_configs: Optional list of model configurations
        overwrite: Whether to overwrite existing debate results
        max_workers: Maximum number of concurrent workers

    Raises:
        ValueError: If entry format is invalid
        RuntimeError: If debate execution fails
    """
    try:
        id_ = str(entry.get("id", "unknown"))
        logger.info(f"Starting debate for entry ID: {id_}")

        if not isinstance(entry, pd.Series):
            logger.error("Invalid entry type")
            raise ValueError("Entry must be a pandas Series.")

        output_dir = base_dir / id_
        logger.debug(f"Output directory set to: {output_dir}")

        if output_dir.exists() and not overwrite:
            debate_files = [
                output_dir / f"debate_round_{i}.json" for i in range(max_rounds)
            ]
            if any(f.exists() for f in debate_files):
                logger.info(f"Skipping entry {id_} - debate results exist")
                return

        output_dir.mkdir(parents=True, exist_ok=True)

        prompt_params["use_cot"] = use_cot
        prompt_builder = PromptBuilder(
            round_zero_fn=round_zero_fn,
            round_n_fn=round_n_fn,
            prompt_params=prompt_params,
        )
        agents_ensemble = AgentsEnsemble(
            config_list=model_configs, max_workers=max_workers
        )

        logger.info("Starting debate execution")
        debate(
            max_rounds=max_rounds,
            prompt_builder=prompt_builder,
            agents_ensemble=agents_ensemble,
            output_dir=output_dir,
        )
        logger.info("Debate completed successfully")

    except Exception as e:
        logger.error(f"Debate execution failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Debate execution failed: {str(e)}") from e
