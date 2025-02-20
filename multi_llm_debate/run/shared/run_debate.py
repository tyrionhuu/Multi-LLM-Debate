from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...debate.agents_ensemble import AgentsEnsemble
from ...debate.debate import debate
from ...llm.prompt_builder import PromptBuilder
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from ...utils.progress import progress

logger = setup_logging(__name__)


def run_debate_single_entry(
    entry: pd.Series,
    prompt_builder: PromptBuilder,
    required_columns: List[str],
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
        prompt_builder: Configured PromptBuilder instance
        required_columns: List of required column names in the entry
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
        if not isinstance(entry, pd.Series):
            logger.error("Invalid entry type")
            raise ValueError("Entry must be a pandas Series.")

        # Validate required columns
        missing_columns = [col for col in required_columns if col not in entry.index]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(
                f"Entry must contain columns: {', '.join(required_columns)}"
            )

        id_ = str(entry.get("id", "unknown"))
        logger.info(f"Starting debate for entry ID: {id_}")

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

        prompt_builder.prompt_params["use_cot"] = use_cot
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


def run_debate(
    dataframe: pd.DataFrame,
    prompt_builder: PromptBuilder,
    required_columns: List[str],
    max_rounds: int = 10,
    base_dir: Path = Path("data"),
    use_cot: bool = True,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
) -> Dict[str, Any]:
    """Run debates on a DataFrame with configurable prompt functions.

    Args:
        dataframe: Pandas DataFrame containing the debate entries
        prompt_builder: Configured PromptBuilder instance
        required_columns: List of required column names in the DataFrame
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting
        model_configs: Optional list of model configurations
        overwrite: Whether to overwrite existing debate results
        max_workers: Maximum number of concurrent workers

    Returns:
        Dict containing summary of execution including failed entries

    Raises:
        ValueError: If DataFrame format is invalid
    """
    failed_entries = []
    processed_count = 0

    try:
        logger.info("Starting debate execution")

        if not isinstance(dataframe, pd.DataFrame):
            logger.error("Invalid DataFrame type")
            raise ValueError("Dataframe must be a pandas DataFrame.")

        missing_columns = [
            col for col in required_columns if col not in dataframe.columns
        ]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(
                f"DataFrame must contain columns: {', '.join(required_columns)}"
            )

        from ..shared.utils import build_config_desc

        config_desc = build_config_desc(model_configs, use_cot, max_rounds)
        with progress.main_bar(
            total=len(dataframe), desc=f"Running debates [{config_desc}]", unit="debate"
        ) as pbar:
            for _, entry in dataframe.iterrows():
                try:
                    run_debate_single_entry(
                        entry=entry,
                        prompt_builder=prompt_builder,
                        required_columns=required_columns,
                        max_rounds=max_rounds,
                        base_dir=base_dir,
                        use_cot=use_cot,
                        model_configs=model_configs,
                        overwrite=overwrite,
                        max_workers=max_workers,
                    )
                    processed_count += 1
                    pbar.update(1)
                except Exception as e:
                    entry_id = entry.get("id", "unknown")
                    logger.error(f"Failed to process entry {entry_id}: {str(e)}")
                    failed_entries.append({"id": entry_id, "error": str(e)})
                    continue

    except Exception as e:
        logger.error(f"Global execution error: {str(e)}", exc_info=True)
        raise RuntimeError(f"Global execution error: {str(e)}") from e

    finally:
        total_entries = len(dataframe)
        failed_count = len(failed_entries)
        success_rate = (
            (processed_count / total_entries) * 100 if total_entries > 0 else 0
        )

        logger.info("Debate execution completed")
        logger.info(f"Total entries processed: {total_entries}")
        logger.info(f"Successful: {processed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Success rate: {success_rate:.2f}%")

        if failed_entries:
            logger.warning("Failed entries:")
            for entry in failed_entries:
                logger.warning(f"ID: {entry['id']}, Error: {entry['error']}")

    return {
        "total_entries": total_entries,
        "processed_count": processed_count,
        "failed_entries": failed_entries,
        "success_rate": success_rate,
    }
