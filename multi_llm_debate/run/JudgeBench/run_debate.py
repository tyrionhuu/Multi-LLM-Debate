from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...debate.agents_ensemble import AgentsEnsemble
from ...debate.debate import debate
from ...llm.prompt_builder import PromptBuilder
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from ...utils.progress import progress
from .prompts import (
    build_judge_bench_round_n_prompt,
    build_judge_bench_round_zero_prompt,
)

logger = setup_logging(__name__)


def run_judge_bench(
    dataframe: pd.DataFrame,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "judge_bench",
    use_cot: bool = True,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
) -> Dict[str, Any]:
    """Run the JudgeBench task on a DataFrame.

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
    failed_entries = []
    processed_count = 0

    try:
        logger.info("Starting debate for JudgeBench task")

        # Check if the DataFrame is valid
        if not isinstance(dataframe, pd.DataFrame):
            logger.error("Invalid DataFrame type")
            raise ValueError("Dataframe must be a pandas DataFrame.")

        required_columns = ["question", "response_A", "response_B", "pair_id"]
        missing_columns = [
            col for col in required_columns if col not in dataframe.columns
        ]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        if dataframe.empty:
            logger.error("DataFrame is empty")
            raise ValueError("DataFrame is empty. Please provide valid data.")

        with progress.main_bar(
            total=len(dataframe), desc="Running debates", unit="debate"
        ) as pbar:
            for _, entry in dataframe.iterrows():
                try:
                    run_judge_bench_single_entry(
                        entry,
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
                    entry_id = entry.get("pair_id", "unknown")
                    logger.error(f"Error processing entry {entry_id}: {str(e)}")
                    failed_entries.append(
                        {
                            "id": entry_id,
                            "error": str(e),
                            "question": entry.get("question", ""),
                        }
                    )
                    continue

    except Exception as e:
        logger.error(f"Global execution error: {str(e)}", exc_info=True)
        raise RuntimeError(f"Global execution error: {str(e)}") from e

    finally:
        # Log summary
        total_entries = len(dataframe)
        logger.info(
            f"Processed {processed_count}/{total_entries} entries. Failed: {len(failed_entries)}"
        )
        if failed_entries:
            logger.error("Failed entries: " + str(failed_entries))
        success_rate = (
            (processed_count / total_entries) * 100 if total_entries > 0 else 0
        )
        logger.info(f"Success rate: {success_rate:.2f}%")
        if failed_entries:
            logger.warning("Failed entries:")
            for entry in failed_entries:
                logger.warning(f"ID: {entry['id']}, Error: {entry['error']}")
        if len(failed_entries) == total_entries:
            raise RuntimeError(
                f"All {total_entries} entries failed. Check logs for details."
            )
    # End of try block
    # Return summary
    return {
        "total_entries": total_entries,
        "processed_count": processed_count,
        "failed_entries": failed_entries,
        "success_rate": success_rate,
    }


def run_judge_bench_single_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "judge_bench",
    use_cot: bool = True,
    model_configs: Optional[List[ModelConfig]] = None,
    overwrite: bool = False,
    max_workers: Optional[int] = 4,
) -> None:
    """Run a single JudgeBench entry.

    Args:
        entry: Pandas Series containing question, response_A, response_B and pair_id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting (default: True)
        model_configs: Optional list of model configurations. If None,
                    default configs will be used.
        overwrite: Whether to overwrite existing debate results (default: False)
        max_workers: Maximum number of concurrent workers (default: 4)

    Raises:
        ValueError: If entry format is invalid
    """
    try:
        logger.info("Starting debate for entry ID: %s", entry.get("pair_id", "unknown"))

        # Check if the entry is valid
        if not isinstance(entry, pd.Series):
            logger.error("Invalid entry type")
            raise ValueError("Entry must be a pandas Series.")
        required_columns = ["question", "response_A", "response_B", "pair_id"]

        missing_columns = [
            col for col in required_columns if col not in entry or pd.isna(entry[col])
        ]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Extract values from the entry
        question = entry["question"]
        response_A = entry["response_A"]
        response_B = entry["response_B"]
        pair_id = str(entry["pair_id"])
        if not isinstance(pair_id, str):
            logger.error("Invalid pair_id type")
            raise ValueError("pair_id must be a string.")

        output_dir = base_dir / pair_id
        logger.info(f"Output directory: {output_dir}")

        # Check if response already exists
        if output_dir.exists() and not overwrite:
            debate_files = [
                output_dir / f"debate_round_{i}.json" for i in range(1, max_rounds + 1)
            ]
            debate_files_exist = any(f.exists() for f in debate_files)
            if debate_files_exist:
                logger.info(f"Debate files already exist for entry {pair_id}. ")
                return
            else:
                logger.debug(
                    f"Debate files do not exist for entry {pair_id}. Overwriting enabled."
                )

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise RuntimeError(f"Failed to create output directory: {e}")

        logger.debug("Initializing prompt builder and agents ensemble")
        prompt_builder = PromptBuilder(
            round_zero_fn=build_judge_bench_round_zero_prompt,
            round_n_fn=build_judge_bench_round_n_prompt,
            prompt_params={
                "question": question,
                "response_a": response_A,
                "response_b": response_B,
                "use_cot": use_cot,
            },
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
        logger.error(
            f"Debate execution failed for entry {entry.get('pair_id')}: {str(e)}",
            exc_info=True,
        )
        raise RuntimeError(f"Debate execution failed: {str(e)}") from e

