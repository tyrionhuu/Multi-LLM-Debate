from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...debate.agents_ensemble import AgentsEnsemble
from ...debate.debate import debate
from ...llm.prompt_builder import PromptBuilder
from ...llm.prompts import build_bool_q_round_n_prompt, build_bool_q_round_zero_prompt
from ...utils.logging_config import setup_logging
from ...utils.model_config import ModelConfig
from ...utils.progress import progress

logger = setup_logging(__name__)


# def run_judge_bench(
#     dataframe: pd.DataFrame,
#     max_rounds: int = 10,
#     base_dir: Path = Path("data") / "judge_bench",
#     use_cot: bool = True,
#     model_configs: Optional[List[ModelConfig]] = None,
#     overwrite: bool = False,
#     max_workers: Optional[int] = 4,
# ) -> Dict[str, Any]:
#     """Run the JudgeBench task on a DataFrame.

#     Args:
#         dataframe: Pandas DataFrame containing question, answer, passage and id
#         max_rounds: Maximum number of debate rounds
#         base_dir: Base directory for output files
#         use_cot: Whether to use chain-of-thought prompting (default: True)
#         model_configs: Optional list of model configurations. If None,
#                     default configs will be used.
#         overwrite: Whether to overwrite existing debate results (default: False)
#         max_workers: Maximum number of concurrent workers (default: 4)

#     Returns:
#         Dict containing summary of execution including failed entries

#     Raises:
#         ValueError: If DataFrame format is invalid
#     """
#     failed_entries = []
#     processed_count = 0

#     try:
#         logger.info("Starting debate for JudgeBench task")

#         # Check if the DataFrame is valid
#         if not isinstance(dataframe, pd.DataFrame):
#             logger.error("Invalid DataFrame type")
#             raise ValueError("Dataframe must be a pandas DataFrame.")

#         required_columns = ["question", "response_A", "response_B", "pair_id"]
#         missing_columns = [
#             col for col in required_columns if col not in dataframe.columns
#         ]
#         if missing_columns:
#             logger.error(f"Missing required columns: {missing_columns}")
#             raise ValueError(f"Missing required columns: {missing_columns}")
#         if dataframe.empty:
#             logger.error("DataFrame is empty")
#             raise ValueError("DataFrame is empty. Please provide valid data.")

#         with progress.main_bar(
#             total=len(dataframe), desc="Running debates", unit="debate"
#         ) as pbar:
#             for _, entry in dataframe.iterrows():
#                 try:
#                     run_judge_bench_single_entry(
#                         entry,
#                         max_rounds=max_rounds,
#                         base_dir=base_dir,
#                         use_cot=use_cot,
#                         model_configs=model_configs,
#                         overwrite=overwrite,
#                         max_workers=max_workers,
#                     )
#                     processed_count += 1
#                     pbar.update(1)
#                 except Exception as e:
#                     entry_id = entry.get("pair_id", "unknown")
#                     logger.error(f"Error processing entry {entry_id}: {str(e)}")
#                     failed_entries.append(
#                         {
#                             "id": entry_id,
#                             "error": str(e),
#                             "question": entry.get("question", ""),
#                         }
#                     )
#                     continue

#     except Exception as e:
#         logger.error(f"Global execution error: {str(e)}", exc_info=True)
#         raise RuntimeError(f"Global execution error: {str(e)}") from e

#     finally:
#         # Log summary
#         total_entries = len(dataframe)
#         logger.info(
#             f"Processed {processed_count}/{total_entries} entries. Failed: {len(failed_entries)}"
#         )
#         if failed_entries:
#             logger.error("Failed entries: " + str(failed_entries))
#         success_rate = (
#             (processed_count / total_entries) * 100 if total_entries > 0 else 0
#         )
#         logger.info(f"Success rate: {success_rate:.2f}%")
#         if failed_entries:
#             logger.warning("Failed entries:")
#             for entry in failed_entries:
#                 logger.warning(f"ID: {entry['id']}, Error: {entry['error']}")
#         if len(failed_entries) == total_entries:
#             raise RuntimeError(
#                 f"All {total_entries} entries failed. Check logs for details."
#             )
#     # End of try block
#     # Return summary
#     return {
#         "total_entries": total_entries,
#         "processed_count": processed_count,
#         "failed_entries": failed_entries,
#         "success_rate": success_rate,
#     }
