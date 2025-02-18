from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from tqdm import tqdm

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
    base_dir: Path = Path("data") / "bool_q",
    use_cot: bool = True,
    agents: Optional[List[Agent]] = None,
) -> Dict[str, Any]:
    """Run the Boolean Question task on a DataFrame.

    Args:
        dataframe: Pandas DataFrame containing question, answer, passage and id
        max_rounds: Maximum number of debate rounds
        base_dir: Base directory for output files
        use_cot: Whether to use chain-of-thought prompting (default: True)
        agents: Optional list of Agent instances to use in debate. If None,
               default agents will be used.

    Returns:
        Dict containing summary of execution including failed entries

    Raises:
        ValueError: If DataFrame format is invalid
    """
    failed_entries = []
    processed_count = 0
    
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

        # Iterate over each entry in the DataFrame with progress bar
        for _, entry in tqdm(
            dataframe.iterrows(),
            total=len(dataframe),
            desc="Running debates",
            unit="debate",
        ):
            try:
                run_bool_q_single_entry(entry, max_rounds, base_dir, use_cot, agents)
                processed_count += 1
            except Exception as e:
                entry_id = entry.get('id', 'unknown')
                logger.error(f"Failed to process entry {entry_id}: {str(e)}")
                failed_entries.append({
                    'id': entry_id,
                    'error': str(e)
                })
                continue

    except Exception as e:
        logger.error(f"Global execution error: {str(e)}", exc_info=True)
        raise RuntimeError(f"Global execution error: {str(e)}") from e
    
    finally:
        # Log summary
        total_entries = len(dataframe)
        failed_count = len(failed_entries)
        success_rate = (processed_count / total_entries) * 100 if total_entries > 0 else 0
        
        logger.info(f"Debate execution completed")
        logger.info(f"Total entries processed: {total_entries}")
        logger.info(f"Successful: {processed_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        
        if failed_entries:
            logger.warning("Failed entries:")
            for entry in failed_entries:
                logger.warning(f"ID: {entry['id']}, Error: {entry['error']}")
    
    return {
        'total_entries': total_entries,
        'processed_count': processed_count,
        'failed_entries': failed_entries,
        'success_rate': success_rate
    }


def run_bool_q_single_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    base_dir: Path = Path("data") / "bool_q",
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
        # print(f"Question: {question}")
        passage = entry["passage"]
        # print(f"Passage: {passage}")
        id_ = str(entry["id"])  # Convert ID to string

        output_dir = base_dir / id_  # Now using string ID with Path
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
            max_rounds=max_rounds,
            prompt_builder=prompt_builder,
            agents_ensemble=agents_ensemble,
            output_dir=output_dir,
        )
        logger.info("Debate completed successfully")

    except Exception as e:
        logger.error(f"Debate execution failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Debate execution failed: {str(e)}") from e
