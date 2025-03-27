import logging
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
from .utils import extract_caption_a_b_answer

# Fix the setup_logging call by removing the level parameter
logger = setup_logging(__name__)
# Set debug level separately
logger.setLevel(logging.INFO)


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

        required_columns = ["question", "response_A", "response_B", "id"]
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
                    entry_id = entry.get("id", "unknown")
                    logger.error(f"Error processing entry {entry_id}: {str(e)}")
                    failed_entries.append(
                        {
                            "id": entry_id,
                            "error": str(e),
                            "question": entry.get("question", ""),
                        }
                    )
                    pbar.update(1)  # Update progress even for failures
                    continue

    except Exception as e:
        logger.error(f"Global execution error: {str(e)}", exc_info=True)
        raise RuntimeError(f"Global execution error: {str(e)}") from e

    finally:
        # Log summary
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

        if len(failed_entries) == total_entries and total_entries > 0:
            logger.error(f"All {total_entries} entries failed. Check logs for details.")

    # Return summary with consistent format
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
        entry: Pandas Series containing question, response_A, response_B and id
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
        logger.info("Starting debate for entry ID: %s", entry.get("id", "unknown"))

        # Check if the entry is valid
        if not isinstance(entry, pd.Series):
            logger.error("Invalid entry type")
            raise ValueError("Entry must be a pandas Series.")
        required_columns = ["question", "response_A", "response_B", "id"]

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
        id = str(entry["id"])
        if not isinstance(id, str):
            logger.error("Invalid id type")
            raise ValueError("id must be a string.")

        output_dir = base_dir / id
        logger.info(f"Output directory: {output_dir}")

        # Check if response already exists
        if output_dir.exists() and not overwrite:
            round_files = list(output_dir.glob("debate_round_*.json"))
            if len(round_files) > 0:
                logger.info(
                    f"Partial or complete debate files found for entry {id}. "
                    "Skipping due to overwrite=False."
                )
                return

        elif output_dir.exists() and overwrite:
            logger.info(
                f"Overwrite enabled for entry {id}. Will regenerate all debate files."
            )
            # When overwrite is True, we continue execution and the existing files
            # will be overwritten when debate() writes new results

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

        logger.debug("Model configurations: %s", model_configs)
        agents_ensemble = AgentsEnsemble(
            config_list=model_configs, max_workers=max_workers
        )

        logger.info("Starting debate execution")
        # Execute debate and capture the results for logging
        debate_results = debate(
            max_rounds=max_rounds,
            prompt_builder=prompt_builder,
            agents_ensemble=agents_ensemble,
            output_dir=output_dir,
            process_answer=extract_caption_a_b_answer,
        )

        # Log model outputs in debug mode
        if debate_results:
            logger.debug("Debate results summary:")
            for round_num, round_data in enumerate(debate_results):
                logger.debug(f"Round {round_num} output:")
                if isinstance(round_data, dict):
                    for agent_id, response in round_data.items():
                        logger.debug(f"Agent {agent_id}: {response}")
                else:
                    logger.debug(f"Round data: {str(round_data)}")

        logger.info("Debate completed successfully")

    except Exception as e:
        logger.error(
            f"Debate execution failed for entry {entry.get('id')}: {str(e)}",
            exc_info=True,
        )
        raise RuntimeError(f"Debate execution failed: {str(e)}") from e


def main() -> None:
    """Run the JudgeBench task."""
    from ...utils.model_config import ModelConfig

    id = "e302b0a0-28d5-5a3c-b1af-fedcf5543e72"
    question = "A college student initiated a criminal case against his former roommate, claiming the former roommate snuck in and removed his state-of-the-art home theater equipment from his house. The college student took the complaint papers that were approved by the prosecution and filed them with a magistrate court. An ongoing police investigation later determined, however, that the college student was a vengeful person not averse to creating stories, and that he and the former roommate had been arguing over several related issues. They also determined that he never had any quality sound or video equipment. The police dropped the prosecution, and the criminal case was dismissed at the preliminary hearing. When the former roommate filed a civil case against the college student alleging the tort of malicious prosecution, will evidence of the student's reputation for dishonesty be admissible and why?\n(A) Yes, because the student's reputation for dishonesty directly relates to the tort of malicious prosecution.\n(B) No, because reputation evidence is considered too prejudicial to be admissible.\n(C) Yes, because dishonesty at any time in the past is always relevant to a malicious prosecution complaint.\n(D) Yes, because the student's dishonesty directly affects the credibility of his initial criminal complaint.\n(E) Yes, because reputation evidence, just like most character evidence, is virtually always admissible.\n(F) Yes, because even though character evidence is too collateral to be admitted for circumstantial evidence, it is admissible if it is directly at issue in the case.\n(G) No, because character evidence is only admissible if it is directly related to the act in question.\n(H) No, because the past reputation of a person is not considered as evidence in court.\n(I) Yes, because while most character evidence is not admissible, some is, and reputation evidence is always admissible.\n(J) No, because character evidence is never admissible in any case.\nIf you cannot determine the correct multiple-choice answer, take your best guess. Once you have your answer, please duplicate that letter five times in a single string. For example, if the answer is K, then write KKKKK.\nLet's think step by step."
    response_A = "To determine if evidence of the student's reputation for dishonesty is admissible in a claim of malicious prosecution, we need to consider several legal principles, especially those concerning character evidence and its relevance.\n\n1. **Relevance to Malicious Prosecution**:\n   - Malicious prosecution involves proving that the defendant initiated a legal action without probable cause and with malice. Evidence suggesting the initiator's dishonesty could help show the lack of probable cause and the element of malice.\n\n2. **General Rules on Character Evidence**:\n   - Generally, character evidence (reputation, opinion, or specific instances of conduct) is not admissible to prove conduct on a particular occasion (Federal Rules of Evidence 404(a)).\n   - There are exceptions where character evidence is admissible, typically in cases where character itself is an essential element of a claim or defense (e.g., defamation, negligent hiring, etc.).\n\n3. **Specific to Malicious Prosecution**:\n   - For malicious prosecution, the plaintiff would need to show that the college student acted with malice in initiating the criminal case.\n   - The student's reputation for dishonesty is relevant to understanding whether the student acted with malice or fabricated the entire situation.\n\nAnalyzing the options step-by-step:\n\n(A) Yes, because the student's reputation for dishonesty directly relates to the tort of malicious prosecution.\n   - This is a strong consideration since dishonesty could indicate malice and lack of probable cause.\n\n(B) No, because reputation evidence is considered too prejudicial to be admissible.\n   - While reputation evidence can be prejudicial, it is not an absolute bar, especially when the character is a pertinent element.\n\n(C) Yes, because dishonesty at any time in the past is always relevant to a malicious prosecution complaint.\n   - The phrase \"always relevant\" is too broad and may be misleading.\n\n(D) Yes, because the student's dishonesty directly affects the credibility of his initial criminal complaint.\n   - This is true, but it should focus on relevance to the malicious prosecution claim rather than just general credibility.\n\n(E) Yes, because reputation evidence, just like most character evidence, is virtually always admissible.\n   - Incorrect, as most character evidence rules say it is generally not admissible unless specific exceptions apply.\n\n(F) Yes, because even though character evidence is too collateral to be admitted for circumstantial evidence, it is admissible if it is directly at issue in the case.\n   - This correctly addresses that character is directly at issue in the case of malicious prosecution.\n\n(G) No, because character evidence is only admissible if it is directly related to the act in question.\n   - This is correct but incomplete in understanding how the student's character for dishonesty specifically relates to malice and the claim.\n\n(H) No, because the past reputation of a person is not considered as evidence in court.\n   - Incorrect, past reputation can be considered where it is directly related to the issues of the case.\n\n(I) Yes, because while most character evidence is not admissible, some is, and reputation evidence is always admissible.\n   - Incorrect, not always admissible.\n\n(J) No, because character evidence is never admissible in any case.\n   - Incorrect, as there are specific contexts in which it is admissible.\n\nThe best-balanced reasoning aligns with choice (F), as it clearly states that when character is directly at issue, such as in a malicious prosecution claim, it becomes admissible.\n\nHence my answer is:\nFFFFF"
    response_B = "To determine whether the former roommate can use evidence of the college student's reputation for dishonesty in his civil case for malicious prosecution, we need to consider rules of admissibility of character evidence, especially reputation evidence, in the context of a civil lawsuit.\n\n1. **Understand the nature of the civil lawsuit**:\n   - The case is about malicious prosecution, which involves proving that the defendant (college student) initiated an unjust legal proceeding with malice and without probable cause.\n\n2. **Evaluate the rules of evidence for character/reputation**:\n   - In legal proceedings, character evidence is generally inadmissible to prove that a person acted in accordance with that character on a specific occasion.\n   - However, character evidence can be admissible if it is directly related to proving an element of a claim, defense, or charge.\n\n3. **Relevance to malicious prosecution**:\n   - Malicious prosecution claims often focus on the intent and credibility of the defendant.\n   - The student's reputation for dishonesty might be directly relevant to showing that the initial complaint was maliciously false.\n\n4. **Admissibility based on relevance**:\n   - Federal Rules of Evidence (FRE) and many state rules allow the introduction of character evidence if the trait is pertinent to the issue at hand. In this case, dishonesty is a core aspect of the malicious prosecution claim.\n   - The character trait of dishonesty would be directly relevant because it pertains to the potential fabrication of the initial complaint.\n\nBased on these considerations, the best answer is:\n(A) Yes, because the student's reputation for dishonesty directly relates to the tort of malicious prosecution.\n\nSo, the correct and best-supported answer is:\nAAAAA"

    dataframe = pd.DataFrame(
        {
            "question": [question],
            "response_A": [response_A],
            "response_B": [response_B],
            "id": [id],
        }
    )

    model_config = ModelConfig(
        provider="ollama",
        name="llama3",
        quantity=3,
    )
    model_configs = [model_config]

    run_judge_bench(
        dataframe=dataframe,
        max_rounds=5,
        base_dir=Path("data") / "judge_bench",
        use_cot=True,
        model_configs=model_configs,
        overwrite=False,
        max_workers=4,
    )


if __name__ == "__main__":
    main()
