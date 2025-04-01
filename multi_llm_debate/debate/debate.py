import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from ..llm.prompt_builder import PromptBuilder
from ..utils.logging_config import setup_logging
from .agents_ensemble import AgentsEnsemble
from .round_n import run_debate_round_n
from .round_zero import run_debate_round_zero

logger = setup_logging(__name__)
logger.setLevel(logging.INFO)


def debate(
    max_rounds: int,
    prompt_builder: PromptBuilder,
    agents_ensemble: AgentsEnsemble,
    output_dir: str | Path,
    json_mode: bool = False,
    process_answer_func: Optional[Callable] = None,
) -> List[List[dict]]:
    """Run a full debate with multiple rounds using the given prompts and agents.

    Coordinates multiple rounds of debate between agents, starting with round zero
    and continuing through subsequent rounds. Logs progress and saves results.
    Files are only saved if the debate completes successfully. If interrupted,
    all generated files from this debate are deleted.

    Args:
        max_rounds: Maximum number of debate rounds to run.
        prompt_builder: PromptBuilder instance to generate prompts for each round.
        agents_ensemble: Collection of LLM agents participating in the debate.
        output_dir: Directory path where debate responses will be saved.
        json_mode: Whether to use JSON mode for responses.
        process_answer_func: Function to process answers from responses. Defaults to
            None, in which case extract_bool_answer will be used.

    Returns:
        List[List[dict]]: List of responses from each round, where each round's
            responses is a list of dictionaries containing agent responses.

    Raises:
        Exception: If any error occurs during the debate process.
            Original exception is logged and re-raised.
    """
    # If process_answer_func is None, use extract_bool_answer as default
    if process_answer_func is None:
        logger.error("No process_answer_func function provided")
        raise ValueError("process_answer_func function must be provided")

    logger.info(f"Starting debate with max_rounds={max_rounds}, json_mode={json_mode}")
    logger.info(f"Using agents ensemble: {agents_ensemble}")

    # Create a temporary directory for intermediate files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"debate_temp_{uuid.uuid4().hex}_", dir=output_dir)
    )
    logger.debug(f"Created temporary directory for debate: {temp_dir}")

    all_responses = []

    try:
        for i in range(max_rounds):
            logger.info(f"Starting debate round {i}")
            if i == 0:
                logger.info("Running round 0 (initial statements)")
                prompt = prompt_builder.build_round_zero()
                logger.debug(f"Round 0 prompt built: {prompt[:100]}...")
                round_responses = run_debate_round_zero(
                    prompt=prompt,
                    agents_ensemble=agents_ensemble,
                    output_dir=temp_dir,
                    json_mode=json_mode,
                )
            else:
                extracted_responses = [
                    response["response"] for response in all_responses[-1]
                ]
                logger.info(
                    f"Running debate round {i} with {len(extracted_responses)} previous responses"
                )
                logger.debug(
                    f"Extracted responses for round {i}: {extracted_responses}"
                )
                try:
                    if check_convergence(extracted_responses, process_answer_func):
                        logger.info(
                            f"Convergence detected after round {i-1}, ending debate early"
                        )
                        break
                except Exception as e:
                    logger.error(f"Error checking convergence: {str(e)}", exc_info=True)
                    raise

                prompt = prompt_builder.build_round_n(extracted_responses)
                logger.debug(f"Round {i} prompt built: {prompt[:100]}...")
                round_responses = run_debate_round_n(
                    prompt=prompt,
                    agents_ensemble=agents_ensemble,
                    output_dir=temp_dir,
                    round_num=i,
                    json_mode=json_mode,
                )
            all_responses.append(round_responses)
            logger.info(
                f"Completed debate round {i} with {len(round_responses)} agent responses"
            )

        # Debate completed successfully, move files from temp_dir to output_dir
        file_count = len(list(temp_dir.glob("*")))
        logger.info(
            f"Debate completed successfully after {len(all_responses)} rounds, saving {file_count} files"
        )

        for file_path in temp_dir.glob("*"):
            target_path = output_dir / file_path.name
            shutil.copy2(file_path, target_path)
            logger.debug(f"Saved debate artifact: {target_path}")

        return all_responses
    except Exception as e:
        logger.error(f"Error during debate: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up the temporary directory and its contents
        if temp_dir.exists():
            logger.debug(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


def run_debate_with_retry(
    max_rounds: int,
    run_debate_func: Callable,
    prompt: str,
    agents_ensemble: AgentsEnsemble,
    output_dir: Union[str, Path],
    round_num: int,
    json_mode: bool = False,
    process_answer_func: Optional[Callable] = None,
    max_retries: int = 3,
) -> List[Dict]:
    """Run multiple debate rounds with retry capabilities.

    Args:
        max_rounds (int): Maximum number of debate rounds to run.
        run_debate_func (Callable): Function to run each debate round.
        prompt (str): The debate prompt including previous context.
        agents_ensemble (AgentsEnsemble): Collection of LLM agents for the debate.
        output_dir (str | Path): Directory path for saving debate responses.
        json_mode (bool): Whether to expect JSON responses from agents.
        round_num (Optional[int]): Starting round number (defaults to 1 if None).
        process_answer_func (Optional[Callable]): Function to process responses
            between rounds.
        max_retries (int): Maximum retry attempts per round.

    Returns:
        List[Dict]: Combined list of agent responses from all rounds.

    Raises:
        RuntimeError: If maximum retries are exceeded for any round.
    """
    current_round = round_num if round_num is not None else 1
    all_responses = []
    current_prompt = prompt

    logger.info(
        f"Starting debate with retry. Max rounds: {max_rounds}, "
        f"Starting round: {current_round}, Max retries: {max_retries}"
    )

    while current_round <= max_rounds:
        retry_count = 0
        round_success = False

        while not round_success and retry_count < max_retries:
            try:
                logger.info(
                    f"Running round {current_round} "
                    f"(Attempt {retry_count + 1}/{max_retries})"
                )

                round_responses = run_debate_func(
                    prompt=current_prompt,
                    agents_ensemble=agents_ensemble,
                    output_dir=output_dir,
                    round_num=current_round,
                    json_mode=json_mode,
                )

                # Process responses if a processing function is provided
                if process_answer_func:
                    logger.debug(f"Processing responses for round {current_round}")
                    current_prompt = process_answer_func(
                        round_responses, current_prompt, current_round
                    )

                all_responses.extend(round_responses)
                round_success = True
                logger.info(f"Round {current_round} completed successfully")

            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"Error in round {current_round}, attempt "
                    f"{retry_count}: {str(e)}"
                )
                if retry_count >= max_retries:
                    logger.error(
                        f"Maximum retries ({max_retries}) exceeded for "
                        f"round {current_round}"
                    )
                    raise RuntimeError(
                        f"Failed to complete round {current_round} after "
                        f"{max_retries} attempts"
                    ) from e

                # Wait before retrying (with exponential backoff)
                wait_time = 2**retry_count  # 2, 4, 8 seconds...
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)

        current_round += 1

    logger.info(f"Debate completed successfully after {current_round - 1} rounds")
    return all_responses


def check_convergence(
    responses: List[Dict], process_answer_func: Optional[Callable] = None
) -> bool:
    """Check if the responses from all agents have converged to the same answer.

    Args:
        responses: List of agent responses from the most recent round of debate.
        process_answer_func: Function to process answers from responses. Defaults to
            None, in which case extract_bool_answer will be used.

    Returns:
        bool: True if all responses are the same, False otherwise.
    """
    # If process_answer_func is None, use extract_bool_answer as default
    if process_answer_func is None:
        logger.error("No process_answer_func function provided for convergence check")
        raise ValueError("process_answer_func function must be provided")

    try:
        answers = [process_answer_func(response) for response in responses]
        logger.debug(f"Processed answers for convergence check: {answers}")
        is_converged = len(set(answers)) == 1
        if is_converged:
            logger.info(f"Debate has converged on answer: {list(set(answers))[0]}")
        return is_converged
    except Exception as e:
        logger.error(f"Error checking convergence: {str(e)}", exc_info=False)
        raise


def main():
    from ..run.bool_q.prompts import (
        build_bool_q_round_n_prompt,
        build_bool_q_round_zero_prompt,
    )

    question = "Is the sky blue?"
    passage = "The sky is blue."
    prompt_builder = PromptBuilder(
        round_zero_fn=build_bool_q_round_zero_prompt,
        round_n_fn=build_bool_q_round_n_prompt,
        prompt_params={"question": question, "passage": passage},
    )
    agents_ensemble = AgentsEnsemble()
    output_dir = "data/test"
    debate(3, prompt_builder, agents_ensemble, output_dir)


if __name__ == "__main__":
    main()
