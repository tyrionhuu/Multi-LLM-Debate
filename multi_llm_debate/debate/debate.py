import logging
import shutil
import tempfile
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
    extract_func: Callable,
    output_dir: Union[str, Path],
    json_mode: bool = False,
    max_retries: int = 3,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    parallel: bool = False,
    diversity_pruning_func: Callable = None,
    pruning_amount: int = 5,
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
        extract_func: Function to process answers from responses.
        max_retries: Maximum retry attempts for each round. Defaults to 3.
        temperature: Sampling temperature for the model. Defaults to 1.0.
        max_tokens: Maximum number of tokens in the response. Defaults to 6400.
        parallel: Whether to run agents in parallel.
        diversity_pruning_func: Optional function for diversity pruning.
        pruning_amount: Amount of selected responses for diversity pruning.

    Returns:
        List[List[dict]]: List of responses from each round, where each round's
            responses is a list of dictionaries containing agent responses.

    Raises:
        Exception: If any error occurs during the debate process.
            Original exception is logged and re-raised.
    """
    # If extract_func is None, use extract_bool_answer as default
    if extract_func is None:
        logger.error("No extract_func function provided")
        raise ValueError("extract_func function must be provided")

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
                round_responses = run_debate_with_retry(
                    max_rounds=max_rounds,
                    prompt=prompt,
                    agents_ensemble=agents_ensemble,
                    output_dir=temp_dir,
                    round_num=i,
                    extract_func=extract_func,
                    json_mode=json_mode,
                    max_retries=max_retries,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    parallel=parallel,
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
                    if check_convergence(extracted_responses, extract_func):
                        logger.info(
                            f"Convergence detected after round {i-1}, ending debate early"
                        )
                        break
                except Exception as e:
                    logger.error(f"Error checking convergence: {str(e)}", exc_info=True)
                    raise
                
                # Apply diversity pruning if specified
                if diversity_pruning_func:
                    logger.info(
                        f"Applying diversity pruning for round {i} with amount={pruning_amount}"
                    )
                    pruned_responses = diversity_pruning_func(
                        extracted_responses,
                        selected_amount=pruning_amount,
                        extract_func=extract_func,
                    )
                
                prompt = prompt_builder.build_round_n(pruned_responses)
                logger.debug(f"Round {i} prompt built: {prompt[:100]}...")
                round_responses = run_debate_with_retry(
                    max_rounds=max_rounds,
                    prompt=prompt,
                    agents_ensemble=agents_ensemble,
                    output_dir=temp_dir,
                    round_num=i,
                    extract_func=extract_func,
                    json_mode=json_mode,
                    max_retries=max_retries,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    parallel=parallel,
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
    prompt: str,
    agents_ensemble: AgentsEnsemble,
    output_dir: Union[str, Path],
    round_num: int,
    extract_func: Callable,
    json_mode: bool = False,
    max_retries: int = 3,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    parallel: bool = False,
) -> List[Dict]:
    """Run a debate round with retry capabilities.

    If extract_func raises an error, the function will retry the debate
    round up to max_retries times.

    Args:
        max_rounds: Maximum number of debate rounds.
        prompt: The debate prompt including previous context.
        agents_ensemble: Collection of LLM agents for the debate.
        output_dir: Directory path for saving debate responses.
        round_num: The current round number.
        extract_func: Function to process responses between rounds.
        json_mode: Whether to expect JSON responses from agents.
        max_retries: Maximum retry attempts for the round.
        temperature: Sampling temperature for the model.
        max_tokens: Maximum number of tokens in the response.
        parallel: Whether to run agents in parallel.

    Returns:
        List[Dict]: List of agent responses from the round.

    Raises:
        RuntimeError: If maximum retries are exceeded.
        ValueError: If round_num is invalid.
    """
    if round_num < 0 or round_num >= max_rounds:
        logger.error(
            f"Invalid round number: {round_num}. Must be between 0 and {max_rounds - 1}."
        )
        raise ValueError(f"Round number must be between 0 and {max_rounds - 1}.")
    if max_retries < 1:
        logger.error("max_retries must be at least 1")
        raise ValueError("max_retries must be at least 1")

    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    if not output_dir.exists():
        logger.debug(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.debug(f"Output directory already exists: {output_dir}")

    if not output_dir.is_dir():
        logger.error(f"Output path {output_dir} is not a directory.")
        raise ValueError(f"Output path {output_dir} must be a directory.")

    if extract_func is None:
        logger.error("No extract_func function provided for debate round")
        raise ValueError("extract_func function must be provided")

    # Log the start of the debate round with retries
    logger.info(f"Starting debate round {round_num} with max_retries={max_retries}")

    for attempt in range(1, max_retries + 1):
        try:
            # Run the appropriate debate function based on round number
            if round_num == 0:
                responses = run_debate_round_zero(
                    prompt=prompt,
                    agents_ensemble=agents_ensemble,
                    output_dir=output_dir,
                    json_mode=json_mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    parallel=parallel,
                )
            else:
                responses = run_debate_round_n(
                    prompt=prompt,
                    agents_ensemble=agents_ensemble,
                    output_dir=output_dir,
                    round_num=round_num,
                    json_mode=json_mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    parallel=parallel,
                )

            try:
                for response in responses:
                    extract_func(response["response"])
            except Exception as e:
                logger.warning(f"Error processing response with extract_func: {str(e)}")
                raise  # Re-raise to trigger retry

            logger.info(
                f"Debate round {round_num} completed successfully on attempt {attempt}"
            )
            return responses

        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"Error in debate round {round_num}, attempt {attempt}/{max_retries}: {str(e)}. "
                )
            else:
                logger.error(
                    f"Maximum retries ({max_retries}) exceeded for debate round {round_num}: {str(e)}"
                )
                raise RuntimeError(
                    f"Failed to complete debate round {round_num} after {max_retries} attempts"
                ) from e


def check_convergence(
    responses: List[Dict], extract_func: Optional[Callable] = None
) -> bool:
    """Check if the responses from all agents have converged to the same answer.

    Args:
        responses: List of agent responses from the most recent round of debate.
        extract_func: Function to process answers from responses. Defaults to
            None, in which case extract_bool_answer will be used.

    Returns:
        bool: True if all responses are the same, False otherwise.
    """
    # If extract_func is None, use extract_bool_answer as default
    if extract_func is None:
        logger.error("No extract_func function provided for convergence check")
        raise ValueError("extract_func function must be provided")

    try:
        answers = [extract_func(response) for response in responses]
        logger.debug(f"Processed answers for convergence check: {answers}")
        is_converged = len(set(answers)) == 1
        if is_converged:
            logger.info(f"Debate has converged on answer: {list(set(answers))[0]}")
        return is_converged
    except Exception as e:
        logger.error(f"Error checking convergence: {str(e)}", exc_info=False)
        raise


def main():
    """Test the debate functionality with a simple example.

    This function demonstrates how to run a debate with retry capabilities
    using a simple boolean question.
    """
    import time

    from ..run.bool_q.prompts import (
        build_bool_q_round_n_prompt,
        build_bool_q_round_zero_prompt,
    )

    # Define a simple question and passage
    question = "Is the sky blue?"
    passage = "The sky appears blue to the human eye during the day because of Rayleigh scattering."

    # Create a prompt builder with the question and passage
    prompt_builder = PromptBuilder(
        round_zero_fn=build_bool_q_round_zero_prompt,
        round_n_fn=build_bool_q_round_n_prompt,
        prompt_params={"question": question, "passage": passage},
    )

    # Create an agents ensemble for the debate
    agents_ensemble = AgentsEnsemble()

    # Define the output directory
    output_dir = Path("data/test_with_retry")

    # Define a custom extract_func that sometimes fails
    # to demonstrate retry capability
    def test_extract_func(response: str) -> bool:
        """Process the response to extract a boolean answer.

        This function randomly fails occasionally to test the retry mechanism.

        Args:
            response: The text response from an agent.

        Returns:
            bool: True if the answer is 'yes', False if 'no'.

        Raises:
            ValueError: If the response cannot be processed or if
                random failure is triggered.
        """
        # Randomly fail sometimes to test retry
        if time.time() % 10 < 3:  # Will fail ~30% of the time
            logger.warning("Simulated random failure in extract_func")
            raise ValueError("Simulated random failure to test retry mechanism")

        # Extract the answer from the response
        response = response.lower()
        if "yes" in response:
            return True
        elif "no" in response:
            return False
        else:
            raise ValueError(f"Could not extract boolean answer from: {response}")

    # Log test parameters
    logger.info("=== Starting Debate Test with Retry Capability ===")
    logger.info(f"Question: {question}")
    logger.info(f"Passage: {passage}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Run the debate with 2 rounds, 3 retries max
        results = debate(
            max_rounds=2,
            prompt_builder=prompt_builder,
            agents_ensemble=agents_ensemble,
            output_dir=output_dir,
            extract_func=test_extract_func,
            max_retries=3,
            json_mode=False,
        )

        # Print results summary
        logger.info("=== Debate Completed Successfully ===")
        logger.info(f"Total rounds completed: {len(results)}")
        for i, round_results in enumerate(results):
            logger.info(f"Round {i} had {len(round_results)} responses")

    except Exception as e:
        logger.error(f"Debate test failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
