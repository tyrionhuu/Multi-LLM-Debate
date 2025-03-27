import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional
import logging
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
    process_answer: Optional[Callable] = None,
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
        process_answer: Function to process answers from responses. Defaults to
            None, in which case extract_bool_answer will be used.

    Returns:
        List[List[dict]]: List of responses from each round, where each round's
            responses is a list of dictionaries containing agent responses.

    Raises:
        Exception: If any error occurs during the debate process.
            Original exception is logged and re-raised.
    """
    # If process_answer is None, use extract_bool_answer as default
    if process_answer is None:
        raise ValueError("process_answer function must be provided")

    # Create a temporary directory for intermediate files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"debate_temp_{uuid.uuid4().hex}_", dir=output_dir)
    )

    all_responses = []

    try:
        for i in range(max_rounds):
            if i == 0:
                # print("Running round 0")
                prompt = prompt_builder.build_round_zero()
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
                logger.debug(
                    f"Extracted responses for round {i}: {extracted_responses}"
                )
                try:
                    if check_convergence(extracted_responses, process_answer):
                        # print("Convergence detected, ending debate")
                        break
                except Exception as e:
                    logger.error(f"Error checking convergence: {str(e)}", exc_info=True)
                    raise
                # print(f"Running debate round {i}")
                prompt = prompt_builder.build_round_n(extracted_responses)
                round_responses = run_debate_round_n(
                    prompt=prompt,
                    agents_ensemble=agents_ensemble,
                    output_dir=temp_dir,
                    round_num=i,
                    json_mode=json_mode,
                )
            all_responses.append(round_responses)
            # print(f"Completed debate round {i}")

        # Debate completed successfully, move files from temp_dir to output_dir
        for file_path in temp_dir.glob("*"):
            target_path = output_dir / file_path.name
            shutil.copy2(file_path, target_path)

        # print("Debate completed successfully")
        return all_responses
    except Exception as e:
        logger.error(f"Error during debate: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up the temporary directory and its contents
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def check_convergence(
    responses: List[Dict], process_answer: Optional[Callable] = None
) -> bool:
    """Check if the responses from all agents have converged to the same answer.

    Args:
        responses: List of agent responses from the most recent round of debate.
        process_answer: Function to process answers from responses. Defaults to
            None, in which case extract_bool_answer will be used.

    Returns:
        bool: True if all responses are the same, False otherwise.
    """
    # If process_answer is None, use extract_bool_answer as default
    if process_answer is None:
        raise ValueError("process_answer function must be provided")

    try:
        answers = [process_answer(response) for response in responses]
        return len(set(answers)) == 1
    except Exception as e:
        logger.error(f"Error checking convergence: {str(e)}", exc_info=True)
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
