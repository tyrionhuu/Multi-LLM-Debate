from pathlib import Path
from typing import Dict, List

from ..llm.prompts import PromptBuilder
from ..utils.logging_config import setup_logging
from .agents_ensemble import AgentsEnsemble
from .round_n import run_debate_round_n
from .round_zero import run_debate_round_zero

logger = setup_logging(__name__)


def run_debate(
    max_rounds: int,
    prompt_builder: PromptBuilder,
    agents_ensemble: AgentsEnsemble,
    output_dir: str | Path,
) -> List[List[dict]]:
    """Run a full debate with multiple rounds using the given prompts and agents.

    Coordinates multiple rounds of debate between agents, starting with round zero
    and continuing through subsequent rounds. Logs progress and saves results.

    Args:
        max_rounds: Maximum number of debate rounds to run.
        prompt_builder: PromptBuilder instance to generate prompts for each round.
        agents_ensemble: Collection of LLM agents participating in the debate.
        output_dir: Directory path where debate responses will be saved.

    Returns:
        List[List[dict]]: List of responses from each round, where each round's
            responses is a list of dictionaries containing agent responses.

    Raises:
        Exception: If any error occurs during the debate process.
            Original exception is logged and re-raised.
    """
    print(f"Starting debate with {len(agents_ensemble)} agents")
    print(f"Maximum rounds: {max_rounds}")
    print(f"Output directory: {output_dir}")

    all_responses = []

    try:
        for i in range(max_rounds):
            print(f"Starting debate round {i}")
            if i == 0:
                prompt = prompt_builder.build_round_zero()
                round_responses = run_debate_round_zero(
                    prompt, agents_ensemble, output_dir
                )
            else:
                extracted_responses = [
                    response["response"] for response in all_responses[-1]
                ]
                if check_convergence(extracted_responses):
                    print("Convergence reached, ending debate")
                    break
                prompt = prompt_builder.build_round_n(extracted_responses)
                round_responses = run_debate_round_n(
                    prompt, agents_ensemble, output_dir, i
                )
            all_responses.append(round_responses)
            print(f"Completed debate round {i}")

        print("Debate completed successfully")
        return all_responses
    except Exception as e:
        logger.error(f"Error during debate: {str(e)}", exc_info=True)
        raise


def check_convergence(responses: List[Dict]) -> bool:
    """Check if the responses from all agents have converged to the same answer.

    Args:
        responses: List of agent responses from the most recent round of debate.

    Returns:
        bool: True if all responses are the same, False otherwise.
    """
    return len(set(response["answer"] for response in responses)) == 1


def main():
    from ..llm.prompts import (
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
    run_debate(3, prompt_builder, agents_ensemble, output_dir)


if __name__ == "__main__":
    main()
