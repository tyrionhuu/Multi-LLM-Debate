import json
import logging
import time
from pathlib import Path
from typing import ByteString, List, Union

from .agents_ensemble import AgentsEnsemble

logger = logging.getLogger(__name__)


def run_debate_round_zero(
    prompt: str,
    agents_ensemble: AgentsEnsemble,
    output_dir: Union[str, Path],
    images: Union[
        str,
        Path,
        List[str],
        List[Path],
        bytes,
        List[bytes],
        ByteString,
        List[ByteString],
        None,
    ] = None,
    json_mode: bool = False,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    batch: bool = False,
    batch_size: int = 11,
) -> List[dict]:
    """Run the initial round (round zero) of a debate.

    Gets responses from all agents for the initial prompt and saves them to a JSON file.
    Logs the process and response metadata.

    Args:
        prompt: The initial prompt/question to start the debate.
        agents_ensemble: Collection of LLM agents participating in the debate.
        output_dir: Directory path where debate responses will be saved.
        images: Optional image data, which can be either file paths or raw image
            bytes. Supported formats include: string paths, Path objects, bytes,
            ByteString, or lists of any of these types.
        json_mode: Whether to expect JSON responses from agents.
        temperature: Sampling temperature for the model.
        max_tokens: Maximum number of tokens in the response.
        batch: Whether to run in batch mode.
        batch_size: Size of the batch.

    Returns:
        List[dict]: List of agent responses, where each response is a dictionary.

    Raises:
        OSError: If unable to create output directory or save results file.
        json.JSONDecodeError: If unable to serialize responses to JSON.
        Exception: If any error occurs when requesting agent responses.
    """
    logger.info(f"Starting debate round zero with {len(agents_ensemble.agents)} agents")
    logger.debug(f"Initial prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    start_time = time.time()

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.debug(f"Output directory created/verified: {output_dir}")

    try:
        logger.debug("Requesting responses from all agents...")
        response_start_time = time.time()
        responses = agents_ensemble.get_responses(
            prompt=prompt,
            images=images,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            batch=batch,
            batch_size=batch_size,
        )
        response_time = time.time() - response_start_time
        logger.info(f"All agent responses received in {response_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in round zero: {str(e)}", exc_info=False)
        logger.debug(
            f"Failed prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}"
        )
        raise

    for i, response in enumerate(responses):
        agent_id = response["agent_id"]
        response_text = response.get("response", "")
        logger.debug(f"Agent {agent_id} responded (#{i+1}/{len(responses)})")
        logger.debug(f"Agent {agent_id} response length: {len(response_text)} chars")

        # Log a preview of each response
        if response_text:
            preview = response_text[:100] + ("..." if len(response_text) > 100 else "")
            logger.debug(f"Agent {agent_id} response preview: {preview}")

    output_file = output_dir / "debate_round_0.json"
    try:
        with open(output_file, "w") as f:
            json.dump(responses, f, indent=2)
        logger.info(f"Round zero responses saved to {output_file}")
    except (OSError, json.JSONDecodeError) as e:
        logger.error(
            f"Failed to save responses to {output_file}: {str(e)}", exc_info=True
        )
        raise

    total_time = time.time() - start_time
    logger.info(f"Round zero completed in {total_time:.2f} seconds")
    return responses
