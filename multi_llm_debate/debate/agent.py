import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from ..llm.llm import call_model
from ..utils.logging_config import setup_logging

# Set up logger
logger = setup_logging(__name__)
logger.setLevel(logging.INFO)


class Agent:
    """A class representing an individual LLM agent.

    This class encapsulates a language model agent with specific model configuration
    and base URL for API calls.

    Attributes:
        agent_id (int): Unique identifier for the agent.
        model (str): Name of the language model being used.
        base_url (Optional[str]): Base URL for the API calls.
    """

    def __init__(
        self, agent_id: int, model: str, base_url: Optional[str] = None
    ) -> None:
        """Initialize an Agent instance.

        Args:
            agent_id (int): Unique identifier for the agent.
            model (str): Name of the language model.
            base_url (Optional[str]): Base URL for the OpenAI API calls.
                Default is None, which uses the OpenAI default.
        """
        self.agent_id = agent_id
        self.model = model
        self.base_url = base_url
        logger.debug(
            f"Initialized Agent {agent_id} with model {model} "
            f"(base_url: {'custom' if base_url else 'default'})"
        )

    def __str__(self) -> str:
        """Return a string representation of the agent.

        Returns:
            str: String representation of the agent.
        """
        return f"Agent {self.agent_id} ({self.model})"

    def __repr__(self) -> str:
        """Return a string representation of the agent.

        Returns:
            str: String representation of the agent.
        """
        return str(self)

    def respond(
        self,
        prompt: str,
        images: Union[str, Path, List[str], List[Path], None] = None,
        json_mode: bool = False,
        timeout: Optional[int] = None,
        max_retries: int = 0,
        api_key: Optional[str] = None,
        max_tokens: int = 6400,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate a response to the given prompt.

        Args:
            prompt (str): The input prompt to send to the language model.
            images (Union[str, Path, List[str], List[Path], None], optional):
                Image file paths for vision models. Can be a single path or a list.
            json_mode (bool, optional): Whether to expect JSON response.
                Defaults to False.
            timeout (Optional[int], optional): Maximum time to wait for response
                in seconds. Defaults to None, which uses the API's default timeout.
            max_retries (int, optional): Maximum number of retry attempts if the
                request fails. Defaults to 0 (no retries).
            api_key (Optional[str], optional): API key to use for this request.
                Defaults to None, which uses the one from config.
            max_tokens (int, optional): Maximum number of tokens in response.
                Defaults to 6400.
            temperature (float, optional): Controls randomness in the response.
                Defaults to 1.0. Lower values make responses more deterministic.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - agent_id: The ID of the responding agent
                - model: The model name
                - response: The model's response (can be dict or str)
                - error: If an error occurred, contains the error message (optional)

        Raises:
            ConnectionError: If there's a network or timeout issue.
            Exception: If some other error occurs during processing.
        """
        start_time = time.time()
        logger.debug(
            f"Agent {self.agent_id} ({self.model}) starting request "
            f"(timeout: {timeout}s, json_mode: {json_mode}, max_retries: {max_retries}, "
            f"temperature: {temperature})"
        )

        # Truncate prompt for logging
        prompt_preview = prompt[:100] + ("..." if len(prompt) > 100 else "")
        logger.debug(f"Agent {self.agent_id} prompt: {prompt_preview}")

        errors = []
        retry_delay = 1.0  # Default retry delay in seconds

        if images:
            # Convert single image input to list for consistency
            if not isinstance(images, list):
                images = [images]

            # Validate all images
            for img in images:
                if isinstance(img, (str, Path)):
                    img_path = Path(img)
                    if not img_path.exists():
                        raise ValueError(f"Image file {img_path} does not exist.")
                else:
                    raise ValueError(
                        f"Invalid image type: {type(img)}. Expected str or Path."
                    )
                    
        # Try up to max_retries + 1 times (original attempt + retries)
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry #{attempt} for agent {self.agent_id}")
                    # Exponential backoff for retry delay
                    current_delay = retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Waiting {current_delay:.2f}s before retry")
                    time.sleep(current_delay)

                # Make the actual API call
                api_start = time.time()
                logger.info(f"Agent {self.agent_id} ({self.model}) sending request")
                raw_response = call_model(
                    model_name=self.model,
                    base_url=self.base_url,
                    prompt=prompt,
                    images=images if images else None,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    api_key=api_key,
                    temperature=temperature,
                )
                api_time = time.time() - api_start
                logger.info(
                    f"Agent {self.agent_id} ({self.model}) "
                    f"received raw response in {api_time:.2f}s"
                )

                # If it's already a dictionary, use it directly
                if isinstance(raw_response, dict):
                    logger.debug(
                        f"Agent {self.agent_id} response was already a dictionary"
                    )
                    parsed_response = raw_response
                else:
                    # Try to parse as JSON, but keep as string if parsing fails
                    try:
                        logger.debug(
                            f"Agent {self.agent_id} attempting to parse JSON response"
                        )
                        parsed_response = json.loads(raw_response)
                        logger.debug(
                            f"Agent {self.agent_id} successfully parsed JSON response"
                        )
                    except (json.JSONDecodeError, TypeError):
                        logger.debug(
                            f"Agent {self.agent_id} response is not valid JSON, using as string"
                        )
                        parsed_response = str(raw_response)

                response = {
                    "agent_id": self.agent_id,
                    "model": self.model,
                    "response": parsed_response,
                }

                # Log response size and time
                if isinstance(parsed_response, str):
                    response_length = len(parsed_response)
                else:
                    response_length = len(json.dumps(parsed_response))

                total_time = time.time() - start_time
                logger.info(
                    f"Agent {self.agent_id} ({self.model}) completed in "
                    f"{total_time:.2f}s with {response_length} chars"
                )

                return response

            except ConnectionError as e:
                # Record the error and retry if we have retries left
                elapsed = time.time() - start_time
                error_msg = f"Connection error on attempt {attempt+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(
                    f"Agent {self.agent_id} ({self.model}) connection error "
                    f"after {elapsed:.2f}s: {str(e)}"
                )
                # If this was the last attempt, re-raise the exception
                if attempt == max_retries:
                    raise ConnectionError(
                        f"Failed after {max_retries+1} attempts: {'; '.join(errors)}"
                    )

            except Exception as e:
                # Record the error and retry if we have retries left
                elapsed = time.time() - start_time
                error_msg = f"Error on attempt {attempt+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(
                    f"Agent {self.agent_id} ({self.model}) unexpected error "
                    f"after {elapsed:.2f}s: {str(e)}",
                    exc_info=False,
                )
                # If this was the last attempt, re-raise the exception
                if attempt == max_retries:
                    raise Exception(
                        f"Failed after {max_retries+1} attempts: {'; '.join(errors)}"
                    )
