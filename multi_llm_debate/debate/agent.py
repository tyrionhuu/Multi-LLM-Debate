import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..llm.llm import call_model, call_model_batch

# Set up logger
logger = logging.getLogger(__name__)


class Agent:
    """A class representing an individual LLM agent.

    DEPRECATED: This class is deprecated and will be removed in a future version.
    Use the AgentsEnsemble class directly instead, which now includes all agent
    functionality.

    This class encapsulates a language model agent with specific model configuration
    and base URL for API calls.

    Attributes:
        agent_id (int): Unique identifier for the agent.
        model (str): Name of the language model being used.
        base_url (Optional[str]): Base URL for the API calls.
        api_key (Optional[str]): API key for the agent.
    """

    def __init__(
        self,
        agent_id: int,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize an Agent instance.

        Args:
            agent_id (int): Unique identifier for the agent.
            model (str): Name of the language model.
            base_url (Optional[str]): Base URL for the OpenAI API calls.
                Default is None, which uses the OpenAI default.
            api_key (Optional[str]): API key for the agent. Default is None.
        """
        warnings.warn(
            "The Agent class is deprecated. Use AgentsEnsemble directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.agent_id = agent_id
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        logger.debug(
            f"Initialized Agent {agent_id} with model {model} "
            f"(base_url: {'custom' if base_url else 'default'}, "
            f"api_key: {'set' if api_key else 'not set'})"
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
                    api_key=self.api_key,
                    prompt=prompt,
                    images=images,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    timeout=timeout,
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

    def respond_batch(
        self,
        prompts: List[str],
        images: Optional[List[Union[str, Path, List[str], List[Path], None]]] = None,
        json_mode: bool = False,
        timeout: Optional[int] = None,
        max_retries: int = 0,
        max_tokens: int = 6400,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Generate responses to multiple prompts in batch mode.

        Args:
            prompts (List[str]): List of input prompts to send to the language model.
            images (Optional[List[Union[str, Path, List[str], List[Path], None]]]):
                Optional list of images for each prompt. Must match length of prompts
                or be None. Each element can be a single image path or a list of paths.
            json_mode (bool, optional): Whether to expect JSON responses.
                Defaults to False.
            timeout (Optional[int], optional): Maximum time to wait for response
                in seconds. Defaults to None, which uses the API's default timeout.
            max_retries (int, optional): Maximum number of retry attempts if the
                request fails. Defaults to 0 (no retries).
            max_tokens (int, optional): Maximum number of tokens in response.
                Defaults to 6400.
            temperature (float, optional): Controls randomness in the response.
                Defaults to 1.0. Lower values make responses more deterministic.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing for each prompt:
                - agent_id: The ID of the responding agent
                - model: The model name
                - response: The model's response (can be dict or str)
                - error: If an error occurred, contains the error message (optional)

        Raises:
            ValueError: If prompts is empty.
            ConnectionError: If there's a network or timeout issue after all retries.
            Exception: If some other error occurs after all retries.
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")

        start_time = time.time()
        prompt_count = len(prompts)
        logger.debug(
            f"Agent {self.agent_id} ({self.model}) starting batch request with "
            f"{prompt_count} prompts (timeout: {timeout}s, json_mode: {json_mode}, "
            f"max_retries: {max_retries}, temperature: {temperature})"
        )

        # Validate image inputs if provided
        if images is not None:
            if len(images) != len(prompts):
                raise ValueError("Length of images must match length of prompts")

            # Validate all images
            for img_set in images:
                if img_set is None:
                    continue

                # Convert single image to list for consistency
                if not isinstance(img_set, list):
                    img_set = [img_set]

                for img in img_set:
                    if isinstance(img, (str, Path)):
                        img_path = Path(img)
                        if not img_path.exists():
                            raise ValueError(f"Image file {img_path} does not exist.")
                    else:
                        raise ValueError(
                            f"Invalid image type: {type(img)}. Expected str or Path."
                        )

        errors = []
        retry_delay = 1.0  # Default retry delay in seconds

        # Try up to max_retries + 1 times (original attempt + retries)
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retry #{attempt} for agent {self.agent_id} batch request"
                    )
                    # Exponential backoff for retry delay
                    current_delay = retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Waiting {current_delay:.2f}s before retry")
                    time.sleep(current_delay)

                # Make the actual API call
                api_start = time.time()
                logger.info(
                    f"Agent {self.agent_id} ({self.model}) sending batch request "
                    f"with {prompt_count} prompts"
                )
                raw_responses = call_model_batch(
                    model_name=self.model,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    prompts=prompts,
                    images=images,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    temperature=temperature,
                )
                api_time = time.time() - api_start
                logger.info(
                    f"Agent {self.agent_id} ({self.model}) "
                    f"received {len(raw_responses)} raw responses in {api_time:.2f}s"
                )

                # Process each response
                processed_responses = []
                for raw_response in raw_responses:
                    # If it's already a dictionary, use it directly
                    if isinstance(raw_response, dict):
                        parsed_response = raw_response
                    else:
                        # Try to parse as JSON, but keep as string if parsing fails
                        try:
                            parsed_response = json.loads(raw_response)
                        except (json.JSONDecodeError, TypeError):
                            parsed_response = str(raw_response)

                    processed_responses.append(
                        {
                            "agent_id": self.agent_id,
                            "model": self.model,
                            "response": parsed_response,
                        }
                    )

                total_time = time.time() - start_time
                logger.info(
                    f"Agent {self.agent_id} ({self.model}) batch completed in "
                    f"{total_time:.2f}s with {len(processed_responses)} responses"
                )

                return processed_responses

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
                    exc_info=True,
                )
                # If this was the last attempt, re-raise the exception
                if attempt == max_retries:
                    raise Exception(
                        f"Failed after {max_retries+1} attempts: {'; '.join(errors)}"
                    )
