import concurrent.futures
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from ..llm.llm import call_model, call_model_batch
from ..utils.config_manager import get_models
from ..utils.model_config import ModelConfig

# Use setup_logging to ensure consistent logging
logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Simple data class to store agent information.

    Attributes:
        agent_id (int): Unique identifier for the agent.
        model (str): Name of the language model being used.
        base_url (Optional[str]): Base URL for the API calls.
        api_key (Optional[str]): API key for the agent.
    """

    agent_id: int
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"Agent {self.agent_id} ({self.model})"


class AgentsEnsemble:
    """A collection of LLM agents that can be used together.

    This class manages multiple agents and provides methods to interact with them
    collectively. It can be initialized automatically from configuration or built manually.

    Agent functionality is directly integrated into this class instead of using
    a separate Agent class, simplifying the architecture.

    Attributes:
        agents (List[AgentInfo]): List of agent information in the ensemble.
        job_delay (float): Delay in seconds between consecutive agent calls.
        timeout (float): Maximum time in seconds to wait for agent responses.
        max_retries (int): Maximum number of retry attempts for failed requests.
    """

    def __init__(
        self,
        config_list: Optional[List[ModelConfig]] = None,
        job_delay: float = 0.5,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize an AgentsEnsemble instance.

        Args:
            config_list (Optional[List[ModelConfig]]): List of model configurations.
            job_delay (float, optional): Delay in seconds between agent calls.
            timeout (float, optional): Maximum time in seconds to wait for agent responses.
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        """
        self.job_delay = job_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.agents: List[AgentInfo] = []  # List to hold agent information

        if config_list is not None:
            self._initialize_from_config_list(config_list)
        else:
            self._initialize_from_config()

    def _initialize_from_config(self) -> None:
        """Initialize agents from configuration.

        Loads model configurations and creates agents accordingly.
        Each agent is assigned a unique ID starting from 0.
        """
        models = get_models()
        agent_id = 0
        for model_info in models:
            base_url = None
            api_key = None

            # Handle different formats returned by get_models()
            if isinstance(model_info, tuple) and len(model_info) >= 3:
                # Old format: (model_name, base_url, quantity)
                model_name, base_url, quantity = model_info[:3]
            elif isinstance(model_info, dict):
                # New format: {"name": model_name, "base_url": url, "quantity": qty, "api_key": ...}
                model_name = model_info.get("name", "")
                base_url = model_info.get("base_url")
                quantity = model_info.get("quantity", 1)
                api_key = model_info.get("api_key")
            else:
                logger.warning(f"Skipping unrecognized model info format: {model_info}")
                continue

            for _ in range(quantity):
                agent_info = AgentInfo(
                    agent_id=agent_id,
                    model=model_name,
                    base_url=base_url,
                    api_key=api_key,
                )
                self.add_agent(agent_info)
                agent_id += 1

    def _initialize_from_config_list(self, config_list: List[ModelConfig]) -> None:
        """Initialize agents from a list of model configurations.

        Args:
            config_list (List[ModelConfig]): List of model configurations.

        Raises:
            ValueError: If the configuration list is empty.
        """
        if not config_list:
            raise ValueError("Config list cannot be empty")

        agent_id = 0
        for config in config_list:
            base_url = config.get("base_url")
            quantity = config.get("quantity", 1)
            model_name = config["name"]
            api_key = config.get("api_key")

            for _ in range(quantity):
                agent_info = AgentInfo(
                    agent_id=agent_id,
                    model=model_name,
                    base_url=base_url,
                    api_key=api_key,
                )
                self.add_agent(agent_info)
                agent_id += 1

    def add_agent(self, agent_info: AgentInfo) -> None:
        """Add an agent to the ensemble.

        Args:
            agent_info (AgentInfo): The agent to add to the ensemble.
        """
        self.agents.append(agent_info)
        logger.debug(
            f"Added Agent {agent_info.agent_id} with model {agent_info.model} "
            f"(base_url: {'custom' if agent_info.base_url else 'default'}, "
            f"api_key: {'set' if agent_info.api_key else 'not set'})"
        )

    def _count_unique_models(self) -> int:
        """Return the number of unique models among agents."""
        return len({agent.model for agent in self.agents})

    def _group_agents_by_model(self) -> Dict[str, List[AgentInfo]]:
        """Group agents by their model name.

        Returns:
            Dict[str, List[AgentInfo]]: Mapping from model name to list of agents.
        """
        model_groups: Dict[str, List[AgentInfo]] = {}
        for agent in self.agents:
            model_groups.setdefault(agent.model, []).append(agent)
        return model_groups

    def _retry_with_backoff(
        self,
        func,
        *args,
        retries: int,
        base_delay: float = 1.0,
        max_delay: float = 16.0,
        **kwargs,
    ):
        """Generic retry logic with exponential backoff and jitter.

        Args:
            func: Function to call.
            *args: Positional arguments for func.
            retries (int): Number of retries.
            base_delay (float): Initial delay.
            max_delay (float): Maximum delay.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of func(*args, **kwargs).

        Raises:
            Exception: If all retries fail.
        """
        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except (ConnectionError, Exception) as e:
                if attempt >= retries:
                    logger.error(f"All retries failed for {func.__name__}")
                    raise
                delay = min(max_delay, base_delay * (2**attempt))
                jitter = random.uniform(0, delay / 2)
                total_delay = delay + jitter
                logger.warning(
                    f"Retry {attempt+1}/{retries} for {func.__name__} after {total_delay:.2f}s due to error: {e}"
                )
                time.sleep(total_delay)
                attempt += 1

    def _process_response(
        self, agent_info: AgentInfo, raw_response: Any
    ) -> Dict[str, Any]:
        """Process the raw response from the LLM API.

        Args:
            agent_info (AgentInfo): Information about the agent.
            raw_response: The raw response from the API.

        Returns:
            Dict[str, Any]: Processed response with agent information.
        """
        # If it's already a dictionary, use it directly
        if isinstance(raw_response, dict):
            logger.debug(
                f"Agent {agent_info.agent_id} response was already a dictionary"
            )
            parsed_response = raw_response
        else:
            # Try to parse as JSON, but keep as string if parsing fails
            try:
                logger.debug(
                    f"Agent {agent_info.agent_id} attempting to parse JSON response"
                )
                parsed_response = json.loads(raw_response)
                logger.debug(
                    f"Agent {agent_info.agent_id} successfully parsed JSON response"
                )
            except (json.JSONDecodeError, TypeError):
                logger.debug(
                    f"Agent {agent_info.agent_id} response is not valid JSON, using as string"
                )
                parsed_response = str(raw_response)

        response = {
            "agent_id": agent_info.agent_id,
            "model": agent_info.model,
            "response": parsed_response,
        }

        return response

    def _respond(
        self,
        agent_info: AgentInfo,
        prompt: str,
        images: Union[str, Path, List[str], List[Path], None] = None,
        json_mode: bool = False,
        timeout: Optional[int] = None,
        max_retries: int = 0,
        max_tokens: int = 6400,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate a response to the given prompt for a specific agent.

        Args:
            agent_info (AgentInfo): The agent information.
            prompt (str): The input prompt to send to the language model.
            images: Image file paths for vision models.
            json_mode (bool): Whether to expect JSON response.
            timeout (Optional[int]): Maximum time to wait for response.
            max_retries (int): Maximum number of retry attempts.
            max_tokens (int): Maximum number of tokens in response.
            temperature (float): Controls randomness in the response.

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
            f"Agent {agent_info.agent_id} ({agent_info.model}) starting request "
            f"(timeout: {timeout}s, json_mode: {json_mode}, "
            f"max_retries: {max_retries}, temperature: {temperature})"
        )

        # Truncate prompt for logging
        prompt_preview = prompt[:100] + ("..." if len(prompt) > 100 else "")
        logger.debug(f"Agent {agent_info.agent_id} prompt: {prompt_preview}")

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
                    logger.info(f"Retry #{attempt} for agent {agent_info.agent_id}")
                    # Exponential backoff for retry delay
                    current_delay = retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Waiting {current_delay:.2f}s before retry")
                    time.sleep(current_delay)

                # Make the actual API call
                api_start = time.time()
                logger.info(
                    f"Agent {agent_info.agent_id} ({agent_info.model}) sending request"
                )
                raw_response = call_model(
                    model_name=agent_info.model,
                    base_url=agent_info.base_url,
                    api_key=agent_info.api_key,
                    prompt=prompt,
                    images=images,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    temperature=temperature,
                )
                api_time = time.time() - api_start
                logger.info(
                    f"Agent {agent_info.agent_id} ({agent_info.model}) "
                    f"received raw response in {api_time:.2f}s"
                )

                response = self._process_response(agent_info, raw_response)

                # Log response size and time
                if isinstance(response["response"], str):
                    response_length = len(response["response"])
                else:
                    response_length = len(json.dumps(response["response"]))

                total_time = time.time() - start_time
                logger.info(
                    f"Agent {agent_info.agent_id} ({agent_info.model}) completed in "
                    f"{total_time:.2f}s with {response_length} chars"
                )

                return response

            except ConnectionError as e:
                # Record the error and retry if we have retries left
                elapsed = time.time() - start_time
                error_msg = f"Connection error on attempt {attempt+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(
                    f"Agent {agent_info.agent_id} ({agent_info.model}) connection error "
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
                    f"Agent {agent_info.agent_id} ({agent_info.model}) unexpected error "
                    f"after {elapsed:.2f}s: {str(e)}",
                    exc_info=True,
                )
                # If this was the last attempt, re-raise the exception
                if attempt == max_retries:
                    raise Exception(
                        f"Failed after {max_retries+1} attempts: {'; '.join(errors)}"
                    )

    def _respond_batch(
        self,
        agent_info: AgentInfo,
        prompt: str,
        images: Union[str, Path, List[str], List[Path], None] = None,
        json_mode: bool = False,
        timeout: Optional[int] = None,
        max_retries: int = 0,
        max_tokens: int = 6400,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Generate batch responses to a single prompt for a specific agent.
        
        This method creates a list of identical prompts internally and uses
        call_model_batch for efficient processing.

        Args:
            agent_info (AgentInfo): The agent information.
            prompt (str): Input prompt to send.
            images: Image file path(s) for vision models.
            json_mode (bool): Whether to expect JSON responses.
            timeout (Optional[int]): Maximum time to wait for response.
            max_retries (int): Maximum number of retry attempts.
            max_tokens (int): Maximum number of tokens in responses.
            temperature (float): Controls randomness in the responses.

        Returns:
            List[Dict[str, Any]]: List of response dictionaries.

        Raises:
            ConnectionError: If there's a network or timeout issue after all retries.
            Exception: If some other error occurs after all retries.
        """
        start_time = time.time()
        batch_size = len(self.agents)  # Create a batch with identical prompts
        prompts = [prompt] * batch_size
        
        logger.debug(
            f"Agent {agent_info.agent_id} ({agent_info.model}) starting batch request with "
            f"{batch_size} identical prompts (timeout: {timeout}s, json_mode: {json_mode}, "
            f"max_retries: {max_retries}, temperature: {temperature})"
        )

        # Process images to match the batch size
        batch_images = None
        if images is not None:
            # Convert single image to list for consistency
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
            
            # Create a list of identical image sets for each prompt
            batch_images = [images] * batch_size

        errors = []
        retry_delay = 1.0  # Default retry delay in seconds

        # Try up to max_retries + 1 times (original attempt + retries)
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retry #{attempt} for agent {agent_info.agent_id} batch request"
                    )
                    # Exponential backoff for retry delay
                    current_delay = retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Waiting {current_delay:.2f}s before retry")
                    time.sleep(current_delay)

                # Make the actual API call
                api_start = time.time()
                logger.info(
                    f"Agent {agent_info.agent_id} ({agent_info.model}) sending batch request "
                    f"with {batch_size} identical prompts"
                )
                raw_responses = call_model_batch(
                    model_name=agent_info.model,
                    base_url=agent_info.base_url,
                    api_key=agent_info.api_key,
                    prompts=prompts,
                    images=batch_images,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    temperature=temperature,
                )
                api_time = time.time() - api_start
                logger.info(
                    f"Agent {agent_info.agent_id} ({agent_info.model}) "
                    f"received {len(raw_responses)} raw responses in {api_time:.2f}s"
                )

                # Process each response
                processed_responses = []
                for raw_response in raw_responses:
                    processed_response = self._process_response(
                        agent_info, raw_response
                    )
                    processed_responses.append(processed_response)

                total_time = time.time() - start_time
                logger.info(
                    f"Agent {agent_info.agent_id} ({agent_info.model}) batch completed in "
                    f"{total_time:.2f}s with {len(processed_responses)} responses"
                )

                return processed_responses

            except ConnectionError as e:
                # Record the error and retry if we have retries left
                elapsed = time.time() - start_time
                error_msg = f"Connection error on attempt {attempt+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(
                    f"Agent {agent_info.agent_id} ({agent_info.model}) connection error "
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
                    f"Agent {agent_info.agent_id} ({agent_info.model}) unexpected error "
                    f"after {elapsed:.2f}s: {str(e)}",
                    exc_info=True,
                )
                # If this was the last attempt, re-raise the exception
                if attempt == max_retries:
                    raise Exception(
                        f"Failed after {max_retries+1} attempts: {'; '.join(errors)}"
                    )

    def get_responses(
        self,
        prompt: str,
        images: Union[str, Path, List[str], List[Path], None] = None,
        json_mode: bool = False,
        max_retries: Optional[int] = None,
        max_tokens: int = 6400,
        temperature: float = 1.0,
        parallel: bool = False,
        batch: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get responses from all agents.

        Args:
            prompt (str): Prompt to send. If batch=True,
                this must be a list of prompts.
            images: Image file paths for vision models. If batch=True, this
                should be a list matching the prompts list length, where each
                element corresponds to a prompt's images.
            json_mode (bool): Expect JSON responses.
            max_retries (Optional[int]): Max retries.
            max_tokens (int): Max tokens.
            temperature (float): Response randomness.
            parallel (bool): Whether to process in parallel. Automatically
                disabled if only one model type is present as it adds
                unnecessary overhead.
            batch (bool): If True, process the input as a batch of prompts.
                'prompt' must be a list of strings. Returns a flat list of
                all responses from all prompts. Defaults to False.

        Returns:
            List[Dict[str, Any]]: Agent responses. If batch=True, this is a
                flat list containing responses for all prompts from all agents.

        Raises:
            ValueError: If batch=True and prompt is not a list, or if images
                are provided in batch mode but don't match prompt length.
        """
        if batch:
            # Validate images for batch mode
            batch_images = None
            if images is not None:
                if not isinstance(images, list) or len(images) != len(prompt):
                    raise ValueError(
                        "In batch mode, 'images' must be a list matching the "
                        "length of 'prompts'."
                    )
                batch_images = images  # Use the provided list directly

            logger.info(f"Getting responses in batch mode for {len(prompt)} prompts")
            # Use the batch processing method
            batch_responses_nested = self.get_responses_batch(
                prompt=prompt,
                images=batch_images,
                json_mode=json_mode,
                max_retries=max_retries,
                max_tokens=max_tokens,
                temperature=temperature,
                parallel=parallel,
            )

            # Flatten the results: List[List[Dict]] -> List[Dict]
            flattened_responses = [
                response
                for prompt_responses in batch_responses_nested
                for response in prompt_responses
            ]
            logger.info(
                f"Returning {len(flattened_responses)} total responses from batch mode"
            )
            return flattened_responses

        if not isinstance(prompt, str):
            raise ValueError("When batch=False, prompt must be a single string")

        responses = []

        unique_models = set(agent.model for agent in self.agents)
        use_parallel = parallel and len(unique_models) > 1

        if use_parallel:
            logger.info("Getting responses in parallel mode")
            start_time = time.time()

            model_groups = self._group_agents_by_model()
            logger.info(f"Processing {len(model_groups)} unique models in parallel")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(model_groups)
            ) as executor:
                model_futures = {}
                for model, agents_group in model_groups.items():
                    future = executor.submit(
                        self._process_agent_group,
                        agents=agents_group,
                        prompt=prompt,
                        json_mode=json_mode,
                        max_retries=max_retries,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        images=images,
                    )
                    model_futures[future] = model

                for future in concurrent.futures.as_completed(model_futures):
                    model = model_futures[future]
                    try:
                        model_responses = future.result()
                        responses.extend(model_responses)
                        logger.info(
                            f"Completed processing {len(model_responses)} agents for model {model}"
                        )
                    except Exception as e:
                        logger.error(f"Error processing model {model}: {str(e)}")
                        raise

            elapsed = time.time() - start_time
            logger.info(f"Received {len(responses)} responses in {elapsed:.2f}s")
        else:
            if parallel and len(unique_models) <= 1:
                logger.info("Parallel processing disabled: only one model type present")

            retries = self.max_retries if max_retries is None else max_retries
            retry_msg = f"{retries} retries" if retries > 0 else "no retries"
            logger.info(
                f"Getting responses from {len(self.agents)} agents sequentially with {retry_msg}"
            )
            start_time = time.time()

            for i, agent_info in enumerate(
                tqdm(self.agents, desc="Processing Agents", unit="agent")
            ):
                logger.info(
                    f"Requesting response from agent {i+1}/{len(self.agents)}: {agent_info.agent_id}"
                )
                try:
                    response = self._retry_with_backoff(
                        self._respond,
                        agent_info,
                        prompt,
                        images=images,
                        json_mode=json_mode,
                        timeout=int(self.timeout),
                        max_retries=0,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        retries=retries,
                    )
                except (ConnectionError, Exception) as e:
                    logger.error(
                        f"All retries failed for agent {agent_info.agent_id}, aborting."
                    )
                    raise RuntimeError(str(e)) from e

                responses.append(response)

                if self.job_delay > 0 and i < len(self.agents) - 1:
                    logger.debug(f"Waiting {self.job_delay}s before next agent")
                    time.sleep(self.job_delay)

            elapsed = time.time() - start_time
            logger.info(f"Received {len(responses)} responses in {elapsed:.2f}s")

        return responses

    def get_responses_batch(
        self,
        prompt: str,
        images: Union[str, Path, List[str], List[Path], None] = None,
        json_mode: bool = False,
        max_retries: Optional[int] = None,
        max_tokens: int = 6400,
        temperature: float = 1.0,
        parallel: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """Get batch responses from all agents for multiple prompts.

        Note: This method returns responses grouped by prompt. If you need a
        flat list of all responses, use get_responses(batch=True) instead.

        Args:
            prompts (List[str]): List of prompts to send to each agent.
            images: Optional list of images for each prompt. Must match length
                of prompts or be None. Each element corresponds to a prompt.
            json_mode (bool): Whether to expect JSON responses.
            max_retries (Optional[int]): Maximum number of retry attempts.
            max_tokens (int): Maximum number of tokens in responses.
            temperature (float): Controls randomness in the responses.
            parallel (bool): Whether to process in parallel across different models.
                Automatically disabled if only one model type is present.

        Returns:
            List[List[Dict[str, Any]]]: List where each element is a list
            containing responses from all agents for a single prompt. The outer
            list corresponds to the input prompts order.

        Raises:
            ValueError: If prompts is empty or if images length mismatches prompts.
        """
        all_agent_responses_flat = []

        unique_models = set(agent.model for agent in self.agents)
        use_parallel = parallel and len(unique_models) > 1

        if use_parallel and len(unique_models) > 1:
            logger.info(
                f"Getting batch responses in parallel mode for {len(self.agents)} agents "
            )
            start_time = time.time()

            model_groups = self._group_agents_by_model()
            logger.info(f"Processing {len(model_groups)} unique models in parallel")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(model_groups)
            ) as executor:
                model_futures = {}
                for model, agents_group in model_groups.items():
                    future = executor.submit(
                        self._process_agent_group_batch,
                        agents=agents_group,
                        prompt=prompt,
                        json_mode=json_mode,
                        max_retries=max_retries,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        images=images,
                    )
                    model_futures[future] = model

                for future in concurrent.futures.as_completed(model_futures):
                    model = model_futures[future]
                    try:
                        model_responses = future.result()
                        all_agent_responses_flat.extend(model_responses)
                        logger.info(f"Completed batch processing for model {model}")
                    except Exception as e:
                        logger.error(f"Error batch processing model {model}: {str(e)}")
                        raise

            elapsed = time.time() - start_time
            logger.info(
                f"Received batch responses from {len(self.agents)} agents in {elapsed:.2f}s (parallel)"
            )
        else:
            if parallel and len(unique_models) <= 1:
                logger.info("Parallel processing disabled: only one model type present")

            retries = self.max_retries if max_retries is None else max_retries
            retry_msg = f"{retries} retries" if retries > 0 else "no retries"
            logger.info(
                f"Getting batch responses from {len(self.agents)} agents sequentially "
                f"with {retry_msg}"
            )
            start_time = time.time()

            agent_results_list = []
            for i, agent_info in enumerate(
                tqdm(self.agents, desc="Processing Agents", unit="agent")
            ):
                logger.info(
                    f"Requesting batch responses from agent {i+1}/{len(self.agents)}: "
                    f"{agent_info.agent_id}"
                )
                try:
                    agent_responses = self._retry_with_backoff(
                        self._respond_batch,
                        agent_info,
                        prompt=prompt,
                        images=images,
                        json_mode=json_mode,
                        timeout=int(self.timeout),
                        max_retries=0,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        retries=retries,
                    )
                    for p_idx, resp in enumerate(agent_responses):
                        resp["prompt_index"] = p_idx
                    agent_results_list.append(agent_responses)

                except (ConnectionError, Exception) as e:
                    logger.error(
                        f"All retries failed for agent {agent_info.agent_id} batch request, aborting."
                    )
                    failed_responses = []
                    for p_idx in range(len(self.agents)):
                        failed_responses.append(
                            {
                                "agent_id": agent_info.agent_id,
                                "model": agent_info.model,
                                "response": f"Error: Failed after retries - {str(e)}",
                                "error": str(e),
                                "prompt_index": p_idx,
                            }
                        )
                    agent_results_list.append(failed_responses)

                if self.job_delay > 0 and i < len(self.agents) - 1:
                    logger.debug(f"Waiting {self.job_delay}s before next agent")
                    time.sleep(self.job_delay)

            all_agent_responses_flat = [
                resp for agent_resps in agent_results_list for resp in agent_resps
            ]

            elapsed = time.time() - start_time
            logger.info(
                f"Received batch responses from {len(self.agents)} agents in {elapsed:.2f}s (sequential)"
            )

        responses_by_prompt: List[List[Dict[str, Any]]] = [
            [] for _ in range(len(self.agents))
        ]
        agent_ids_order = {agent.agent_id: idx for idx, agent in enumerate(self.agents)}

        for response in all_agent_responses_flat:
            p_idx = response.get("prompt_index", -1)
            if 0 <= p_idx < len(self.agents):
                agent_order_idx = agent_ids_order.get(response["agent_id"], -1)
                if agent_order_idx != -1:
                    while len(responses_by_prompt[p_idx]) <= agent_order_idx:
                        responses_by_prompt[p_idx].append({})
                    responses_by_prompt[p_idx][agent_order_idx] = response
            else:
                logger.warning(
                    f"Response missing or has invalid prompt_index: {response}"
                )

        for p_idx in range(len(self.agents)):
            for agent_idx, agent_info in enumerate(self.agents):
                if (
                    agent_idx >= len(responses_by_prompt[p_idx])
                    or not responses_by_prompt[p_idx][agent_idx]
                ):
                    logger.warning(
                        f"Missing response for agent {agent_info.agent_id} in prompt {p_idx}"
                    )
                    placeholder = {
                        "agent_id": agent_info.agent_id,
                        "model": agent_info.model,
                        "response": "Error: Missing response",
                        "error": "Response not generated or collected",
                        "prompt_index": p_idx,
                    }
                    if agent_idx >= len(responses_by_prompt[p_idx]):
                        responses_by_prompt[p_idx].append(placeholder)
                    else:
                        responses_by_prompt[p_idx][agent_idx] = placeholder

        return responses_by_prompt

    def _process_agent_group(
        self,
        agents: List[AgentInfo],
        prompt: str,
        json_mode: bool,
        max_retries: Optional[int],
        max_tokens: int,
        temperature: float,
        images: Union[str, Path, List[str], List[Path], None] = None,
    ) -> List[Dict[str, Any]]:
        """Process a group of agents with the same model.

        Args:
            agents: List of agents with the same model
            prompt: Prompt to send
            json_mode: Whether to use JSON mode
            max_retries: Maximum retries
            max_tokens: Maximum tokens
            temperature: Temperature setting
            images: Image file paths for vision models.

        Returns:
            List of responses from all agents in the group
        """
        group_responses = []
        model_name = agents[0].model if agents else "unknown"

        logger.debug(
            f"Processing group of {len(agents)} agents with model {model_name}"
        )

        for agent_info in agents:
            try:
                response = self._retry_with_backoff(
                    self._respond,
                    agent_info,
                    prompt,
                    json_mode=json_mode,
                    images=images,
                    max_retries=max_retries,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    retries=self.max_retries,
                )
                group_responses.append(response)

                if self.job_delay > 0 and agent_info != agents[-1]:
                    time.sleep(self.job_delay)

            except Exception as e:
                logger.error(
                    f"Failed to get response from agent {agent_info.agent_id}: {str(e)}"
                )
                raise

        return group_responses

    def _process_agent_group_batch(
        self,
        agents: List[AgentInfo],
        prompt: str,
        json_mode: bool,
        max_retries: Optional[int],
        max_tokens: int,
        temperature: float,
        images: Union[str, Path, List[str], List[Path], None] = None,
    ) -> List[Dict[str, Any]]:
        """Process a group of agents with the same model for batch requests.

        Args:
            agents: List of agents with the same model
            prompt: List of prompts to send
            json_mode: Whether to use JSON mode
            max_retries: Maximum retries
            max_tokens: Maximum tokens
            temperature: Temperature setting
            images: Optional list of images for each prompt

        Returns:
            List of all agent responses for all prompts
        """
        all_responses = []
        model_name = agents[0].model if agents else "unknown"

        logger.debug(
            f"Processing batch for group of {len(agents)} agents with model {model_name}"
        )

        for agent_info in agents:
            try:
                agent_responses = self._retry_with_backoff(
                    self._respond_batch,
                    agent_info,
                    prompt,
                    json_mode=json_mode,
                    images=images,
                    max_retries=max_retries,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    retries=self.max_retries,
                )

                for i, response in enumerate(agent_responses):
                    response["prompt_index"] = i

                all_responses.extend(agent_responses)

                if self.job_delay > 0 and agent_info != agents[-1]:
                    time.sleep(self.job_delay)

            except Exception as e:
                logger.error(
                    f"Failed to get batch responses from agent {agent_info.agent_id}: {str(e)}"
                )
                raise

        return all_responses

    def get_agent_by_id(self, agent_id: int) -> AgentInfo:
        """Get an agent by its ID.

        Args:
            agent_id (int): The ID of the agent to retrieve.

        Returns:
            AgentInfo: The agent with the specified ID.

        Raises:
            ValueError: If no agent with the specified ID is found.
        """
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        raise ValueError(f"Agent with ID {agent_id} not found")

    def __len__(self) -> int:
        """Return the number of agents in the ensemble.

        Returns:
            int: The number of agents.
        """
        return len(self.agents)

    def __str__(self) -> str:
        """Return a string representation of the ensemble.

        Returns:
            str: String representation of the ensemble.
        """
        return f"AgentsEnsemble with {len(self)} agents"
