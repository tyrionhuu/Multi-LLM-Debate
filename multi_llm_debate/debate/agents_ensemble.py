import concurrent.futures
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from ..utils.config_manager import get_models
from ..utils.model_config import ModelConfig
from .agent import Agent

# Use setup_logging to ensure consistent logging
logger = logging.getLogger(__name__)


class AgentsEnsemble:
    """A collection of LLM agents that can be used together.

    This class manages multiple Agent instances and provides methods to interact with them
    collectively. It can be initialized automatically from configuration or built manually.

    Attributes:
        agents (List[Agent]): List of Agent instances in the ensemble.
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
        self.agents: List[Agent] = []  # List to hold Agent instances

        if config_list is not None:
            self._initialize_from_config_list(config_list)
        else:
            self._initialize_from_config()

    def _initialize_from_config(self) -> None:
        """Initialize agents from configuration.

        Loads model configurations and creates Agent instances accordingly.
        Each agent is assigned a unique ID starting from 0.
        """
        models = get_models()
        agent_id = 0
        for model_info in models:
            base_url = None
            api_key = None  # Add api_key support

            # Handle different formats returned by get_models()
            if isinstance(model_info, tuple) and len(model_info) >= 3:
                # Old format: (model_name, base_url, quantity)
                model_name, base_url, quantity = model_info[:3]
                # No api_key in old format
            elif isinstance(model_info, dict):
                # New format: {"name": model_name, "base_url": url, "quantity": qty, "api_key": ...}
                model_name = model_info.get("name", "")
                base_url = model_info.get("base_url")
                quantity = model_info.get("quantity", 1)
                api_key = model_info.get("api_key")  # Extract api_key if present
            else:
                logger.warning(f"Skipping unrecognized model info format: {model_info}")
                continue

            for _ in range(quantity):
                agent = Agent(
                    agent_id=agent_id,
                    model=model_name,
                    base_url=base_url,
                    api_key=api_key,
                )
                self.add_agent(agent)
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
            api_key = config.get("api_key")  # Extract api_key if present

            for _ in range(quantity):
                agent = Agent(
                    agent_id=agent_id,
                    model=model_name,
                    base_url=base_url,
                    api_key=api_key,
                )
                self.add_agent(agent)
                agent_id += 1

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the ensemble.

        Args:
            agent (Agent): The agent instance to add to the ensemble.
        """
        self.agents.append(agent)

    def _count_unique_models(self) -> int:
        """Return the number of unique models among agents."""
        return len({agent.model for agent in self.agents})

    def _get_response_with_retry(
        self,
        agent: Agent,
        prompt: str,
        json_mode: bool,
        images: Union[str, Path, List[str], List[Path], None] = None,
        max_retries: Optional[int] = None,
        max_tokens: int = 6400,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Attempt to get a response from an agent with retry logic.

        Args:
            agent (Agent): The agent to get a response from.
            prompt (str): The input prompt to send to the agent.
            json_mode (bool): Whether to expect JSON response.
            images (Union[str, Path, List[str], List[Path], None], optional):
                Image file paths for vision models. Can be a single path or a list.
            max_retries (Optional[int], optional): Maximum number of retry attempts.
                If None, use the ensemble's default max_retries. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens in response.
                Defaults to 6400.
            temperature (float, optional): Controls randomness in the response.
                Defaults to 1.0. Lower values make responses more deterministic.

        Returns:
            Dict[str, Any]: Response from the agent.

        Raises:
            ConnectionError: If there's a network or timeout issue after all retries.
            Exception: If some other error occurs after all retries.
        """
        retries = self.max_retries if max_retries is None else max_retries
        base_delay = 1.0  # seconds
        max_delay = 16.0  # seconds

        attempt = 0
        while True:
            try:
                return agent.respond(
                    prompt=prompt,
                    images=images,
                    json_mode=json_mode,
                    timeout=int(self.timeout),
                    max_retries=0,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except (ConnectionError, Exception) as e:
                if attempt >= retries:
                    logger.error(
                        f"All retries failed for agent {agent.agent_id} ({agent.model})"
                    )
                    raise
                # Exponential backoff with jitter
                delay = min(max_delay, base_delay * (2**attempt))
                jitter = random.uniform(0, delay / 2)
                total_delay = delay + jitter
                logger.warning(
                    f"Retry {attempt+1}/{retries} for agent {agent.agent_id} "
                    f"({agent.model}) after {total_delay:.2f}s due to error: {e}"
                )
                time.sleep(total_delay)
                attempt += 1

    def get_responses(
        self,
        prompt: str,
        images: Union[str, Path, List[str], List[Path], None] = None,
        json_mode: bool = False,
        max_retries: Optional[int] = None,
        max_tokens: int = 6400,
        temperature: float = 1.0,
        parallel: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get responses from all agents.

        Args:
            prompt (str): Prompt to send.
            images (Union[str, Path, List[str], List[Path], None]): Image file paths
                for vision models. Can be a single path or a list.
            json_mode (bool): Expect JSON responses.
            max_retries (Optional[int]): Max retries.
            max_tokens (int): Max tokens.
            temperature (float): Response randomness.
            parallel (bool): Whether to process in parallel.

        Returns:
            List[Dict[str, Any]]: Agent responses.
        """
        responses = []
        if parallel:
            logger.info("Getting responses in parallel mode")
            start_time = time.time()

            # Group agents by model type
            model_groups = {}
            for agent in self.agents:
                if agent.model not in model_groups:
                    model_groups[agent.model] = []
                model_groups[agent.model].append(agent)

            logger.info(f"Processing {len(model_groups)} unique models in parallel")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(model_groups)
            ) as executor:
                # Submit one job per model type
                model_futures = {}
                for model, agents_group in model_groups.items():
                    future = executor.submit(
                        self._process_agent_group,
                        agents=agents_group,
                        prompt=prompt,
                        json_mode=json_mode,
                        images=images,
                        max_retries=max_retries,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    model_futures[future] = model

                # Collect results
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
            retries = self.max_retries if max_retries is None else max_retries
            retry_msg = f"{retries} retries" if retries > 0 else "no retries"
            logger.info(
                f"Getting responses from {len(self.agents)} agents sequentially with {retry_msg}"
            )
            start_time = time.time()

            for i, agent in enumerate(
                tqdm(self.agents, desc="Processing Agents", unit="agent")
            ):
                logger.info(
                    f"Requesting response from agent {i+1}/{len(self.agents)}: {agent.agent_id}"
                )
                try:
                    response = self._get_response_with_retry(
                        agent=agent,
                        prompt=prompt,
                        json_mode=json_mode,
                        images=images,
                        max_retries=max_retries,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except (ConnectionError, Exception) as e:
                    logger.error(
                        f"All retries failed for agent {agent.agent_id}, aborting."
                    )
                    raise RuntimeError(str(e)) from e

                responses.append(response)

                if self.job_delay > 0 and i < len(self.agents) - 1:
                    logger.debug(f"Waiting {self.job_delay}s before next agent")
                    time.sleep(self.job_delay)

            elapsed = time.time() - start_time
            logger.info(f"Received {len(responses)} responses in {elapsed:.2f}s")

        return responses

    def _process_agent_group(
        self,
        agents: List[Agent],
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
            api_key: Optional API key
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

        for agent in agents:
            try:
                response = self._get_response_with_retry(
                    agent=agent,
                    prompt=prompt,
                    json_mode=json_mode,
                    images=images,
                    max_retries=max_retries,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                group_responses.append(response)

                # Apply job delay between requests to the same model
                if self.job_delay > 0 and agent != agents[-1]:
                    time.sleep(self.job_delay)

            except Exception as e:
                logger.error(
                    f"Failed to get response from agent {agent.agent_id}: {str(e)}"
                )
                raise

        return group_responses

    def get_agent_by_id(self, agent_id: int) -> Agent:
        """Get an agent by its ID.

        Args:
            agent_id (int): The ID of the agent to retrieve.

        Returns:
            Agent: The agent with the specified ID.

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
