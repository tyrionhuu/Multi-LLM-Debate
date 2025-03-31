import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError, wait
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils.config_manager import get_models
from ..utils.model_config import ModelConfig
from .agent import Agent, LLMConnectionError


class AgentsEnsemble:
    """A collection of LLM agents that can be used together.

    This class manages multiple Agent instances and provides methods to interact with them
    collectively. It can be initialized automatically from configuration or built manually.

    Attributes:
        agents (List[Agent]): List of Agent instances in the ensemble.
        concurrent (bool): Whether to use concurrent execution for responses.
        max_workers (int): Maximum number of concurrent workers when concurrent is True.
        job_delay (float): Delay in seconds between consecutive agent calls.
        timeout (float): Maximum time in seconds to wait for agent responses.
        max_retries (int): Maximum number of retry attempts for failed requests.
        retry_delay (float): Delay in seconds between retry attempts.
    """

    def __init__(
        self,
        config_list: Optional[List[ModelConfig]] = None,
        concurrent: bool = True,
        max_workers: Optional[int] = 4,
        job_delay: float = 0.5,
        timeout: float = 180.0,  # 3 minute timeout
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize an AgentsEnsemble instance.

        Args:
            config_list (Optional[List[ModelConfig]]): List of model configurations.
                If None, default configs will be loaded.
            concurrent (bool, optional): Whether to use concurrent execution. Defaults to True.
            max_workers (int, optional): Maximum number of concurrent workers. Defaults to 4.
            job_delay (float, optional): Delay in seconds between agent calls. Defaults to 0.5.
            timeout (float, optional): Maximum time in seconds to wait for agent responses.
                Defaults to 180.0 (3 minutes).
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 2.
            retry_delay (float, optional): Delay in seconds between retry attempts.
                Defaults to 1.0.

        Raises:
            ValueError: If initialization fails.
        """
        self.concurrent = concurrent
        self.max_workers = max_workers
        self.job_delay = job_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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
        for provider, model_name, quantity in models:
            for _ in range(quantity):
                agent = Agent(agent_id=agent_id, model=model_name, provider=provider)
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
            for _ in range(config["quantity"]):
                agent = Agent(
                    agent_id=agent_id, model=config["name"], provider=config["provider"]
                )
                self.add_agent(agent)
                agent_id += 1

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the ensemble.

        Args:
            agent (Agent): The agent instance to add to the ensemble.
        """
        self.agents.append(agent)

    def _get_response_with_retry(
        self, agent: Agent, prompt: str, json_mode: bool
    ) -> Dict[str, Any]:
        """Attempt to get a response from an agent with retry logic.

        Args:
            agent (Agent): The agent to get a response from.
            prompt (str): The input prompt to send to the agent.
            json_mode (bool): Whether to expect JSON response.

        Returns:
            Dict[str, Any]: Response from the agent.

        Raises:
            LLMConnectionError: If all retry attempts fail.
        """
        errors = []
        for attempt in range(self.max_retries + 1):
            try:
                return agent.respond(
                    prompt, json_mode=json_mode, timeout=int(self.timeout)
                )
            except LLMConnectionError as e:
                errors.append(f"Attempt {attempt+1}: {str(e)}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        raise LLMConnectionError(
            f"Failed after {self.max_retries + 1} attempts: {'; '.join(errors)}"
        )

    def _get_response_concurrent(
        self, prompt: str, json_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """Get responses from all agents concurrently with improved resource management.

        Uses a controlled batch approach to prevent overwhelming Ollama and better
        manage resources, especially when running with local LLMs.

        Args:
            prompt (str): The input prompt to send to all agents.
            json_mode (bool, optional): Whether to expect JSON response. Defaults to False.

        Returns:
            List[Dict[str, Any]]: List of responses from all agents.

        Raises:
            LLMConnectionError: If no responses could be collected at all.
        """
        responses = []
        errors = []
        timeout_errors = []

        # Log the start of the process
        logging.info(f"Starting concurrent requests for {len(self.agents)} agents")

        # If max_workers is None or 1, use sequential processing
        if self.max_workers is None or self.max_workers <= 1:
            for agent in self.agents:
                try:
                    response = self._get_response_with_retry(agent, prompt, json_mode)
                    responses.append(response)
                except LLMConnectionError as e:
                    errors.append(str(e))

                if self.job_delay > 0:
                    time.sleep(self.job_delay)
        else:
            # Determine optimal batch size based on max_workers
            # For Ollama, smaller batches often work better
            batch_size = min(3, self.max_workers)

            # Calculate per-agent timeout with margin
            # Give each agent slightly less than the total timeout to allow for scheduling
            per_agent_timeout = max(30, self.timeout * 0.9)

            # Group agents by provider to optimize scheduling
            agents_by_provider = {}
            for agent in self.agents:
                provider = agent.provider.lower()
                if provider not in agents_by_provider:
                    agents_by_provider[provider] = []
                agents_by_provider[provider].append(agent)

            logging.info(
                f"Agent distribution by provider: {', '.join([f'{k}: {len(v)}' for k, v in agents_by_provider.items()])}"
            )

            # Process each provider's agents separately with appropriate batching
            for provider, provider_agents in agents_by_provider.items():
                # For Ollama, use smaller batches and add delays
                provider_batch_size = batch_size
                provider_delay = self.job_delay
                if provider == "ollama":
                    provider_batch_size = min(2, batch_size)
                    provider_delay = max(
                        1.0, self.job_delay
                    )  # Ensure at least 1 second delay for Ollama

                logging.info(
                    f"Processing {len(provider_agents)} {provider} agents with batch size {provider_batch_size}"
                )

                # Process this provider's agents in batches
                for i in range(0, len(provider_agents), provider_batch_size):
                    batch = provider_agents[i : i + provider_batch_size]
                    batch_results = self._process_agent_batch(
                        batch, prompt, json_mode, per_agent_timeout, provider_delay
                    )

                    batch_responses, batch_errors, batch_timeouts = batch_results
                    responses.extend(batch_responses)
                    errors.extend(batch_errors)
                    timeout_errors.extend(batch_timeouts)

                    # Add a delay between batches to prevent overwhelming Ollama
                    if provider == "ollama" and i + provider_batch_size < len(
                        provider_agents
                    ):
                        delay_time = max(2.0, provider_delay * 2)
                        logging.info(
                            f"Adding {delay_time}s delay between Ollama batches"
                        )
                        time.sleep(delay_time)

        total_errors = len(errors) + len(timeout_errors)
        if responses:
            logging.info(
                f"Completed with {len(responses)} responses and {total_errors} errors"
            )

        # Only raise an error if we got no responses at all
        if not responses:
            error_messages = []
            if errors:
                error_messages.append(f"Connection errors: {'; '.join(errors)}")
            if timeout_errors:
                error_messages.append(f"Timeout errors: {'; '.join(timeout_errors)}")
            raise LLMConnectionError("; ".join(error_messages))

        # Log warnings about partial errors if we got some responses
        if errors or timeout_errors:
            if errors:
                logging.warning(f"Some agents encountered errors: {'; '.join(errors)}")
            if timeout_errors:
                logging.warning(f"Some agents timed out: {'; '.join(timeout_errors)}")

        return responses

    def _process_agent_batch(
        self,
        agents: List[Agent],
        prompt: str,
        json_mode: bool,
        timeout_per_agent: float,
        delay_between_agents: float,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Process a batch of agents with controlled concurrency.

        Args:
            agents: List of agents to process
            prompt: The prompt to send to each agent
            json_mode: Whether to use JSON mode
            timeout_per_agent: Timeout for each individual agent
            delay_between_agents: Delay between submitting agent tasks

        Returns:
            Tuple containing (responses, errors, timeout_errors)
        """
        responses = []
        errors = []
        timeout_errors = []

        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {}
            active_futures: Set = set()

            # Submit all jobs for this batch
            for agent in agents:
                if delay_between_agents > 0:
                    time.sleep(delay_between_agents)
                future = executor.submit(
                    self._get_response_with_retry, agent, prompt, json_mode
                )
                futures[future] = agent
                active_futures.add(future)

                # Log submission
                logging.debug(
                    f"Submitted request to {agent.provider} agent {agent.agent_id} ({agent.model})"
                )

            # Set the end time for our batch timeout
            end_time = (
                time.time() + timeout_per_agent + (len(agents) * delay_between_agents)
            )

            # Process futures as they complete
            while active_futures and time.time() < end_time:
                # Wait for the next future to complete with a short timeout
                timeout_remaining = max(0.1, end_time - time.time())
                completed, active_futures = wait(
                    active_futures,
                    timeout=timeout_remaining,
                    return_when=FIRST_COMPLETED,
                )

                # Process completed futures
                for future in completed:
                    agent = futures[future]
                    try:
                        response = future.result(
                            timeout=0.1
                        )  # Short timeout as it should be done
                        responses.append(response)
                        logging.info(
                            f"Received response from {agent.provider} agent {agent.agent_id}"
                        )
                    except TimeoutError:
                        timeout_errors.append(
                            f"Agent {agent.agent_id} ({agent.model}) timed out during result retrieval"
                        )
                        logging.warning(
                            f"Timeout retrieving result from agent {agent.agent_id}"
                        )
                    except LLMConnectionError as e:
                        error_msg = (
                            f"Agent {agent.agent_id} ({agent.model}) error: {str(e)}"
                        )
                        errors.append(error_msg)
                        logging.warning(
                            f"Connection error from agent {agent.agent_id}: {str(e)}"
                        )

            # For any remaining active futures, record them as timeouts
            if active_futures:
                for future in active_futures:
                    agent = futures[future]
                    timeout_errors.append(
                        f"Agent {agent.agent_id} ({agent.model}, {agent.provider}) timed out after {timeout_per_agent} seconds"
                    )
                    # Cancel any remaining futures
                    future.cancel()
                logging.warning(f"{len(active_futures)} agents timed out in batch")

        return responses, errors, timeout_errors

    def get_responses(
        self, prompt: str, json_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """Get responses from all agents for a given prompt.

        Args:
            prompt (str): The input prompt to send to all agents.
            json_mode (bool, optional): Whether to expect JSON response. Defaults to False.

        Returns:
            List[Dict[str, Any]]: List of responses from all agents.

        Raises:
            LLMConnectionError: If any agent encounters a connection error.
        """
        if self.concurrent:
            return self._get_response_concurrent(prompt, json_mode=json_mode)

        responses = []
        errors = []

        for agent in self.agents:
            try:
                response = self._get_response_with_retry(agent, prompt, json_mode)
                responses.append(response)
            except LLMConnectionError as e:
                errors.append(str(e))

            if self.job_delay > 0:
                time.sleep(self.job_delay)

        if errors:
            raise LLMConnectionError(
                f"Connection errors occurred with some agents: {'; '.join(errors)}"
            )

        return responses

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
        return len(self.agents)

    def __str__(self) -> str:
        return f"AgentsEnsemble with {len(self)} agents"
