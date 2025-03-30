import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any, Dict, List, Optional

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
        timeout: float = 120.0,  # 2 minute timeout
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
                Defaults to 120.0 (2 minutes).
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
        self.agents = []

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
        """Get responses from all agents concurrently.

        Args:
            prompt (str): The input prompt to send to all agents.
            json_mode (bool, optional): Whether to expect JSON response. Defaults to False.

        Returns:
            List[Dict[str, Any]]: List of responses from all agents.

        Raises:
            LLMConnectionError: If any agent encounters a connection error.
        """
        responses = []
        errors = []

        # If max_workers is None or 1, use sequential processing instead of ThreadPoolExecutor
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
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}

                for agent in self.agents:
                    if self.job_delay > 0:
                        time.sleep(self.job_delay)
                    future = executor.submit(
                        self._get_response_with_retry, agent, prompt, json_mode
                    )
                    futures[future] = agent.agent_id

                for future in as_completed(futures, timeout=self.timeout):
                    try:
                        response = future.result(
                            timeout=1
                        )  # Short timeout to check completion
                        responses.append(response)
                    except TimeoutError:
                        agent_id = futures[future]
                        errors.append(
                            f"Agent {agent_id} timed out after {self.timeout} seconds"
                        )
                    except LLMConnectionError as e:
                        errors.append(str(e))

        if errors:
            raise LLMConnectionError(
                f"Connection errors occurred with some agents: {'; '.join(errors)}"
            )

        return responses

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
