import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TypedDict

from tqdm import tqdm

from ..utils.config_manager import get_models
from .agent import Agent
from ..utils.model_config import ModelConfig




class AgentsEnsemble:
    """A collection of LLM agents that can be used together.

    This class manages multiple Agent instances and provides methods to interact with them
    collectively. It can be initialized automatically from configuration or built manually.

    Attributes:
        agents (List[Agent]): List of Agent instances in the ensemble.
        concurrent (bool): Whether to use concurrent execution for responses.
        max_workers (int): Maximum number of concurrent workers when concurrent is True.
        job_delay (float): Delay in seconds between consecutive agent calls.
    """

    def __init__(
        self,
        agents: Optional[List[Agent]] = None,
        auto_init: bool = True,
        concurrent: bool = True,
        max_workers: Optional[int] = 4,
        job_delay: float = 0.5,
        config_list: Optional[List[ModelConfig]] = None,
    ) -> None:
        """Initialize an AgentsEnsemble instance.

        Args:
            agents (Optional[List[Agent]], optional): List of pre-configured agents.
                If provided, other initialization methods will be ignored. Defaults to None.
            auto_init (bool, optional): Whether to automatically initialize agents from config.
                Defaults to True.
            concurrent (bool, optional): Whether to use concurrent execution. Defaults to True.
            max_workers (int, optional): Maximum number of concurrent workers. Defaults to 4.
            job_delay (float, optional): Delay in seconds between agent calls. Defaults to 0.5.
            config_list (Optional[List[ModelConfig]], optional): List of model configurations.
                If provided, auto_init will be ignored. Defaults to None.

        Raises:
            ValueError: If agents list is empty or if initialization fails.
        """
        self.concurrent = concurrent
        self.max_workers = max_workers
        self.job_delay = job_delay

        if agents is not None:
            if not agents:
                raise ValueError("Cannot initialize ensemble with empty agents list")
            self.agents = agents
        elif config_list is not None:
            self.agents = []
            self._initialize_from_config_list(config_list)
        else:
            self.agents = []
            if auto_init:
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

    @classmethod
    def create_from_config(cls) -> "AgentsEnsemble":
        """Factory method to create an ensemble from configuration.

        Returns:
            AgentsEnsemble: A new instance initialized from configuration.
        """
        return cls(auto_init=True)

    @classmethod
    def create_from_config_list(
        cls, config_list: List[ModelConfig], **kwargs
    ) -> "AgentsEnsemble":
        """Factory method to create an ensemble from a configuration list.

        Args:
            config_list (List[ModelConfig]): List of model configurations.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            AgentsEnsemble: A new instance initialized from the configuration list.
        """
        return cls(config_list=config_list, auto_init=False, **kwargs)

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the ensemble.

        Args:
            agent (Agent): The agent instance to add to the ensemble.
        """
        self.agents.append(agent)

    def _get_response_concurrent(
        self, prompt: str, json_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """Get responses from all agents concurrently.

        Args:
            prompt (str): The input prompt to send to all agents.
            json_mode (bool, optional): Whether to request JSON formatted responses.
                Defaults to False.

        Returns:
            List[Dict[str, Any]]: List of response dictionaries from all agents.
        """
        responses = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            # Submit jobs with delay
            for agent in self.agents:
                if self.job_delay > 0:
                    time.sleep(self.job_delay)
                futures.append(
                    executor.submit(agent.respond, prompt, json_mode=json_mode)
                )

            # Use tqdm to create progress bar
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Collecting agent responses",
                unit="agent",
            ):
                response = future.result()
                responses.append(response)
        return responses

    def get_responses(
        self, prompt: str, json_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """Get responses from all agents for a given prompt.

        Args:
            prompt (str): The input prompt to send to all agents.
            json_mode (bool, optional): Whether to request JSON formatted responses.
                Defaults to False.

        Returns:
            List[Dict[str, Any]]: List of response dictionaries from all agents.
            Each dictionary contains agent information and the parsed response.
        """
        if self.concurrent:
            return self._get_response_concurrent(prompt, json_mode=json_mode)

        responses = []
        for agent in tqdm(self.agents, desc="Collecting agent responses", unit="agent"):
            response = agent.respond(prompt, json_mode=json_mode)
            responses.append(response)
            if self.job_delay > 0:
                time.sleep(self.job_delay)
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
