from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.config_manager import get_models
from .agent import Agent


class AgentsEnsemble:
    """A collection of LLM agents that can be used together.

    This class manages multiple Agent instances and provides methods to interact with them
    collectively. It can be initialized automatically from configuration or built manually.

    Attributes:
        agents (List[Agent]): List of Agent instances in the ensemble.
        concurrent (bool): Whether to use concurrent execution for responses.
        max_workers (int): Maximum number of concurrent workers when concurrent is True.
    """

    def __init__(self, auto_init: bool = True, concurrent: bool = False, max_workers: int = None) -> None:
        """Initialize an AgentsEnsemble instance.

        Args:
            auto_init (bool, optional): Whether to automatically initialize agents from config.
                Defaults to True.
            concurrent (bool, optional): Whether to use concurrent execution. Defaults to False.
            max_workers (int, optional): Maximum number of concurrent workers. Defaults to None
                (ThreadPoolExecutor default).
        """
        self.agents: List[Agent] = []
        self.concurrent = concurrent
        self.max_workers = max_workers
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

    @classmethod
    def create_from_config(cls) -> "AgentsEnsemble":
        """Factory method to create an ensemble from configuration.

        Returns:
            AgentsEnsemble: A new instance initialized from configuration.
        """
        return cls(auto_init=True)

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the ensemble.

        Args:
            agent (Agent): The agent instance to add to the ensemble.
        """
        self.agents.append(agent)

    def _get_response_concurrent(self, prompt: str) -> List[Dict[str, Any]]:
        """Get responses from all agents concurrently.

        Args:
            prompt (str): The input prompt to send to all agents.

        Returns:
            List[Dict[str, Any]]: List of response dictionaries from all agents.
        """
        responses = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent = {
                executor.submit(agent.respond, prompt): agent
                for agent in self.agents
            }
            for future in as_completed(future_to_agent):
                response = future.result()
                responses.append(response)
        return responses

    def get_responses(self, prompt: str) -> List[Dict[str, Any]]:
        """Get responses from all agents for a given prompt.

        Args:
            prompt (str): The input prompt to send to all agents.

        Returns:
            List[Dict[str, Any]]: List of response dictionaries from all agents.
            Each dictionary contains agent information and the parsed response.
        """
        if self.concurrent:
            return self._get_response_concurrent(prompt)
        
        responses = []
        for agent in self.agents:
            response = agent.respond(prompt)
            responses.append(response)
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
