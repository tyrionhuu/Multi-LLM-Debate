from typing import List
from ..utils.config_manager import get_models
from .agent import Agent


class AgentsEnsemble:
    def __init__(self, auto_init: bool = True):
        self.agents: List[Agent] = []
        if auto_init:
            self._initialize_from_config()

    def _initialize_from_config(self) -> None:
        """Initialize agents from configuration."""
        models = get_models()
        agent_id = 0
        for provider, model_name, quantity in models:
            for _ in range(quantity):
                agent = Agent(agent_id=agent_id, model=model_name, provider=provider)
                self.add_agent(agent)
                agent_id += 1

    @classmethod
    def create_from_config(cls) -> 'AgentsEnsemble':
        """Factory method to create an ensemble from configuration."""
        return cls(auto_init=True)

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the ensemble."""
        self.agents.append(agent)

    def get_responses(self, prompt: str) -> List[str]:
        """Get responses from all agents for a given prompt."""
        responses = []
        for agent in self.agents:
            response = agent.respond(prompt)
            responses.append(response)
        return responses

    def get_agent_by_id(self, agent_id: int) -> Agent:
        """Get an agent by its ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        raise ValueError(f"Agent with ID {agent_id} not found")

    def __len__(self) -> int:
        return len(self.agents)

    def __str__(self) -> str:
        return f"AgentsEnsemble with {len(self)} agents"
