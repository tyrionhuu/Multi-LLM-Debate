from typing import List
from .agent import Agent

class AgentsEnsemble:
    def __init__(self):
        self.agents: List[Agent] = []

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