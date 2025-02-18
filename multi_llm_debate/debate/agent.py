import json
from typing import Any, Dict

from ..llm.llm import call_model


class Agent:
    """A class representing an individual LLM agent.

    This class encapsulates a language model agent with specific provider and model configurations.
    Each agent has a unique ID and can generate responses to prompts.

    Attributes:
        agent_id (int): Unique identifier for the agent.
        model (str): Name of the language model being used.
        provider (str): Name of the model provider (e.g., 'OpenAI', 'Anthropic').
    """

    def __init__(self, agent_id: int, model: str, provider: str) -> None:
        """Initialize an Agent instance.

        Args:
            agent_id (int): Unique identifier for the agent.
            model (str): Name of the language model.
            provider (str): Name of the model provider.
        """
        self.agent_id = agent_id
        self.model = model
        self.provider = provider.lower()

    def __str__(self):
        return f"Agent {self.agent_id} ({self.model})"

    def __repr__(self):
        return str(self)

    def respond(self, prompt: str) -> Dict[str, Any]:
        """Generate a response to the given prompt.

        Args:
            prompt (str): The input prompt to send to the language model.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - agent_id: The ID of the responding agent
                - model: The model name
                - provider: The provider name
                - response: The parsed response from the model (either as dict or wrapped raw content)

        Raises:
            Any exceptions from the underlying LLM call are propagated.
        """
        raw_response = call_model(
            model_name=self.model,
            provider=self.provider,
            prompt=prompt,
        )

        try:
            # Try to parse the response as JSON if it's a string
            if isinstance(raw_response, str):
                parsed_response = json.loads(raw_response)
            elif isinstance(raw_response, dict):
                parsed_response = raw_response
            else:
                parsed_response = {"raw_content": str(raw_response)}

        except json.JSONDecodeError:
            # If parsing fails, wrap the raw response in a dict
            parsed_response = {"raw_content": raw_response}

        return {
            "agent_id": self.agent_id,
            "model": self.model,
            # "provider": self.provider,
            "response": parsed_response,
        }
