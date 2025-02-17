import json
from typing import Dict, Any
from ..llm.llm import call_model


class Agent:
    def __init__(self, agent_id: int, model: str, provider: str):
        self.agent_id = agent_id
        self.model = model
        self.provider = provider

    def __str__(self):
        return f"Agent {self.agent_id} ({self.model})"

    def __repr__(self):
        return str(self)

    def respond(self, prompt: str) -> Dict[str, Any]:
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
            "provider": self.provider,
            "response": parsed_response
        }
