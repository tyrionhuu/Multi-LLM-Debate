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
    
    def respond(self, prompt: str) -> str:
        response = call_model(
            model_name=self.model,
            provider=self.provider,
            prompt=prompt,
        )