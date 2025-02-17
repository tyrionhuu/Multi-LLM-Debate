class Agent:
    def __init__(self, agent_id: int, model: str):
        self.agent_id = agent_id
        self.model = model
        
    def __str__(self):
        return f"Agent {self.agent_id} ({self.model})"