import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from ..utils.config_manager import get_models
from ..utils.logging_config import setup_logging
from ..utils.model_config import ModelConfig
from .agent import Agent

# Use setup_logging to ensure consistent logging
logger = setup_logging(__name__)
logger.setLevel(logging.INFO)


class AgentsEnsemble:
    """A collection of LLM agents that can be used together.

    This class manages multiple Agent instances and provides methods to interact with them
    collectively. It can be initialized automatically from configuration or built manually.

    Attributes:
        agents (List[Agent]): List of Agent instances in the ensemble.
        job_delay (float): Delay in seconds between consecutive agent calls.
        timeout (float): Maximum time in seconds to wait for agent responses.
        max_retries (int): Maximum number of retry attempts for failed requests.
    """

    def __init__(
        self,
        config_list: Optional[List[ModelConfig]] = None,
        job_delay: float = 0.5,
        timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize an AgentsEnsemble instance.

        Args:
            config_list (Optional[List[ModelConfig]]): List of model configurations.
            job_delay (float, optional): Delay in seconds between agent calls.
            timeout (float, optional): Maximum time in seconds to wait for agent responses.
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        """
        self.job_delay = job_delay
        self.timeout = timeout
        self.max_retries = max_retries
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
        for model_info in models:
            base_url = None
            
            # Handle different formats returned by get_models()
            if isinstance(model_info, tuple) and len(model_info) >= 3:
                # Old format: (provider, model_name, quantity)
                # We'll use provider as base_url if it's not "openai"
                provider, model_name, quantity = model_info[:3]
                if provider.lower() != "openai":
                    base_url = provider
            elif isinstance(model_info, dict):
                # New format: {"name": model_name, "base_url": url, "quantity": qty}
                model_name = model_info.get("name", "")
                base_url = model_info.get("base_url")
                quantity = model_info.get("quantity", 1)
            else:
                logger.warning(f"Skipping unrecognized model info format: {model_info}")
                continue
                
            for _ in range(quantity):
                agent = Agent(agent_id=agent_id, model=model_name, base_url=base_url)
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
            base_url = None
            
            # Convert provider to base_url if present
            if "provider" in config and config["provider"].lower() != "openai":
                base_url = config["provider"]
            elif "base_url" in config:
                base_url = config["base_url"]
                
            quantity = config.get("quantity", 1)
            model_name = config["name"]
            
            for _ in range(quantity):
                agent = Agent(
                    agent_id=agent_id, model=model_name, base_url=base_url
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
        self,
        agent: Agent,
        prompt: str,
        json_mode: bool,
        max_retries: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attempt to get a response from an agent with retry logic.

        Args:
            agent (Agent): The agent to get a response from.
            prompt (str): The input prompt to send to the agent.
            json_mode (bool): Whether to expect JSON response.
            max_retries (Optional[int], optional): Maximum number of retry attempts.
                If None, use the ensemble's default max_retries. Defaults to None.
            api_key (Optional[str], optional): API key to use for this request.
                Defaults to None, which uses the one from config.

        Returns:
            Dict[str, Any]: Response from the agent.

        Raises:
            ConnectionError: If there's a network or timeout issue after all retries.
            Exception: If some other error occurs after all retries.
        """
        # Use ensemble's default if max_retries not specified
        retries = self.max_retries if max_retries is None else max_retries

        if retries <= 0:
            # No retries, call agent.respond directly
            logger.debug(
                f"No retries set for agent {agent.agent_id} ({agent.model})"
            )
            return agent.respond(
                prompt, 
                json_mode=json_mode, 
                timeout=int(self.timeout), 
                max_retries=0,
                api_key=api_key
            )

        # Use the agent's built-in retry mechanism
        logger.debug(
            f"Using {retries} retries for agent {agent.agent_id} ({agent.model})"
        )
        return agent.respond(
            prompt, 
            json_mode=json_mode, 
            timeout=int(self.timeout), 
            max_retries=retries,
            api_key=api_key
        )

    def get_responses(
        self, 
        prompt: str, 
        json_mode: bool = False, 
        max_retries: Optional[int] = None,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get responses from all agents for a given prompt.

        Args:
            prompt (str): The input prompt to send to all agents.
            json_mode (bool, optional): Whether to expect JSON response.
                Defaults to False.
            max_retries (Optional[int], optional): Maximum number of retry attempts.
                If None, use the ensemble's default max_retries. Defaults to None.
            api_key (Optional[str], optional): API key to use for this request.
                Defaults to None, which uses the one from config.

        Returns:
            List[Dict[str, Any]]: List of responses from all agents.

        Raises:
            RuntimeError: If an agent fails all retries.
        """
        retries = self.max_retries if max_retries is None else max_retries
        retry_msg = f"{retries} retries" if retries > 0 else "no retries"
        logger.info(
            f"Getting responses from {len(self.agents)} agents sequentially with {retry_msg}"
        )
        start_time = time.time()

        responses = []

        for i, agent in enumerate(
            tqdm(self.agents, desc="Processing Agents", unit="agent")
        ):
            logger.info(
                f"Requesting response from agent {i+1}/{len(self.agents)}: {agent.agent_id}"
            )
            try:
                response = self._get_response_with_retry(
                    agent, prompt, json_mode, max_retries=max_retries, api_key=api_key
                )
            except (ConnectionError, Exception) as e:
                logger.error(
                    f"All retries failed for agent {agent.agent_id}, aborting."
                )
                raise RuntimeError(str(e)) from e

            responses.append(response)

            if self.job_delay > 0 and i < len(self.agents) - 1:
                logger.debug(f"Waiting {self.job_delay}s before next agent")
                time.sleep(self.job_delay)

        elapsed = time.time() - start_time
        logger.info(f"Received {len(responses)} responses in {elapsed:.2f}s")

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
        """Return the number of agents in the ensemble.
        
        Returns:
            int: The number of agents.
        """
        return len(self.agents)

    def __str__(self) -> str:
        """Return a string representation of the ensemble.
        
        Returns:
            str: String representation of the ensemble.
        """
        return f"AgentsEnsemble with {len(self)} agents"
