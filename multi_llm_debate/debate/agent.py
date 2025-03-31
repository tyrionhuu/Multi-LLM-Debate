import json
import logging
import time
from typing import Any, Dict, Optional

from ..llm.llm import call_model
from ..utils.logging_config import setup_logging

# Set up logger
logger = setup_logging(__name__)
logger.setLevel(logging.INFO)


class LLMConnectionError(Exception):
    """Raised when there is a connection error with the LLM service."""

    pass


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
        logger.debug(f"Initialized Agent {agent_id} with {provider} model {model}")

    def __str__(self):
        return f"Agent {self.agent_id} ({self.model})"

    def __repr__(self):
        return str(self)

    def respond(
        self, prompt: str, json_mode: bool = False, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a response to the given prompt.

        Args:
            prompt (str): The input prompt to send to the language model.
            json_mode (bool, optional): Whether to expect JSON response. Defaults to False.
            timeout (Optional[int], optional): Maximum time to wait for response in seconds.
                Defaults to None, which uses the API's default timeout.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - agent_id: The ID of the responding agent
                - model: The model name
                - response: The model's response (can be dict or str)

        Raises:
            LLMConnectionError: If there is a connection error with the LLM service.
        """
        start_time = time.time()
        logger.debug(
            f"Agent {self.agent_id} ({self.provider}/{self.model}) starting request "
            f"(timeout: {timeout}s, json_mode: {json_mode})"
        )

        # Truncate prompt for logging
        prompt_preview = prompt[:100] + ("..." if len(prompt) > 100 else "")
        logger.debug(f"Agent {self.agent_id} prompt: {prompt_preview}")

        try:
            logger.info(
                f"Agent {self.agent_id} ({self.provider}/{self.model}) sending request"
            )
            api_start = time.time()
            raw_response = call_model(
                model_name=self.model,
                provider=self.provider,
                prompt=prompt,
                json_mode=json_mode,
                max_tokens=6400,
                timeout=timeout,
            )
            api_time = time.time() - api_start
            logger.info(
                f"Agent {self.agent_id} ({self.provider}/{self.model}) "
                f"received raw response in {api_time:.2f}s"
            )

        except ConnectionError as e:
            elapsed = time.time() - start_time
            error_msg = f"Failed to connect to {self.provider} service: {str(e)}"
            logger.error(
                f"Agent {self.agent_id} ({self.provider}/{self.model}) connection error "
                f"after {elapsed:.2f}s: {str(e)}"
            )
            raise LLMConnectionError(error_msg)
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Unexpected error with {self.provider} service: {str(e)}"
            logger.error(
                f"Agent {self.agent_id} ({self.provider}/{self.model}) unexpected error "
                f"after {elapsed:.2f}s: {str(e)}",
                exc_info=True,
            )
            raise LLMConnectionError(error_msg)

        # If it's already a dictionary, use it directly
        if isinstance(raw_response, dict):
            logger.debug(f"Agent {self.agent_id} response was already a dictionary")
            parsed_response = raw_response
        else:
            # Try to parse as JSON, but keep as string if parsing fails
            try:
                logger.debug(f"Agent {self.agent_id} attempting to parse JSON response")
                parsed_response = json.loads(raw_response)
                logger.debug(f"Agent {self.agent_id} successfully parsed JSON response")
            except (json.JSONDecodeError, TypeError):
                logger.debug(
                    f"Agent {self.agent_id} response is not valid JSON, using as string"
                )
                parsed_response = str(raw_response)

        response = {
            "agent_id": self.agent_id,
            "model": self.model,
            "response": parsed_response,
        }

        # Log response size and time
        if isinstance(parsed_response, str):
            response_length = len(parsed_response)
        else:
            response_length = len(json.dumps(parsed_response))

        total_time = time.time() - start_time
        logger.info(
            f"Agent {self.agent_id} ({self.provider}/{self.model}) completed in "
            f"{total_time:.2f}s with {response_length} chars"
        )

        return response
