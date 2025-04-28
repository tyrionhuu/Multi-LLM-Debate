import asyncio
import concurrent.futures
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from openai import RateLimitError
from tqdm import tqdm

from ..llm.llm import call_model, call_model_batch
from ..utils.config_manager import get_models
from ..utils.model_config import ModelConfig

# Use setup_logging to ensure consistent logging
logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Simple data class to store agent information.

    Attributes:
        agent_id (int): Unique identifier for the agent.
        model (str): Name of the language model being used.
        base_url (Optional[str]): Base URL for the API calls.
        api_key (Optional[str]): API key for the agent.
    """

    agent_id: int
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"Agent {self.agent_id} ({self.model})"


class AgentsEnsemble:
    """A collection of LLM agents that can be used together.

    This class manages multiple agents and provides methods to interact with them
    collectively. It can be initialized automatically from configuration or built manually.

    Agent functionality is directly integrated into this class instead of using
    a separate Agent class, simplifying the architecture.

    Attributes:
        agents (List[Agent]): List of agent information in the ensemble.
        job_delay (float): Delay in seconds between consecutive agent calls.
        timeout (float): Maximum time in seconds to wait for agent responses.
        max_retries (int): Maximum number of retry attempts for failed requests.
    """

    def __init__(
        self,
        config_list: Optional[List[ModelConfig]] = None,
        job_delay: float = 0.5,
        timeout: float = 30.0,
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
        self.agents: List[Agent] = []  # List to hold agent information
        self._initialize_from_config(config_list)

    def _initialize_from_config(
        self, config_list: Optional[List[ModelConfig]] = None
    ) -> None:
        """Initialize agents from configuration.

        Loads model configurations and creates agents accordingly.
        Each agent is assigned a unique ID starting from 0.
        """
        agent_id = 0
        models = config_list if config_list is not None else get_models()
        for model_info in models:
            # Handle dict or tuple formats
            if isinstance(model_info, dict):
                model_name = model_info.get("name", "")
                base_url = model_info.get("base_url")
                quantity = model_info.get("quantity", 1)
                api_key = model_info.get("api_key")
            elif isinstance(model_info, tuple) and len(model_info) >= 3:
                model_name, base_url, quantity = model_info[:3]
                api_key = None
            else:
                logger.warning(f"Skipping unrecognized model info format: {model_info}")
                continue

            for _ in range(quantity):
                agent = Agent(
                    agent_id=agent_id,
                    model=model_name,
                    base_url=base_url,
                    api_key=api_key,
                )
                self._add_agent(agent)
                agent_id += 1

    def _add_agent(self, agent: Agent) -> None:
        """Add an agent to the ensemble.

        Args:
            agent (Agent): The agent to add to the ensemble.
        """
        self.agents.append(agent)
        logger.debug(
            f"Added Agent {agent.agent_id} with model {agent.model} "
            f"(base_url: {'custom' if agent.base_url else 'default'}, "
            f"api_key: {'set' if agent.api_key else 'not set'})"
        )

    def _count_unique_models(self) -> int:
        """Return the number of unique models among agents."""
        return len({agent.model for agent in self.agents})

    def _group_agents_by_model(self) -> Dict[str, List[Agent]]:
        """Group agents by their model name.

        Returns:
            Dict[str, List[Agent]]: Mapping from model name to list of agents.
        """
        model_groups: Dict[str, List[Agent]] = {}
        for agent in self.agents:
            model_groups.setdefault(agent.model, []).append(agent)
        return model_groups

    def _parse_response(self, agent: Agent, raw_response: Any) -> Dict[str, Any]:
        """Process the raw response from the LLM API.

        Args:
            agent (Agent): Information about the agent.
            raw_response: The raw response from the API.

        Returns:
            Dict[str, Any]: Processed response with agent information.
        """
        # If it's already a dictionary, use it directly
        if isinstance(raw_response, Dict):
            logger.debug(f"Agent {agent.agent_id} response was already a dictionary")
            parsed_response = raw_response
        else:
            # Try to parse as JSON, but keep as string if parsing fails
            try:
                logger.debug(
                    f"Agent {agent.agent_id} attempting to parse JSON response"
                )
                parsed_response = json.loads(raw_response)
                logger.debug(
                    f"Agent {agent.agent_id} successfully parsed JSON response"
                )
            except (json.JSONDecodeError, TypeError):
                logger.debug(
                    f"Agent {agent.agent_id} response is not valid JSON, using as string"
                )
                parsed_response = str(raw_response)

        response = {
            "agent_id": agent.agent_id,
            "model": agent.model,
            "response": parsed_response,
        }

        return response

    def _respond_with_same_model(
        self,
        agents: Union[Agent, List[Agent]],
        prompt: str,
        images: Union[str, Path, List[str], List[Path], None] = None,
        json_mode: bool = False,
        timeout: Optional[int] = None,
        max_retries: int = 3,
        max_tokens: int = 6400,
        temperature: float = 1.0,
        batch: bool = False,
        batch_size: int = 11,
    ) -> List[Dict[str, Any]]:
        """Call the LLM API for a given prompt using the specified agents.

        Args:
            agents (List[Agent]): List of agents to use for the request.
            prompt (str): The input prompt for the LLM.
            images (Union[str, Path, List[str], List[Path], None]): Optional images to include in the request.
            json_mode (bool): Whether to return the response in JSON format.
            timeout (Optional[int]): Maximum time to wait for a response.
            max_retries (int): Number of retry attempts for failed requests.
            max_tokens (int): Maximum number of tokens in the response.
            temperature (float): Temperature setting for the model.
            batch (bool): Whether to process requests in batch mode.
            batch_size (int): Size of the batch for processing.

        Returns:
            List[Dict[str, Any]]: List of responses from each agent.
        """
        if not agents:
            raise ValueError("No agents available for processing.")
        if isinstance(agents, Agent):
            agents = [agents]
        if images is not None:
            if not isinstance(images, list):
                images = [images]

            # Validate all images
            for img in images:
                if isinstance(img, (str, Path)):
                    img_path = Path(img)
                    if not img_path.exists():
                        raise ValueError(f"Image file {img_path} does not exist.")
                else:
                    raise ValueError(
                        f"Invalid image type: {type(img)}. Expected str or Path."
                    )
        start_time = time.time()
        logger.debug(f"Starting response generation with {len(agents)} agents")
        logger.debug(f"Prompt: {prompt[:50]}...")  # Log first 50 chars of prompt
        logger.debug(f"Images: {images if images else 'None'}")
        logger.debug(f"Batch mode: {batch}, Batch size: {batch_size}")
        model_name = agents[0].model
        base_url = agents[0].base_url
        api_key = agents[0].api_key
        if not batch:
            # Process each agent individually
            responses = []
            errors = []
            for agent in agents:
                try:
                    agent_time = time.time()
                    logger.info(f"Calling model for Agent {agent.agent_id}")
                    raw_response = call_model(
                        model_name=model_name,
                        prompt=prompt,
                        images=images,
                        json_mode=json_mode,
                        base_url=base_url,
                        api_key=api_key,
                        timeout=timeout,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        retries=max_retries,  # pass as retries
                    )
                    logger.info(
                        f"Agent {agent.agent_id} response received in {time.time() - agent_time:.2f}s"
                    )
                    response = self._parse_response(agent, raw_response)
                    responses.append(response)
                except Exception as e:
                    error_message = f"Max retries {max_retries} exceeded for Agent {agent.agent_id}: {str(e)}"
                    logger.error(error_message, exc_info=True)
                    errors.append({"agent_id": agent.agent_id, "error": error_message})

            elapsed = time.time() - start_time
            logger.info(
                f"Response generation completed in {elapsed:.2f}s for {len(agents)} agents"
            )
            if errors:
                logger.warning(f"Errors encountered: {len(errors)}")
                for error in errors:
                    logger.error(f"Agent {error['agent_id']} error: {error['error']}")
            return responses
        else:
            # Process agents in batches
            responses = []
            errors = []
            num_batches = (len(agents) + batch_size - 1) // batch_size
            logger.info(f"Processing {num_batches} batches of size {batch_size}")

            for i in tqdm(range(num_batches), desc="Processing batches"):
                batch_agents = agents[i * batch_size : (i + 1) * batch_size]
                try:
                    agent_time = time.time()
                    logger.info(f"Calling model for batch {i+1}/{num_batches}")
                    raw_responses = asyncio.run(
                        call_model_batch(
                            model_name=model_name,
                            base_url=base_url,
                            api_key=api_key,
                            prompts=[prompt] * len(batch_agents),
                            images=[images] * len(batch_agents) if images else None,
                            json_mode=json_mode,
                            timeout=timeout,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            batch_size=batch_size,
                            retries=max_retries,  # pass as retries
                        )
                    )
                    logger.info(
                        f"Batch {i+1} response received in {time.time() - agent_time:.2f}s"
                    )
                    for j, agent in enumerate(batch_agents):
                        response = self._parse_response(agent, raw_responses[j])
                        responses.append(response)
                except Exception as e:
                    error_message = (
                        f"Max retries {max_retries} exceeded for batch {i+1}: {str(e)}"
                    )
                    logger.error(error_message, exc_info=True)
                    errors.append({"batch": i + 1, "error": error_message})

            elapsed = time.time() - start_time
            logger.info(
                f"Batch processing completed in {elapsed:.2f}s for {len(agents)} agents"
            )
            if errors:
                logger.warning(f"Errors encountered: {len(errors)}")
                for error in errors:
                    logger.error(f"Batch {error['batch']} error: {error['error']}")
            return responses

    def get_responses(
        self,
        prompt: str,
        images: Union[str, Path, List[str], List[Path], None] = None,
        json_mode: bool = False,
        timeout: Optional[int] = None,
        max_retries: int = 3,
        max_tokens: int = 6400,
        temperature: float = 1.0,
        batch: bool = False,
        batch_size: int = 11,
    ) -> List[Dict[str, Any]]:
        """Get responses from all agents for a given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            images (Union[str, Path, List[str], List[Path], None]): Optional images to include in the request.
            json_mode (bool): Whether to return the response in JSON format.
            timeout (Optional[int]): Maximum time to wait for a response.
            max_retries (int): Number of retry attempts for failed requests.
            max_tokens (int): Maximum number of tokens in the response.
            temperature (float): Temperature setting for the model.
            batch (bool): Whether to process requests in batch mode.
            batch_size (int): Size of the batch for processing.

        Returns:
            List[Dict[str, Any]]: List of responses from each agent.
        """
        groups = self._group_agents_by_model()
        all_responses = []
        if len(groups) == 1:
            # Only one model, use the same method for all agents
            model_name = next(iter(groups))
            agents = groups[model_name]
            responses = self._respond_with_same_model(
                agents=agents,
                prompt=prompt,
                images=images,
                json_mode=json_mode,
                timeout=timeout,
                max_retries=max_retries,
                max_tokens=max_tokens,
                temperature=temperature,
                batch=batch,
                batch_size=batch_size,
            )
            # Map agent_id to response for lookup
            response_map = {resp["agent_id"]: resp for resp in responses}
            for agent in agents:
                if agent.agent_id in response_map:
                    all_responses.append(response_map[agent.agent_id])
                else:
                    all_responses.append(
                        {
                            "agent_id": agent.agent_id,
                            "model": agent.model,
                            "response": "Error: Missing response",
                            "error": "Response not generated or collected",
                        }
                    )
        else:
            # Multiple models, process each group parallelly
            logger.info(f"Processing {len(groups)} unique models in parallel")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        self._respond_with_same_model,
                        agents=agents,
                        prompt=prompt,
                        images=images,
                        json_mode=json_mode,
                        timeout=timeout,
                        max_retries=max_retries,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        batch=batch,
                        batch_size=batch_size,
                    ): (model_name, agents)
                    for model_name, agents in groups.items()
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    desc="Processing models",
                    total=len(futures),
                ):
                    model_name, agents = futures[future]
                    try:
                        responses = future.result()
                        response_map = {resp["agent_id"]: resp for resp in responses}
                        for agent in agents:
                            if agent.agent_id in response_map:
                                all_responses.append(response_map[agent.agent_id])
                            else:
                                all_responses.append(
                                    {
                                        "agent_id": agent.agent_id,
                                        "model": agent.model,
                                        "response": "Error: Missing response",
                                        "error": "Response not generated or collected",
                                    }
                                )
                    except Exception as e:
                        logger.error(
                            f"Error processing model {model_name}: {str(e)}",
                            exc_info=True,
                        )
                        for agent in agents:
                            all_responses.append(
                                {
                                    "agent_id": agent.agent_id,
                                    "model": agent.model,
                                    "response": "Error: Missing response",
                                    "error": "Response not generated or collected",
                                }
                            )
        logger.info(
            f"Total responses collected: {len(all_responses)} from {len(self.agents)} agents"
        )
        return all_responses

    def __str__(self) -> str:
        """Return a string representation of the AgentsEnsemble."""
        unique_models = self._count_unique_models()
        return (
            f"AgentsEnsemble with {len(self.agents)} agents "
            f"across {unique_models} unique models."
        )

    def __repr__(self) -> str:
        """Return a detailed string representation of the AgentsEnsemble."""
        return (
            f"AgentsEnsemble(agents={self.agents}, "
            f"job_delay={self.job_delay}, timeout={self.timeout}, "
            f"max_retries={self.max_retries})"
        )
