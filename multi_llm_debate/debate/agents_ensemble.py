import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError, wait
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils.config_manager import get_models
from ..utils.logging_config import setup_logging
from ..utils.model_config import ModelConfig
from .agent import Agent, LLMConnectionError

# Use setup_logging to ensure consistent logging
logger = setup_logging(__name__)
logger.setLevel(logging.DEBUG)


class AgentsEnsemble:
    """A collection of LLM agents that can be used together.

    This class manages multiple Agent instances and provides methods to interact with them
    collectively. It can be initialized automatically from configuration or built manually.

    Attributes:
        agents (List[Agent]): List of Agent instances in the ensemble.
        concurrent (bool): Whether to use concurrent execution for responses.
        max_workers (int): Maximum number of concurrent workers when concurrent is True.
        job_delay (float): Delay in seconds between consecutive agent calls.
        timeout (float): Maximum time in seconds to wait for agent responses.
        max_retries (int): Maximum number of retry attempts for failed requests.
        retry_delay (float): Delay in seconds between retry attempts.
    """

    def __init__(
        self,
        config_list: Optional[List[ModelConfig]] = None,
        concurrent: bool = True,
        max_workers: Optional[int] = 4,
        job_delay: float = 0.5,
        timeout: float = 180.0,  # 3 minute timeout
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize an AgentsEnsemble instance.

        Args:
            config_list (Optional[List[ModelConfig]]): List of model configurations.
                If None, default configs will be loaded.
            concurrent (bool, optional): Whether to use concurrent execution. Defaults to True.
            max_workers (int, optional): Maximum number of concurrent workers. Defaults to 4.
            job_delay (float, optional): Delay in seconds between agent calls. Defaults to 0.5.
            timeout (float, optional): Maximum time in seconds to wait for agent responses.
                Defaults to 180.0 (3 minutes).
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 2.
            retry_delay (float, optional): Delay in seconds between retry attempts.
                Defaults to 1.0.

        Raises:
            ValueError: If initialization fails.
        """
        self.concurrent = concurrent
        self.max_workers = max_workers
        self.job_delay = job_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
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
        for provider, model_name, quantity in models:
            for _ in range(quantity):
                agent = Agent(agent_id=agent_id, model=model_name, provider=provider)
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
            for _ in range(config["quantity"]):
                agent = Agent(
                    agent_id=agent_id, model=config["name"], provider=config["provider"]
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
        self, agent: Agent, prompt: str, json_mode: bool
    ) -> Dict[str, Any]:
        """Attempt to get a response from an agent with retry logic.

        Args:
            agent (Agent): The agent to get a response from.
            prompt (str): The input prompt to send to the agent.
            json_mode (bool): Whether to expect JSON response.

        Returns:
            Dict[str, Any]: Response from the agent.

        Raises:
            LLMConnectionError: If all retry attempts fail.
        """
        errors = []
        logger.debug(
            f"Sending request to agent {agent.agent_id} ({agent.model}, {agent.provider})"
        )
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        f"Retry #{attempt} for agent {agent.agent_id} ({agent.provider})"
                    )

                response = agent.respond(
                    prompt, json_mode=json_mode, timeout=int(self.timeout)
                )
                elapsed = time.time() - start_time
                logger.info(
                    f"Agent {agent.agent_id} ({agent.provider}) responded in {elapsed:.2f}s"
                )
                return response

            except LLMConnectionError as e:
                errors.append(f"Attempt {attempt+1}: {str(e)}")
                logger.warning(
                    f"Agent {agent.agent_id} ({agent.model}, {agent.provider}) attempt {attempt+1} "
                    f"failed after {time.time() - start_time:.2f}s: {str(e)}"
                )
                if attempt < self.max_retries:
                    logger.info(
                        f"Waiting {self.retry_delay}s before retry #{attempt+2}"
                    )
                    time.sleep(self.retry_delay)

        total_time = time.time() - start_time
        error_msg = f"Failed after {self.max_retries + 1} attempts in {total_time:.2f}s: {'; '.join(errors)}"
        logger.error(
            f"Agent {agent.agent_id} ({agent.model}, {agent.provider}): {error_msg}"
        )
        raise LLMConnectionError(error_msg)

    def _get_response_concurrent(
        self, prompt: str, json_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """Get responses from all agents concurrently with improved resource management.

        Uses a controlled batch approach to prevent overwhelming Ollama and better
        manage resources, especially when running with local LLMs.

        Args:
            prompt (str): The input prompt to send to all agents.
            json_mode (bool, optional): Whether to expect JSON response. Defaults to False.

        Returns:
            List[Dict[str, Any]]: List of responses from all agents.

        Raises:
            LLMConnectionError: If no responses could be collected at all.
        """
        responses = []
        errors = []
        timeout_errors = []
        start_time = time.time()

        # Log the start of the process
        logger.info(f"Starting concurrent requests for {len(self.agents)} agents")

        # If max_workers is None or 1, use sequential processing
        if self.max_workers is None or self.max_workers <= 1:
            logger.info("Using sequential processing mode")
            for idx, agent in enumerate(self.agents):
                logger.info(
                    f"Processing agent {idx+1}/{len(self.agents)}: {agent.agent_id} ({agent.provider})"
                )
                try:
                    response = self._get_response_with_retry(agent, prompt, json_mode)
                    responses.append(response)
                except LLMConnectionError as e:
                    errors.append(str(e))

                if self.job_delay > 0:
                    logger.debug(f"Sleeping for {self.job_delay}s before next agent")
                    time.sleep(self.job_delay)

            logger.info(
                f"Sequential processing completed in {time.time() - start_time:.2f}s"
            )
        else:
            # Determine optimal batch size based on max_workers
            # For Ollama, smaller batches often work better
            batch_size = min(3, self.max_workers)

            # Calculate per-agent timeout with margin
            # Give each agent slightly less than the total timeout to allow for scheduling
            per_agent_timeout = max(30, self.timeout * 0.9)

            # Group agents by provider to optimize scheduling
            agents_by_provider = {}
            for agent in self.agents:
                provider = agent.provider.lower()
                if provider not in agents_by_provider:
                    agents_by_provider[provider] = []
                agents_by_provider[provider].append(agent)

            logger.info(
                f"Agent distribution by provider: {', '.join([f'{k}: {len(v)}' for k, v in agents_by_provider.items()])}"
            )

            # Process each provider's agents separately with appropriate batching
            provider_count = 0
            total_providers = len(agents_by_provider)

            for provider, provider_agents in agents_by_provider.items():
                provider_count += 1
                provider_start_time = time.time()

                # For Ollama, use smaller batches and add delays
                provider_batch_size = batch_size
                provider_delay = self.job_delay
                if provider == "ollama":
                    provider_batch_size = min(2, batch_size)
                    provider_delay = max(
                        1.0, self.job_delay
                    )  # Ensure at least 1 second delay for Ollama

                logger.info(
                    f"Processing provider {provider_count}/{total_providers}: "
                    f"{len(provider_agents)} {provider} agents with batch size {provider_batch_size}"
                )

                # Process this provider's agents in batches
                batch_count = 0
                total_batches = (
                    len(provider_agents) + provider_batch_size - 1
                ) // provider_batch_size

                for i in range(0, len(provider_agents), provider_batch_size):
                    batch_count += 1
                    batch_start_time = time.time()

                    batch = provider_agents[i : i + provider_batch_size]
                    logger.info(
                        f"Starting batch {batch_count}/{total_batches} with "
                        f"{len(batch)} {provider} agents"
                    )

                    batch_results = self._process_agent_batch(
                        batch, prompt, json_mode, per_agent_timeout, provider_delay
                    )

                    batch_responses, batch_errors, batch_timeouts = batch_results
                    responses.extend(batch_responses)
                    errors.extend(batch_errors)
                    timeout_errors.extend(batch_timeouts)

                    batch_time = time.time() - batch_start_time
                    logger.info(
                        f"Completed batch {batch_count}/{total_batches} in {batch_time:.2f}s "
                        f"with {len(batch_responses)}/{len(batch)} successful responses"
                    )

                    # Add a delay between batches to prevent overwhelming Ollama
                    if provider == "ollama" and i + provider_batch_size < len(
                        provider_agents
                    ):
                        delay_time = max(2.0, provider_delay * 2)
                        logger.info(
                            f"Adding {delay_time:.2f}s delay between Ollama batches"
                        )
                        time.sleep(delay_time)

                provider_time = time.time() - provider_start_time
                logger.info(f"Completed all {provider} agents in {provider_time:.2f}s")

        total_errors = len(errors) + len(timeout_errors)
        total_time = time.time() - start_time
        if responses:
            logger.info(
                f"Completed all requests in {total_time:.2f}s with "
                f"{len(responses)} responses and {total_errors} errors"
            )

        # Only raise an error if we got no responses at all
        if not responses:
            error_messages = []
            if errors:
                error_messages.append(f"Connection errors: {'; '.join(errors)}")
            if timeout_errors:
                error_messages.append(f"Timeout errors: {'; '.join(timeout_errors)}")
            raise LLMConnectionError("; ".join(error_messages))

        # Log warnings about partial errors if we got some responses
        if errors or timeout_errors:
            if errors:
                logger.warning(f"Some agents encountered errors: {'; '.join(errors)}")
            if timeout_errors:
                logger.warning(f"Some agents timed out: {'; '.join(timeout_errors)}")

        return responses

    def _process_agent_batch(
        self,
        agents: List[Agent],
        prompt: str,
        json_mode: bool,
        timeout_per_agent: float,
        delay_between_agents: float,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Process a batch of agents with controlled concurrency.

        Args:
            agents: List of agents to process
            prompt: The prompt to send to each agent
            json_mode: Whether to use JSON mode
            timeout_per_agent: Timeout for each individual agent
            delay_between_agents: Delay between submitting agent tasks

        Returns:
            Tuple containing (responses, errors, timeout_errors)
        """
        responses = []
        errors = []
        timeout_errors = []
        batch_start_time = time.time()

        logger.info(f"Starting batch processing of {len(agents)} agents")
        
        # Define a thread-safe flag to track executor shutdown status
        shutdown_flag = threading.Event()
        
        # Define a safer way to handle future results with proper timeout
        def get_future_result(future, timeout=0.5):
            """Get future result with strict timeout to avoid hanging."""
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                logger.warning(f"Timed out getting result from future")
                return None
            except Exception as e:
                logger.warning(f"Error getting future result: {str(e)}")
                return None

        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {}
            active_futures: Set = set()

            # Submit all jobs for this batch
            for agent in agents:
                if delay_between_agents > 0:
                    time.sleep(delay_between_agents)

                submission_time = time.time()
                # Use a wrapper function that enforces a timeout for each agent
                future = executor.submit(
                    self._get_response_with_retry, agent, prompt, json_mode
                )
                futures[future] = (agent, submission_time)
                active_futures.add(future)

                # Log submission
                logger.info(
                    f"Submitted request to {agent.provider} agent {agent.agent_id} ({agent.model})"
                )

            # Set the end time for our batch timeout
            batch_timeout = (
                timeout_per_agent + (len(agents) * delay_between_agents) + 10
            )  # Add margin
            end_time = time.time() + batch_timeout

            # Log the expected completion time
            logger.info(
                f"Batch timeout set to {batch_timeout:.2f}s, expected completion by "
                f"{time.strftime('%H:%M:%S', time.localtime(end_time))}"
            )

            # For heartbeat logging
            last_heartbeat = time.time()
            heartbeat_interval = 15  # Log waiting status every 15 seconds
            
            # Create a safety timer for hard termination
            safety_timeout = min(batch_timeout * 1.2, batch_timeout + 60)  # 20% more time or +60s
            safety_deadline = time.time() + safety_timeout
            
            logger.info(f"Safety timeout set to {safety_timeout:.2f}s to prevent hanging")

            # Process futures as they complete
            try:
                while active_futures and time.time() < end_time:
                    # Check for safety timeout - this is a hard limit to prevent complete hanging
                    if time.time() > safety_deadline:
                        logger.critical(
                            f"SAFETY TIMEOUT TRIGGERED after {time.time() - batch_start_time:.2f}s - "
                            f"forcing termination of {len(active_futures)} hanging tasks"
                        )
                        # Force cancellation of all remaining futures
                        for future in active_futures:
                            future.cancel()
                        break
                    
                    # Generate heartbeat log to show we're still waiting
                    current_time = time.time()
                    if current_time - last_heartbeat > heartbeat_interval:
                        waiting_time = current_time - batch_start_time
                        remaining_count = len(active_futures)
                        logger.info(
                            f"Still waiting for {remaining_count}/{len(agents)} responses after "
                            f"{waiting_time:.2f}s (timeout in {end_time - current_time:.1f}s)"
                        )

                        # List still-waiting agents
                        waiting_agents = [
                            f"{futures[f][0].agent_id}({futures[f][0].provider}, waiting {current_time - futures[f][1]:.1f}s)"
                            for f in active_futures
                        ]
                        logger.info(f"Waiting for agents: {', '.join(waiting_agents)}")
                        
                        # Check if any futures are done but not removed from active set
                        for future in list(active_futures):
                            if future.done():
                                logger.warning(
                                    f"Future for agent {futures[future][0].agent_id} is done "
                                    f"but not processed - forcing processing now"
                                )
                                active_futures.remove(future)
                                # Add to completed set for processing below
                                completed = {future}
                                break
                                
                        last_heartbeat = current_time

                    # Wait for the next future to complete with a short timeout
                    # Using a shorter timeout for more responsive handling
                    timeout_remaining = min(2.0, max(0.1, end_time - time.time()))
                    completed, active_futures = wait(
                        active_futures,
                        timeout=timeout_remaining,
                        return_when=FIRST_COMPLETED,
                    )

                    # Process completed futures immediately
                    if completed:
                        for future in completed:
                            agent, submit_time = futures[future]
                            completion_time = time.time()
                            response_time = completion_time - submit_time

                            try:
                                # Use our safer result getter with a strict timeout
                                response = get_future_result(future, timeout=1.0)
                                if response is not None:
                                    responses.append(response)
                                    logger.info(
                                        f"Received response from {agent.provider} agent {agent.agent_id} "
                                        f"in {response_time:.2f}s"
                                    )
                                else:
                                    timeout_errors.append(
                                        f"Agent {agent.agent_id} ({agent.model}) timed out during result retrieval"
                                    )
                                    logger.warning(
                                        f"Timeout retrieving result from agent {agent.agent_id} "
                                        f"after {response_time:.2f}s"
                                    )
                            except Exception as e:
                                error_msg = (
                                    f"Agent {agent.agent_id} ({agent.model}) error: {str(e)}"
                                )
                                errors.append(error_msg)
                                logger.warning(
                                    f"Connection error from agent {agent.agent_id} "
                                    f"after {response_time:.2f}s: {str(e)}"
                                )

            except Exception as e:
                logger.critical(f"Exception during batch processing: {str(e)}", exc_info=True)
                # Signal shutdown for a clean exit
                shutdown_flag.set()
                # Try to cancel any remaining futures
                for future in active_futures:
                    future.cancel()

            # For any remaining active futures, record them as timeouts
            if active_futures:
                current_time = time.time()
                logger.warning(
                    f"{len(active_futures)}/{len(agents)} agents still active after timeout - "
                    f"attempting to cancel and clean up"
                )
                
                # Collect diagnostic info for all hanging agents
                for future in active_futures:
                    agent, submit_time = futures[future]
                    wait_time = current_time - submit_time
                    
                    # Add detailed diagnostics to the error message
                    diagnostic_info = (
                        f"Agent {agent.agent_id} ({agent.model}, {agent.provider}) "
                        f"timed out after {wait_time:.2f}s - "
                        f"future state: done={future.done()}, cancelled={future.cancelled()}, "
                        f"running={future.running()}"
                    )
                    
                    timeout_errors.append(diagnostic_info)
                    logger.error(diagnostic_info)
                    
                    # Attempt to cancel the future
                    try:
                        cancel_result = future.cancel()
                        logger.info(f"Cancel attempt for agent {agent.agent_id}: {cancel_result}")
                    except Exception as e:
                        logger.error(f"Error cancelling future for agent {agent.agent_id}: {str(e)}")
                
                # Log a critical warning about the hanging threads
                logger.critical(
                    f"{len(active_futures)}/{len(agents)} agents timed out in batch "
                    f"after {current_time - batch_start_time:.2f}s - THIS MAY CAUSE HANGING THREADS"
                )

        # After exiting the executor context, log completion
        batch_time = time.time() - batch_start_time
        logger.info(
            f"Batch processing completed in {batch_time:.2f}s: "
            f"{len(responses)} successes, {len(errors)} errors, {len(timeout_errors)} timeouts"
        )
        
        # Final check for hanging threads - this is just diagnostic
        thread_count = threading.active_count()
        if thread_count > 10:  # Arbitrary threshold to detect potential issues
            logger.warning(
                f"High thread count detected: {thread_count} threads still active "
                f"after batch completion. This might indicate hanging threads."
            )
            
        return responses, errors, timeout_errors

    def get_responses(
        self, prompt: str, json_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """Get responses from all agents for a given prompt.

        Args:
            prompt (str): The input prompt to send to all agents.
            json_mode (bool, optional): Whether to expect JSON response. Defaults to False.

        Returns:
            List[Dict[str, Any]]: List of responses from all agents.

        Raises:
            LLMConnectionError: If any agent encounters a connection error.
        """
        logger.info(
            f"Getting responses from {len(self.agents)} agents (concurrent={self.concurrent})"
        )
        start_time = time.time()

        if self.concurrent:
            responses = self._get_response_concurrent(prompt, json_mode=json_mode)
        else:
            responses = []
            errors = []

            for i, agent in enumerate(self.agents):
                logger.info(
                    f"Requesting response from agent {i+1}/{len(self.agents)}: {agent.agent_id}"
                )
                try:
                    response = self._get_response_with_retry(agent, prompt, json_mode)
                    responses.append(response)
                except LLMConnectionError as e:
                    error_msg = f"Agent {agent.agent_id}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)

                if self.job_delay > 0:
                    logger.debug(f"Waiting {self.job_delay}s before next agent")
                    time.sleep(self.job_delay)

            if errors:
                error_msg = f"Connection errors occurred with {len(errors)}/{len(self.agents)} agents"
                logger.error(error_msg)
                if not responses:
                    raise LLMConnectionError(f"{error_msg}: {'; '.join(errors)}")

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
        return len(self.agents)

    def __str__(self) -> str:
        return f"AgentsEnsemble with {len(self)} agents"
