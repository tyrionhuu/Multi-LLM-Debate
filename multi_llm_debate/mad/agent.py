import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import backoff

from ..llm.llm import call_model


class Agent:
    def __init__(
        self,
        model_name: str,
        name: str,
        temperature: float,
        provider: str = "ollama",
        sleep_time: float = 0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        images: Optional[
            Union[str, Path, bytes, List[str], List[Path], List[bytes]]
        ] = None,  # Add images parameter
        verbose: bool = False,  # Add verbose parameter
    ) -> None:
        """Create an agent

        Args:
            model_name (str): model name
            name (str): name of this agent
            temperature (float): higher values make the output more random, while
                lower values make it more focused and deterministic
            provider (str): name of the model provider (e.g., 'ollama', 'openai')
            sleep_time (float): sleep because of rate limits
            base_url (Optional[str]): Base URL for the API calls
            api_key (Optional[str]): API key for the agent
        """
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.provider = provider.lower()
        self.memory_lst = []
        self.sleep_time = sleep_time
        self.base_url = base_url
        self.api_key = api_key
        self.images = images  # Store images for vision models
        self.verbose = verbose  # Store verbose setting

    @backoff.on_exception(backoff.expo, Exception, max_tries=20)
    def query(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: Optional[float] = None,
        json_mode: bool = True,
    ) -> str:
        """Make a query to the language model.

        Args:
            messages (List[Dict[str, str]]): Chat history in message format.
            max_tokens (int): Maximum tokens in API call.
            temperature (Optional[float]): Sampling temperature, uses default if None.
            json_mode (bool): Whether to expect JSON response.

        Raises:
            Exception: For any exceptions raised by the model call.

        Returns:
            str: The response from the model.
        """
        time.sleep(self.sleep_time)

        try:
            temp = temperature if temperature is not None else self.temperature

            # Convert messages to a single prompt string for the call_model function
            prompt = self._messages_to_prompt(messages)

            response = call_model(
                model_name=self.model_name,
                base_url=self.base_url or "",
                prompt=prompt,
                json_mode=json_mode,
                temperature=temp,
                max_tokens=max_tokens,
                api_key=self.api_key,
                images=self.images,  # Pass images to call_model
            )

            # Handle different response formats
            if isinstance(response, dict) and "content" in response:
                return response["content"]  # type: ignore
            else:
                try:
                    parsed_response = json.loads(response)
                    return parsed_response
                except json.JSONDecodeError:
                    return response

        except Exception as e:
            raise Exception(f"Failed to query {self.provider} service: {str(e)}")

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert a list of messages to a single prompt string.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries

        Returns:
            str: Combined prompt string
        """
        prompt_parts = []
        for message in messages:
            role = message.get("role", "")  # type: ignore
            content = message.get("content", "")  # type: ignore

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n\n".join(prompt_parts)

    def set_meta_prompt(self, meta_prompt: str) -> None:
        """Set the meta_prompt

        Args:
            meta_prompt (str): the meta prompt
        """
        # Clean the meta prompt for storage (remove image data)
        clean_meta_prompt = self._clean_memory_for_verbose(meta_prompt)
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

        # If verbose mode is enabled, print the cleaned version
        if hasattr(self, "verbose") and self.verbose:
            print(f"----- {self.name} Meta Prompt -----\n{clean_meta_prompt}\n")

    def add_event(self, event: str) -> None:
        """Add an new event in the memory

        Args:
            event (str): string that describe the event.
        """
        self.memory_lst.append({"role": "user", "content": f"{event}"})

        # If verbose mode is enabled, print the cleaned version
        if hasattr(self, "verbose") and self.verbose:
            clean_event = self._clean_memory_for_verbose(event)
            print(f"----- {self.name} Event -----\n{clean_event}\n")

    def add_memory(self, memory: str, verbose: bool = False) -> None:
        """Monologue in the memory

        Args:
            memory (str): string that generated by the model in the last round.
            verbose (bool): whether to print the memory
        """
        self.memory_lst.append({"role": "assistant", "content": f"{memory}"})
        if verbose:
            # Filter out image data from verbose output
            clean_memory = self._clean_memory_for_verbose(memory)
            print(f"----- {self.name} -----\n{clean_memory}\n")

    def _clean_memory_for_verbose(self, memory: str) -> str:
        """Clean memory content for verbose output by removing image data.

        Args:
            memory (str): The memory content to clean

        Returns:
            str: Cleaned memory content without image data
        """
        if not memory:
            return memory

        # Remove base64 image data (common patterns)
        import re

        # Remove base64 image data (most specific pattern first)
        # Pattern for data:image/...;base64, followed by base64 characters
        memory = re.sub(
            r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE_DATA]", memory
        )

        # Remove image: prefix followed by data (before long base64 pattern)
        memory = re.sub(r"Image:\s*[A-Za-z0-9+/=]+", "Image: [IMAGE_DATA]", memory)

        # Remove long base64 strings (likely image data) - less specific, do last
        memory = re.sub(r"[A-Za-z0-9+/]{100,}={0,2}", "[LONG_BASE64_DATA]", memory)

        # Remove image file paths that might be embedded
        memory = re.sub(
            r"/path/to/.*\.(jpg|jpeg|png|gif|bmp|webp)", "[IMAGE_PATH]", memory
        )

        return memory

    def ask(self, temperature: Optional[float] = None, json_mode: bool = True) -> str:
        """Query for answer based on memory

        Args:
            temperature (Optional[float]): Override default temperature if provided
            json_mode (bool): Whether to expect JSON response

        Returns:
            str: The model's response
        """

        # Make the query
        return self.query(
            self.memory_lst,
            max_tokens=3200,
            temperature=temperature,
            json_mode=json_mode,
        )
