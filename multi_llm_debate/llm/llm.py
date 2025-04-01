import base64
import io
import json
import logging
import os
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Literal, Optional, Union

import ollama
import requests
import requests.exceptions
from ollama import Options
from openai import OpenAI
from PIL import Image
from requests.exceptions import ConnectionError, Timeout
from vllm import LLM, SamplingParams

from ..utils.config_manager import get_api_key, get_base_url, get_vllm_model_path
from ..utils.logging_config import setup_logging
from .utils import AbortableVLLMInference, ThreadSafeTimeout

# Set up logger
logger = setup_logging(__name__)

KEY = get_api_key()
BASE_URL = get_base_url()
VLLM_MODEL_PATH = get_vllm_model_path()

# Global vLLM model cache to avoid reloading
_vllm_models = {}

if KEY.strip() == "":
    KEY = input("Please enter your API key: ")
    from ..utils.config_manager import save_api_key

    save_api_key(KEY)


logging.getLogger("httpx").setLevel(logging.WARNING)


def encode_image(image_path: str) -> str:
    """Encodes an image file to a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded


def call_model(
    model_name: str = "llama3.2:11b",
    provider: Literal["api", "ollama", "openai", "anthropic", "vllm"] = "ollama",
    prompt: str = "",
    temperature: float = 1.0,
    max_tokens: int = 3200,
    json_mode: bool = True,
    timeout: Optional[int] = 30,
    vision: bool = False,
    images: Union[
        str, List[str], bytes, List[bytes], Image.Image, List[Image.Image], None
    ] = None,
) -> str:
    """Routes the call to the appropriate model provider and returns the response.

    Can handle both text-only and vision models based on the vision parameter.

    Args:
        model_name (str): The name of the model to use.
        provider (Literal): The provider of the model.
        prompt (str): The text prompt for the model.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the response.
        json_mode (bool): Whether the response should be in JSON format.
        timeout (Optional[int]): Timeout in seconds for the request. Defaults to 30.
        vision (bool): Whether to use vision models.
        images (Union[str, List[str], bytes, List[bytes], Image.Image, List[Image.Image], None]):
            Image inputs when using vision models.

    Returns:
        str: The generated response from the model.

    Raises:
        ConnectionError: If there's a timeout or connection issue
        ValueError: If the provider is not supported
    """
    start_time = time.time()
    logger.info(
        f"Calling {provider}/{model_name} (timeout={timeout}s, json={json_mode})"
    )

    with ThreadSafeTimeout(timeout, f"{provider}/{model_name} call"):
        try:
            # vLLM doesn't support vision features yet
            if vision and provider != "vllm":
                return call_vision_model(
                    model_name=model_name,
                    provider=provider,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    images=images,
                    json_mode=json_mode,
                    timeout=timeout,
                )
            elif vision and provider == "vllm":
                logger.warning(
                    "vLLM provider does not support vision models, falling back to text-only"
                )

            if provider == "ollama":
                result = generate_with_ollama(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    timeout=timeout,
                )
            elif provider == "api":
                result = generate_with_api(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    timeout=timeout,
                )
            elif provider == "vllm":
                result = generate_with_vllm(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    timeout=timeout,
                )
            elif provider == "openai":
                raise NotImplementedError(
                    "OpenAI API integration is not implemented yet."
                )
            elif provider == "anthropic":
                raise NotImplementedError(
                    "Anthropic API integration is not implemented yet."
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            elapsed = time.time() - start_time
            logger.info(f"Call to {provider}/{model_name} completed in {elapsed:.2f}s")
            return result

        except (ConnectionError, Timeout, FutureTimeoutError) as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Timeout or connection error calling {provider}/{model_name} "
                f"after {elapsed:.2f}s: {str(e)}"
            )
            raise ConnectionError(f"Error with {provider} service: {str(e)}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Error calling {provider}/{model_name} after {elapsed:.2f}s: {str(e)}",
                exc_info=False,
            )
            raise ConnectionError(f"Error with {provider} service: {str(e)}")


def get_or_create_vllm_model(model_name: str) -> LLM:
    """Get or create a vLLM model instance.

    This function caches vLLM models to avoid reloading them for each request.

    Args:
        model_name: The name or path of the model to load

    Returns:
        LLM: A vLLM model instance

    Raises:
        ValueError: If the model cannot be loaded
    """
    global _vllm_models

    if model_name in _vllm_models:
        logger.debug(f"Using cached vLLM model: {model_name}")
        return _vllm_models[model_name]

    # Determine the actual model path
    model_path = VLLM_MODEL_PATH.get(model_name, model_name)

    logger.info(f"Loading vLLM model: {model_path}")
    try:
        # Load the model with vLLM
        model = LLM(model=model_path)
        _vllm_models[model_name] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load vLLM model {model_path}: {str(e)}")
        raise ValueError(f"Failed to load vLLM model: {str(e)}")


def generate_with_vllm(
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool = False,
    timeout: int = 30,
) -> str:
    """Generates a response using the vLLM library.

    This function uses vLLM's Python API directly instead of HTTP requests.

    Args:
        model_name (str): The name or path of the model to use
        prompt (str): The text prompt for the model
        temperature (float): Sampling temperature for the model
        max_tokens (int): Maximum number of tokens in the response
        json_mode (bool): Whether the response should be in JSON format
        timeout (int): Maximum time to wait for the response

    Returns:
        str: The generated response from the model

    Raises:
        TimeoutError: If the inference times out
        ValueError: If there's an issue with the model or parameters
        ImportError: If vLLM is not installed
    """
    logger.info(f"Generating with vLLM library using model: {model_name}")

    try:
        # Enhance the prompt for JSON mode if needed
        if json_mode:
            prompt = "You must respond with valid JSON. " + prompt

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None,  # Can be customized if needed
        )

        # Create an abortable inference wrapper
        inference = AbortableVLLMInference()

        # Define the inference function
        def run_inference():
            # Get or create the model
            model = get_or_create_vllm_model(model_name)

            # Run inference
            outputs = model.generate(prompt, sampling_params)
            return outputs[0].outputs[0].text

        # Run with timeout
        response_text = inference.run_with_timeout(func=run_inference, timeout=timeout)

        # Process JSON if needed
        if json_mode:
            try:
                # Ensure it's valid JSON
                parsed_json = json.loads(response_text)
                return json.dumps(parsed_json)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"vLLM returned invalid JSON despite json_mode=True: {e}"
                )
                return response_text

        return response_text

    except TimeoutError:
        logger.error(f"vLLM inference timed out after {timeout}s")
        raise TimeoutError(f"Inference timed out after {timeout} seconds")
    except ValueError as e:
        if "aborted" in str(e).lower():
            logger.error("vLLM inference was aborted")
            raise TimeoutError("Inference was aborted due to timeout")
        logger.error(f"ValueError in vLLM inference: {str(e)}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in generate_with_vllm: {str(e)}", exc_info=False
        )
        raise ValueError(f"vLLM error: {str(e)}")


def call_vision_model(
    model_name: str = "llama3.2-vision:11b",
    provider: Literal["api", "ollama", "openai", "anthropic"] = "ollama",
    prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 3200,
    images: Union[
        str, List[str], bytes, List[bytes], Image.Image, List[Image.Image], None
    ] = None,
    json_mode: bool = False,
    timeout: Optional[int] = 30,
) -> str:
    """
    Routes the call to the appropriate vision model provider and returns the response.

    Args:
        model_name (str): The name of the model to use.
        provider (Literal): The provider of the vision model.
        prompt (str): The text prompt for the model.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the response.
        images (Union[str, List[str], bytes, List[bytes], Image.Image, List[Image.Image], None]):
            Image file paths, bytes, PIL Images, or lists of any of these. If None, runs in text-only mode.
        json_mode (bool): Whether the response should be in JSON format.
        timeout (Optional[int]): Timeout in seconds for the request. Defaults to 30.

    Returns:
        str: The generated response from the vision model.
    """
    processed_images = []

    if images is not None:
        # Convert single items to list
        if not isinstance(images, list):
            images = [images]

        # Validate and process all images
        for img in images:
            if isinstance(img, str):
                if not os.path.exists(img):
                    raise ValueError(f"Image file not found: {img}")
                processed_images.append(img)
            elif isinstance(img, bytes):
                processed_images.append(img)
            elif isinstance(img, Image.Image):
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format or "PNG")
                processed_images.append(img_byte_arr.getvalue())
            else:
                raise ValueError(
                    f"Invalid image type: {type(img)}. Expected str, bytes, or PIL Image."
                )

    if provider == "ollama":
        return generate_with_ollama(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            images=processed_images,
            json_mode=json_mode,
            timeout=timeout,
        )
    elif provider == "api":
        return generate_with_api(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            images=processed_images,
            json_mode=json_mode,
            timeout=timeout,
        )
    elif provider == "openai":
        raise NotImplementedError("OpenAI API integration is not implemented yet.")
    elif provider == "anthropic":
        raise NotImplementedError("Anthropic API integration is not implemented yet.")
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def retry_json_generation(
    model_name: str,
    prompt: str,
    options: Options,
    max_retries: int = 3,
    images: Optional[List[str | bytes]] = None,
) -> str:
    """
    Retries JSON generation when parsing fails.

    Args:
        model_name (str): The name of the model to use.
        prompt (str): The text prompt for the model.
        options (Options): Ollama options object.
        max_retries (int): Maximum number of retry attempts.
        images (Optional[List[str | bytes]]): Optional images for vision models.

    Returns:
        str: Valid JSON string response.

    Raises:
        ValueError: If unable to get valid JSON after max retries.
    """
    kwargs = {
        "model": model_name,
        "prompt": "You must respond with valid JSON. " + prompt,
        "options": options,
        "format": "json",
    }
    if images:
        kwargs["images"] = images

    for attempt in range(max_retries):
        try:
            response_str = ollama.generate(**kwargs)["response"]
            return json.dumps(json.loads(response_str))
        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                raise ValueError(f"Invalid JSON response after {max_retries} attempts")
            continue


def generate_with_ollama(
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    images: Optional[List[str | bytes]] = None,
    json_mode: bool = False,
    timeout: int = 60,
) -> str:
    """Generates a response using the Ollama model with optional images.

    Args:
        model_name (str): The name of the model to use.
        prompt (str): The text prompt for the model.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the response.
        images (Optional[List[str | bytes]]): Paths to image files or image data.
            If None or empty list, runs in text-only mode.
        json_mode (bool): Whether the response should be in JSON format.
        timeout (int): Maximum time to wait for the response.

    Returns:
        str: The generated response from the model.

    Raises:
        TimeoutError: If the request times out
        ConnectionError: If there's an issue connecting to Ollama
        ValueError: If other validation fails
    """
    max_eof_retries = 3
    for eof_retry in range(max_eof_retries):
        try:
            logger.info(
                f"Sending request to Ollama model {model_name} (attempt "
                f"{eof_retry+1}/{max_eof_retries})"
            )

            options = Options(
                temperature=temperature,
                num_ctx=max_tokens,
                num_predict=max_tokens,
                timeout=timeout,  # Set timeout in Ollama options
            )

            kwargs = {
                "model": model_name,
                "options": options,
            }

            if images:
                kwargs["images"] = images

            if json_mode:
                kwargs["prompt"] = "You must respond with valid JSON. " + prompt
                kwargs["format"] = "json"

                try:
                    result = ollama.generate(**kwargs)
                    # Validate JSON response
                    return json.dumps(json.loads(result["response"]))
                except json.JSONDecodeError:
                    # Retry with explicit JSON formatting if parsing fails
                    return retry_json_generation(
                        model_name, prompt, options, images=images
                    )
            else:
                kwargs["prompt"] = prompt
                result = ollama.generate(**kwargs)
                return result["response"]

        except requests.exceptions.Timeout:
            logger.error(f"Request to {model_name} timed out after {timeout}s")
            raise TimeoutError(f"Request timed out after {timeout} seconds")
        except ConnectionError as e:
            logger.error(f"Connection error with Ollama: {str(e)}")
            raise ConnectionError(
                "Failed to connect to Ollama server. Please check if Ollama is running."
            )
        except Exception as e:
            error_msg = str(e)
            if "EOF" in error_msg and eof_retry < max_eof_retries - 1:
                logging.warning(
                    f"Ollama EOF error encountered, retrying "
                    f"({eof_retry+1}/{max_eof_retries}): {error_msg}"
                )
                continue
            elif "EOF" in error_msg:
                logging.error(f"Persistent EOF error with Ollama: {error_msg}")
                raise ValueError(
                    "Ollama server disconnected unexpectedly (EOF error). "
                    "This usually happens when the Ollama server crashes or restarts. "
                    "Please check if Ollama is running properly, restart it if needed, "
                    "and try again."
                )
            else:
                logging.error(f"Error in generate_with_ollama: {error_msg}")
                raise


def generate_with_api(
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    images: Optional[List[str | bytes]] = None,
    json_mode: bool = False,
    timeout: int = 30,  # Default 30 seconds
) -> str:
    """
    Generates a response using the API with optional images.

    Args:
        model_name (str): The name of the model to use.
        prompt (str): The text prompt for the model.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the response.
        images (Optional[list[str | bytes]]): Paths to image files or image data.
            If None, runs in text-only mode.
        json_mode (bool): Whether the response should be in JSON format.
        timeout (int): Maximum time in seconds to wait for the response.

    Returns:
        str: The generated response from the API.
    """
    try:
        # Initialize OpenAI client with timeout
        client = OpenAI(
            base_url=BASE_URL,
            api_key=KEY,
            timeout=timeout,  # Set the timeout for the client
        )
        messages = generate_api_messages(images=images, prompt=prompt)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"} if json_mode else None,
                seed=42,
            )
            response_str = response.choices[0].message.content

            if json_mode:
                try:
                    return json.dumps(json.loads(response_str))
                except json.JSONDecodeError:
                    return response_str
            return response_str

        except requests.exceptions.Timeout:
            raise TimeoutError(f"API request timed out after {timeout} seconds")

    except ConnectionError:
        raise ConnectionError(
            "Failed to connect to API server. Please check your internet connection and API endpoint."
        )
    except Exception as e:
        logging.error(f"Error in generate_with_api: {str(e)}")
        raise


def generate_api_messages(
    prompt: str,
    images: Optional[List[str | bytes]] = None,
) -> List[Dict[str, Any]]:
    """
    Prepares the messages payload for the API call with optional images and a prompt.

    Args:
        prompt (str): The text prompt for the model.
        images (Optional[list[str | bytes]]): List of image file paths or bytes objects.
            If None, returns text-only message format.

    Returns:
        list[dict]: A list of messages formatted for the API call.
    """
    if not images:
        return [{"role": "user", "content": prompt}]

    if len(images) == 1:
        base64_image = (
            encode_image(images[0])
            if isinstance(images[0], str)
            else base64.b64encode(images[0]).decode("utf-8")
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]
    else:
        base64_images = [
            (
                encode_image(img)
                if isinstance(img, str)
                else base64.b64encode(img).decode("utf-8")
            )
            for img in images
        ]
        content = [
            {
                "type": "text",
                "text": prompt,
            }
        ]
        content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
                for base64_image in base64_images
            ]
        )
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
    return messages


def main():
    # You can set multiple GPUs with comma-separated indices, e.g., "0,1,2"

    # Example usage of the generate_with_api function
    question = "Is the sky blue?"
    prompt = f"{question} Please provide a detailed explanation."
    model_name = "/data/share_weight/Meta-Llama-3-8B"
    provider = "vllm"
    result = call_model(
        model_name=model_name,
        provider=provider,
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        json_mode=True,
        timeout=180,
    )
    print("Generated response:", result)


if __name__ == "__main__":
    import os

    # Set visible GPU devices for vLLM
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
    # This will run the main function to demonstrate the call_model function.
    # You can replace the parameters with actual values as per your requirements.
