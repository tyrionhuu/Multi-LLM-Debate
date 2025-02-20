import base64
import io
import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

import ollama
import requests.exceptions
from ollama import Options
from openai import OpenAI
from PIL import Image
from requests.exceptions import ConnectionError

from ..utils.config_manager import get_api_key, get_base_url

KEY = get_api_key()
BASE_URL = get_base_url()

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
    provider: Literal["api", "ollama", "openai", "anthropic"] = "ollama",
    prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 3200,
    json_mode: bool = True,
    timeout: Optional[int] = None,
    vision: bool = False,
    images: Union[
        str, List[str], bytes, List[bytes], Image.Image, List[Image.Image], None
    ] = None,
) -> str:
    """
    Routes the call to the appropriate model provider and returns the response.
    Can handle both text-only and vision models based on the vision parameter.

    Args:
        model_name (str): The name of the model to use.
        provider (Literal): The provider of the model.
        prompt (str): The text prompt for the model.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the response.
        json_mode (bool): Whether the response should be in JSON format.
        timeout (Optional[int]): Timeout for the HTTP request.
        vision (bool): Whether to use vision models.
        images (Union[str, List[str], bytes, List[bytes], Image.Image, List[Image.Image], None]):
            Image inputs when using vision models.

    Returns:
        str: The generated response from the model.
    """
    if vision:
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

    if provider == "ollama":
        return generate_with_ollama(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
        )
    elif provider == "api":
        return generate_with_api(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
        )
    elif provider == "openai":
        raise NotImplementedError("OpenAI API integration is not implemented yet.")
    elif provider == "anthropic":
        raise NotImplementedError("Anthropic API integration is not implemented yet.")
    else:
        raise ValueError(f"Unsupported provider: {provider}")


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
    timeout: Optional[int] = None,
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
        timeout (Optional[int], optional): Timeout for the HTTP request. Defaults to None.

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
    timeout: int = 30,  # Default 30 seconds
) -> str:
    """
    Generates a response using the Ollama model with optional images.

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
    """
    try:
        options = Options(
            temperature=temperature,
            num_ctx=max_tokens,
            request_timeout=timeout,
        )

        if json_mode:
            return retry_json_generation(
                model_name=model_name,
                prompt=prompt,
                options=options,
                images=images,
            )

        kwargs = {
            "model": model_name,
            "prompt": prompt,
            "options": options,
            "format": "",
        }

        if images:  # Only include images if list is not None and not empty
            kwargs["images"] = images

        return ollama.generate(**kwargs)["response"]

    except requests.exceptions.Timeout:
        raise TimeoutError(f"Request timed out after {timeout} seconds")
    except ConnectionError:
        raise ConnectionError(
            "Failed to connect to Ollama server. Please check if Ollama is running."
        )
    except Exception as e:
        logging.error(f"Error in generate_with_image_ollama: {str(e)}")
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
        timeout (int): Maximum time to wait for the response.

    Returns:
        str: The generated response from the API.
    """
    try:
        # Initialize OpenAI client with timeout
        client = OpenAI(
            base_url=BASE_URL,
            api_key=KEY,
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
                timeout=timeout,
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
