import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from requests.exceptions import ConnectionError, Timeout

from ..utils.config_manager import get_api_key
from ..utils.logging_config import setup_logging
from .utils import encode_image

# Set up logger
logger = setup_logging(__name__)
logger.setLevel(logging.INFO)

KEY = get_api_key()

if KEY.strip() == "":
    KEY = input("Please enter your API key: ")
    from ..utils.config_manager import save_api_key

    save_api_key(KEY)

logging.getLogger("httpx").setLevel(logging.WARNING)


def call_model(
    model_name: str = "gpt-4",
    base_url: str = None,
    prompt: str = "",
    temperature: float = 1.0,
    max_tokens: int = 6400,
    json_mode: bool = False,
    timeout: Optional[int] = 30,
    images: Union[str, Path, List[str], List[Path], None] = None,
    api_key: Optional[str] = None,
) -> str:
    """Calls the OpenAI API with the provided parameters and returns the response.

    Can handle both text-only and vision models based on the presence of images.

    Args:
        model_name (str): The name of the model to use.
        base_url (Optional[str]): The base URL for the OpenAI API.
        prompt (str): The text prompt for the model.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the response.
        json_mode (bool): Whether the response should be in JSON format.
        timeout (Optional[int]): Timeout in seconds for the request. Defaults to 30.
        images (Union[str, Path, List[str], List[Path], None]):
            Image file paths when using vision models.
        api_key (Optional[str]): The API key to use. Defaults to the one from config.

    Returns:
        str: The generated response from the model.

    Raises:
        ConnectionError: If there's a timeout or connection issue
        ValueError: If there's an issue with the parameters
    """
    start_time = time.time()
    logger.info(
        f"Calling OpenAI {model_name} (timeout={timeout}s, json={json_mode}, "
        f"base_url={'custom' if base_url else 'default'})"
    )

    try:
        # Process images if provided
        processed_images: List[str] = []
        if images is not None:
            # Convert single items to list
            if not isinstance(images, list):
                images = [images]

            # Validate and process all images
            for img in images:
                if isinstance(img, (str, Path)):
                    img_path = Path(img)
                    if not img_path.exists():
                        raise ValueError(f"Image file {img_path} does not exist.")
                    processed_images.append(str(img_path))
                else:
                    raise ValueError(
                        "Images must be a string, Path, or list of strings/Paths."
                    )

        # Use the API key from arguments or the global one
        api_key_to_use = api_key or KEY

        # Generate API messages
        messages = generate_api_messages(
            prompt=prompt, images=processed_images if images is not None else None
        )

        # Initialize OpenAI client with timeout and base_url if provided
        client_kwargs = {"api_key": api_key_to_use, "timeout": timeout}

        if not base_url:
            raise ValueError("Base URL is required for OpenAI API calls.")
        client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)

        # Make the API call
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"} if json_mode else None,
            seed=random.randint(0, 2**32 - 1),
        )
        logger.debug(f"API response: {response}")
        # Extract response content
        response_str = response.choices[0].message.content

        # Process JSON response if needed
        if json_mode:
            try:
                return json.dumps(json.loads(response_str))
            except json.JSONDecodeError:
                logger.warning("API returned invalid JSON despite json_mode=True")
                return response_str

        elapsed = time.time() - start_time
        logger.info(f"Call to OpenAI/{model_name} completed in {elapsed:.2f}s")
        return response_str

    except Timeout:
        elapsed = time.time() - start_time
        logger.error(f"Timeout error calling {model_name} after {elapsed:.2f}s")
        raise ConnectionError(
            f"Timeout error with OpenAI service after {timeout} seconds"
        )
    except ConnectionError as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Connection error calling {model_name} after {elapsed:.2f}s: {str(e)}"
        )
        raise ConnectionError(f"Connection error with OpenAI service: {str(e)}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Error calling {model_name} after {elapsed:.2f}s: {str(e)}", exc_info=False
        )
        raise ValueError(f"Error with OpenAI service: {str(e)}")


def generate_api_messages(
    prompt: str,
    images: Optional[List[Union[str, Path]]] = None,
) -> List[Dict[str, Any]]:
    """
    Prepares the messages payload for the API call with optional images and a prompt.

    Args:
        prompt (str): The text prompt for the model.
        images (Optional[List[Union[str, Path]]]): List of image file paths.
            If None, returns text-only message format.

    Returns:
        list[dict]: A list of messages formatted for the API call.
    """
    if not images:
        return [{"role": "user", "content": prompt}]

    # Convert Path objects to str
    image_paths = [str(img) if isinstance(img, Path) else img for img in images]

    if len(image_paths) == 1:
        base64_image = encode_image(image_paths[0])
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
        base64_images = [encode_image(img) for img in image_paths]
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
