import base64
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from PIL import Image
from requests.exceptions import ConnectionError, Timeout

from ..utils.config_manager import get_api_key
from ..utils.logging_config import setup_logging
from .utils import encode_image

# Set up logger
logger = setup_logging(__name__)
logger.setLevel(logging.DEBUG)

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
    vision: bool = False,
    images: Union[
        str, List[str], bytes, List[bytes], Image.Image, List[Image.Image], None
    ] = None,
    api_key: Optional[str] = None,
) -> str:
    """Calls the OpenAI API with the provided parameters and returns the response.

    Can handle both text-only and vision models based on the vision parameter.

    Args:
        model_name (str): The name of the model to use.
        base_url (Optional[str]): The base URL for the OpenAI API.
        prompt (str): The text prompt for the model.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the response.
        json_mode (bool): Whether the response should be in JSON format.
        timeout (Optional[int]): Timeout in seconds for the request. Defaults to 30.
        vision (bool): Whether to use vision models.
        images (Union[str, List[str], bytes, List[bytes], Image.Image, List[Image.Image], None]):
            Image inputs when using vision models.
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
        processed_images = []
        if vision and images is not None:
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

        # Use the API key from arguments or the global one
        api_key_to_use = api_key or KEY

        # Generate API messages
        messages = generate_api_messages(
            prompt=prompt, images=processed_images if vision else None
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
            seed=42,
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
    images: Optional[List[Union[str, bytes]]] = None,
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
    """Example usage of the call_model function."""
    prompt = "Hello, how are you?"
    model_name = "/data/share_weight/Qwen2-7B"
    base_url = "http://localhost:8000/v1"
    response = call_model(
        model_name=model_name,
        prompt=prompt,
        base_url=base_url,
        temperature=0.7,
        max_tokens=1000,
        json_mode=False,
        timeout=30,
    )
    print(response)


if __name__ == "__main__":
    main()
