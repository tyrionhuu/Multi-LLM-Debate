import asyncio
import base64
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import google.auth
import google.auth.transport.requests
from openai import OpenAI, RateLimitError
from requests.exceptions import ConnectionError, Timeout

from ..utils.config_manager import get_api_key
from .utils import encode_image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logger
logger = logging.getLogger(__name__)

KEY = get_api_key()

if KEY.strip() == "":
    KEY = input("Please enter your API key: ")
    from ..utils.config_manager import save_api_key

    save_api_key(KEY)

logging.getLogger("httpx").setLevel(logging.WARNING)


def _get_google_access_token_and_url(
    project_id: str = "multi-llm-debate",
    location: str = "us-central1",
    endpoint_id: str = "openapi",
) -> Tuple[str, str]:
    """Get Google Cloud access token and Gemini endpoint URL.

    Args:
        project_id (str): Google Cloud project ID.
        location (str): Region for the endpoint.
        endpoint_id (str): Endpoint ID ("openapi" for Gemini).

    Returns:
        Tuple[str, str]: (access_token, base_url)
    """
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(google.auth.transport.requests.Request())
    access_token = credentials.token
    base_url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/"
        f"{project_id}/locations/{location}/endpoints/{endpoint_id}"
    )
    return access_token, base_url


def _is_bytes_like(obj: Any) -> bool:
    """Check if an object is bytes-like (bytes or bytearray).

    Args:
        obj (Any): Object to check

    Returns:
        bool: True if the object is bytes-like, False otherwise
    """
    return isinstance(obj, (bytes, bytearray))


def call_model(
    model_name: str = "gpt-4",
    base_url: str = "",
    prompt: str = "",
    temperature: float = 1.0,
    max_tokens: int = 6400,
    json_mode: bool = False,
    timeout: Optional[int] = 30,
    images: Optional[
        Union[str, Path, bytes, List[str], List[Path], List[bytes]]
    ] = None,
    api_key: Optional[str] = None,
    project_id: Optional[str] = "multi-llm-debate",
    location: str = "us-central1",
    endpoint_id: str = "openapi",
) -> str:
    """Calls the OpenAI API or Gemini API with the provided parameters.

    Can handle both text-only and vision models based on the presence of images.

    Args:
        model_name (str): The name of the model to use.
        base_url (Optional[str]): The base URL for the API.
        prompt (str): The text prompt for the model.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the response.
        json_mode (bool): Whether the response should be in JSON format.
        timeout (Optional[int]): Timeout in seconds for the request. Defaults to 30.
        images (Optional[Union[str, Path, bytes, List[str], List[Path], List[bytes]]]):
            Image file paths or raw image bytes when using vision models.
        api_key (Optional[str]): The API key to use. Defaults to the one from config.
        project_id (Optional[str]): GCP project ID for Gemini.
        location (str): GCP region for Gemini.
        endpoint_id (str): Gemini endpoint ID.

    Returns:
        str: The generated response from the model.

    Raises:
        ConnectionError: If there's a timeout or connection issue
        ValueError: If there's an issue with the parameters
    """
    start_time = time.time()
    logger.info(
        f"Calling {model_name} (timeout={timeout}s, json={json_mode}, "
        f"base_url={base_url})"
    )

    try:
        # Process images if provided
        processed_images: List[Union[str, bytes]] = []
        if images is not None:
            # Normalize images to a list, handling all allowed types
            if isinstance(images, (str, Path, bytes)):
                images_list: List[Union[str, Path, bytes]] = [images]
            elif isinstance(images, list):
                images_list = images  # type: ignore
            else:
                raise ValueError(
                    "Images must be a string, Path, bytes, or list of these types."
                )

            for img in images_list:
                if isinstance(img, (str, Path)):
                    img_path = Path(img)
                    if not img_path.exists():
                        raise ValueError(f"Image file {img_path} does not exist.")
                    processed_images.append(str(img_path))
                elif _is_bytes_like(img):
                    processed_images.append(img)
                else:
                    raise ValueError(
                        "Images must be a string, Path, bytes, or list of strings/Paths/bytes."
                    )

        # Detect Gemini model
        if "google" in model_name.lower():
            if not project_id:
                raise ValueError("project_id is required for Google models.")
            access_token, gemini_url = _get_google_access_token_and_url(
                project_id=project_id, location=location, endpoint_id=endpoint_id
            )
            api_key_to_use = access_token
            base_url_to_use = base_url or gemini_url
        else:
            api_key_to_use = api_key or KEY
            if not base_url:
                raise ValueError("Base URL is required for OpenAI API calls.")
            base_url_to_use = base_url

        messages = generate_api_messages(
            prompt=prompt, images=processed_images if images is not None else None
        )

        client_kwargs = {"api_key": api_key_to_use, "timeout": timeout}
        client_kwargs["base_url"] = base_url_to_use
        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"} if json_mode else None,  # type: ignore
            seed=random.randint(0, 2**10 - 1),
        )
        logger.debug(f"API response: {response}")
        response_str = response.choices[0].message.content

        if json_mode:
            try:
                return json.dumps(json.loads(response_str or ""))
            except json.JSONDecodeError:
                logger.warning("API returned invalid JSON despite json_mode=True")
                return response_str or ""

        elapsed = time.time() - start_time
        logger.info(f"Call to {model_name} completed in {elapsed:.2f}s")
        return response_str or ""

    except RateLimitError as e:
        logger.error(f"Rate limit error calling {model_name}: {str(e)}", exc_info=False)
        raise ValueError(f"Rate limit error with service: {str(e)}")
    except Timeout:
        elapsed = time.time() - start_time
        logger.error(f"Timeout error calling {model_name} after {elapsed:.2f}s")
        raise ConnectionError(f"Timeout error with service after {timeout} seconds")
    except ConnectionError as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Connection error calling {model_name} after {elapsed:.2f}s: {str(e)}"
        )
        raise ConnectionError(f"Connection error with service: {str(e)}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Error calling {model_name} after {elapsed:.2f}s: {str(e)}", exc_info=False
        )
        raise ValueError(f"Error with service: {str(e)}")


def generate_api_messages(
    prompt: str,
    images: Optional[Sequence[Union[str, bytes]]] = None,
) -> List[Dict[str, Any]]:
    """
    Prepares the messages payload for the API call with optional images and a prompt.

    Args:
        prompt (str): The text prompt for the model.
        images (Optional[Sequence[Union[str, bytes]]]): List of image file paths
            or raw image bytes. If None, returns text-only message format.

    Returns:
        list[dict]: A list of messages formatted for the API call.
    """
    if not images:
        return [{"role": "user", "content": prompt}]

    if len(images) == 1:
        img = images[0]
        if _is_bytes_like(img):
            base64_image = base64.b64encode(img).decode("utf-8")  # type: ignore
        else:
            base64_image = encode_image(str(img))

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
        base64_images = []
        for img in images:
            if _is_bytes_like(img):
                base64_images.append(base64.b64encode(img).decode("utf-8"))  # type: ignore
            else:
                base64_images.append(encode_image(str(img)))

        content: List[Dict[str, Any]] = [
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


async def call_model_async(
    model_name: str = "gpt-4",
    base_url: Optional[str] = None,
    prompt: str = "",
    temperature: float = 1.0,
    max_tokens: int = 6400,
    json_mode: bool = False,
    timeout: Optional[int] = 30,
    images: Optional[
        Union[str, Path, bytes, List[str], List[Path], List[bytes]]
    ] = None,
    api_key: Optional[str] = None,
    project_id: Optional[str] = "multi-llm-debate",
    location: str = "us-central1",
    endpoint_id: str = "openapi",
) -> str:
    """Async version of call_model.

    Args:
        model_name (str): The name of the model to use.
        base_url (Optional[str]): The base URL for the API.
        prompt (str): The text prompt for the model.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the response.
        json_mode (bool): Whether the response should be in JSON format.
        timeout (Optional[int]): Timeout in seconds for the request.
        images (Optional[Union[str, Path, bytes, List[str], List[Path], List[bytes]]]):
            Image file paths or raw image bytes when using vision models.
        api_key (Optional[str]): The API key to use.
        project_id (Optional[str]): GCP project ID for Gemini.
        location (str): GCP region for Gemini.
        endpoint_id (str): Gemini endpoint ID.

    Returns:
        str: The generated response from the model.
    """
    # Use asyncio to run the synchronous call_model in a separate thread
    return await asyncio.to_thread(
        call_model,
        model_name=model_name,
        base_url=base_url or "",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        timeout=timeout,
        images=images,
        api_key=api_key,
        project_id=project_id,
        location=location,
        endpoint_id=endpoint_id,
    )


async def call_model_batch(
    model_name: str = "gpt-4",
    base_url: Optional[str] = None,
    prompts: Optional[List[str]] = None,
    temperature: float = 1.0,
    max_tokens: int = 6400,
    json_mode: bool = False,
    timeout: Optional[int] = 30,
    images: Optional[
        List[Optional[Union[str, Path, bytes, List[str], List[Path], List[bytes]]]]
    ] = None,
    api_key: Optional[str] = None,
    project_id: Optional[str] = "multi-llm-debate",
    location: str = "us-central1",
    endpoint_id: str = "openapi",
    batch_size: int = 5,
) -> List[str]:
    """Calls the OpenAI API or Gemini API with multiple prompts asynchronously.

    Args:
        model_name (str): The name of the model to use.
        base_url (Optional[str]): The base URL for the API.
        prompts (Optional[List[str]]): List of text prompts for the model.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the response.
        json_mode (bool): Whether the response should be in JSON format.
        timeout (Optional[int]): Timeout in seconds for the request.
        images (Optional[List[Optional[Union[str, Path, bytes, List[str], List[Path], List[bytes]]]]], optional):
            List of image file paths, raw image bytes, or lists of paths/bytes when using vision models.
            Should match the length of prompts or be None.
        api_key (Optional[str]): The API key to use.
        project_id (Optional[str]): GCP project ID for Gemini.
        location (str): GCP region for Gemini.
        endpoint_id (str): Gemini endpoint ID.
        batch_size (int): Maximum number of concurrent API calls. Defaults to 5.

    Returns:
        List[str]: The generated responses from the model.

    Raises:
        ValueError: If prompts is None or empty, or if images is provided but length
            doesn't match prompts.
    """
    if not prompts:
        raise ValueError("prompts must be a non-empty list of strings")

    # Validate images if provided
    if images is not None:
        if len(images) != len(prompts):
            raise ValueError(
                "If images is provided, it must have the same length as prompts"
            )
    else:
        # Set to None for each prompt when not provided
        images = [None] * len(prompts)  # type: ignore

    logger.info(f"Processing batch of {len(prompts)} prompts with model {model_name}")
    start_time = time.time()

    # Process prompts in batches of batch_size
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_end = min(i + batch_size, len(prompts))
        batch_prompts = prompts[i:batch_end]
        batch_images = images[i:batch_end]  # type: ignore

        logger.info(f"Processing batch {i//batch_size + 1}: {i} to {batch_end-1}")

        # Create tasks for current batch
        tasks = []
        for j, prompt in enumerate(batch_prompts):
            img = batch_images[j]
            task = asyncio.create_task(
                call_model_async(
                    model_name=model_name,
                    base_url=base_url,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    timeout=timeout,
                    images=img,
                    api_key=api_key,
                    project_id=project_id,
                    location=location,
                    endpoint_id=endpoint_id,
                )
            )
            tasks.append(task)

        # Wait for all tasks in this batch to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        # Error handling: log and convert exceptions to error messages
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error in batch {i//batch_size + 1}, prompt {i+idx}: {result}"
                )
                raise ValueError(f"Error processing prompt {i+idx}: {str(result)}")
        results.extend(batch_results)

    # Process results - convert exceptions to error messages
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(f"Error: {str(result)}")
        else:
            processed_results.append(result)

    elapsed = time.time() - start_time
    logger.info(
        f"Batch processing completed in {elapsed:.2f}s for {len(prompts)} prompts"
    )

    return processed_results
