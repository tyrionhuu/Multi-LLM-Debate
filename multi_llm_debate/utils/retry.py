import functools
import logging
import time
from typing import Any, Callable, Optional, Type, Union

def retry_with_timeout(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Optional[Union[Type[Exception], tuple[Type[Exception], ...]]] = None
) -> Callable:
    """A decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        backoff_factor: Factor to multiply delay for each subsequent retry.
        exceptions: Exception types to catch and retry. Defaults to Exception.

    Returns:
        Callable: Decorated function with retry logic.
    """
    if exceptions is None:
        exceptions = Exception

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise last_exception
                    
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                        f"Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor

        return wrapper
    return decorator
