import atexit
import logging
import threading
import time
from typing import Optional

import torch.distributed
from requests.exceptions import ConnectionError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ThreadSafeTimeout:
    """A thread-safe timeout handler that uses threading.Timer instead of signals.

    This class provides a thread-safe alternative to signal-based timeouts
    by using threading.Timer, which works in any thread.
    """

    def __init__(self, timeout: Optional[float], operation_name: str = "API call"):
        """Initialize a thread-safe timeout handler.

        Args:
            timeout: Maximum time in seconds before timing out
            operation_name: Name of the operation for logging
        """
        self.timeout = timeout
        self.operation_name = operation_name
        self.timer = None
        self.timed_out = False
        self.exception = None
        self._lock = threading.Lock()

    def _timeout_callback(self):
        """Called when the timer expires."""
        with self._lock:
            if not self.timed_out:
                self.timed_out = True
                self.exception = ConnectionError(
                    f"Operation '{self.operation_name}' timed out after {self.timeout} seconds"
                )
                logger.error(
                    f"Timeout ({self.timeout}s) exceeded for {self.operation_name}"
                )

    def __enter__(self):
        """Start the timeout timer if a timeout is specified."""
        if self.timeout and self.timeout > 0:
            self.timer = threading.Timer(self.timeout, self._timeout_callback)
            self.timer.daemon = (
                True  # Allow the program to exit if only the timer is left
            )
            self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cancel the timer when exiting the context."""
        if self.timer:
            self.timer.cancel()

        # If we timed out and there's no other exception, raise our timeout exception
        if self.timed_out and exc_type is None:
            raise self.exception

        # Return False to propagate any other exception
        return False

    def check_timeout(self):
        """Check if timeout has occurred and raise the exception if so.

        Raises:
            ConnectionError: If the operation has timed out.
        """
        with self._lock:
            if self.timed_out and self.exception:
                raise self.exception


class AbortableOllamaRequest:
    """An abortable wrapper for Ollama requests.

    This class allows for aborting Ollama requests that are taking too long,
    similar to the AbortController in JavaScript.
    """

    def __init__(self):
        """Initialize an abortable Ollama request wrapper."""
        self._abort_event = threading.Event()
        self.response = None
        self.error = None
        self.completed = False

    def abort(self):
        """Signal abort to the running request thread."""
        logger.warning("Aborting Ollama request")
        self._abort_event.set()

    def is_aborted(self) -> bool:
        """Check if the request has been aborted.

        Returns:
            bool: True if aborted, False otherwise
        """
        return self._abort_event.is_set()

    def run_with_timeout(
        self,
        func: callable,
        args: tuple = None,
        kwargs: dict = None,
        timeout: int = 30,
    ):
        """Run a function with a timeout and abort capability.

        Args:
            func: The function to run
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            timeout: Timeout in seconds

        Returns:
            Any: The result of the function if successful

        Raises:
            TimeoutError: If the operation times out
            ValueError: If the operation is aborted
            Exception: Any exception raised by the function
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        def _target():
            try:
                self.response = func(*args, **kwargs)
                self.completed = True
            except Exception as e:
                self.error = e

        thread = threading.Thread(target=_target)
        thread.daemon = True
        thread.start()

        # Wait for completion, timeout, or abort
        timeout_time = time.time() + timeout
        check_interval = 0.1  # Check abort status every 100ms

        while thread.is_alive() and time.time() < timeout_time:
            if self.is_aborted():
                logger.warning("Detected abort signal during execution")
                raise ValueError("Request aborted")
            thread.join(timeout=check_interval)

        if thread.is_alive():
            # Timeout occurred
            self.abort()  # Set abort flag even though we can't truly abort
            logger.error(f"Timeout after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

        if self.error:
            raise self.error

        return self.response


class AbortableVLLMInference:
    """A thread-safe abortable wrapper for vLLM inference.

    This class allows direct use of vLLM library with timeout capability.
    """

    def __init__(self):
        """Initialize an abortable vLLM inference wrapper."""
        self._abort_event = threading.Event()
        self.response = None
        self.error = None
        self.completed = False

    def abort(self):
        """Signal abort to the running inference thread."""
        logger.warning("Aborting vLLM inference")
        self._abort_event.set()

    def is_aborted(self) -> bool:
        """Check if the inference has been aborted.

        Returns:
            bool: True if aborted, False otherwise
        """
        return self._abort_event.is_set()

    def run_with_timeout(
        self,
        func: callable,
        args: tuple = None,
        kwargs: dict = None,
        timeout: int = 30,
    ):
        """Run a function with a timeout and abort capability.

        Args:
            func: The function to run
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            timeout: Timeout in seconds

        Returns:
            Any: The result of the function if successful

        Raises:
            TimeoutError: If the operation times out
            ValueError: If the operation is aborted
            Exception: Any exception raised by the function
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        def _target():
            try:
                self.response = func(*args, **kwargs)
                self.completed = True
            except Exception as e:
                self.error = e

        thread = threading.Thread(target=_target)
        thread.daemon = True
        thread.start()

        # Wait for completion, timeout, or abort
        timeout_time = time.time() + timeout
        check_interval = 0.1  # Check abort status every 100ms

        while thread.is_alive() and time.time() < timeout_time:
            if self.is_aborted():
                logger.warning("Detected abort signal during vLLM inference")
                raise ValueError("Inference aborted")
            thread.join(timeout=check_interval)

        if thread.is_alive():
            # Timeout occurred
            self.abort()
            logger.error(f"vLLM inference timeout after {timeout}s")
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

        if self.error:
            raise self.error

        return self.response


def shutdown_vllm_models() -> None:
    """Properly shutdown all loaded vLLM models and cleanup process groups.

    This function ensures all distributed resources are properly released,
    preventing resource leaks and warnings about destroy_process_group().
    """
    global _vllm_models

    if not _vllm_models:
        return

    logger.info(f"Shutting down {len(_vllm_models)} vLLM models")

    # Delete model instances to free GPU memory
    for model_name, model in _vllm_models.items():
        try:
            logger.debug(f"Shutting down vLLM model: {model_name}")
            del model
        except Exception as e:
            logger.warning(f"Error shutting down model {model_name}: {str(e)}")

    # Clear the models dictionary
    _vllm_models.clear()

    # Cleanup process groups if initialized
    if torch.distributed.is_initialized():
        try:
            logger.debug("Destroying PyTorch distributed process groups")
            torch.distributed.destroy_process_group()
        except Exception as e:
            logger.warning(f"Error destroying process groups: {str(e)}")


# Register shutdown handler to run when the program exits
atexit.register(shutdown_vllm_models)
