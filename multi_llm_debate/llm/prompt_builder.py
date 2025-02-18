from typing import Any, Callable, Dict, List


class PromptBuilder:
    """Base class for building prompts with different parameter requirements."""

    def __init__(
        self,
        round_zero_fn: Callable[..., str],
        round_n_fn: Callable[..., str],
        prompt_params: Dict[str, Any],
    ):
        """
        Args:
            round_zero_fn: Function to build initial round prompt
            round_n_fn: Function to build subsequent round prompts
            prompt_params: Dictionary of parameters needed by prompt functions
        """
        self.round_zero_fn = round_zero_fn
        self.round_n_fn = round_n_fn
        self.prompt_params = prompt_params

    def build_round_zero(self) -> str:
        return self.round_zero_fn(**self.prompt_params)

    def build_round_n(self, responses: List[str]) -> str:
        return self.round_n_fn(**self.prompt_params, responses=responses)
