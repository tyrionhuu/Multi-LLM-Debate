from typing import List, Any, Dict, Callable

NEW_LINE = "\n"

JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the passage",
    "answer": "Yes/No"
}
"""

JSON_FORMAT_COT = """
{
    "reasoning": {
        "step_1": "first step of your reasoning",
        "step_2": "second step of your reasoning",
        "step_3": "third step of your reasoning",
        "...": "continue with as many steps as needed"
    },
    "answer": "Yes/No"
}
"""


def build_bool_q_round_zero_prompt(
    question: str, passage: str, use_cot: bool = True
) -> str:
    prompt = "You will be given a yes-no question which is based on a passage. "
    prompt += "Answer in the following JSON format:" + NEW_LINE
    prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
    prompt += NEW_LINE
    prompt += "Question: " + question + NEW_LINE
    prompt += "Passage: " + passage

    return prompt


def build_bool_q_round_n_prompt(
    question: str, passage: str, responses: List[str], use_cot: bool = True
) -> str:
    prompt = (
        "Several other models have provided responses to a yes-no question, below are their responses: "
        + NEW_LINE
    )

    for i, response in enumerate(responses, 1):
        prompt += f"Model {i}: {response}" + NEW_LINE

    prompt += NEW_LINE
    prompt += (
        "Consider these responses when answering the following yes-no question."
        + NEW_LINE
    )
    prompt += "Answer in the following JSON format:" + NEW_LINE
    prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
    prompt += NEW_LINE
    prompt += "Question: " + question + NEW_LINE
    prompt += "Passage: " + passage

    return prompt


class PromptBuilder:
    """Base class for building prompts with different parameter requirements."""
    
    def __init__(
        self,
        round_zero_fn: Callable[..., str],
        round_n_fn: Callable[..., str],
        prompt_params: Dict[str, Any]
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
