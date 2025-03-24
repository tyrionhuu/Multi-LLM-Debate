from typing import Dict, Literal, Union

Answer = Literal["true", "false"]
Reasoning = Union[str, Dict[str, str]]


def extract_bool_answer(response: str) -> Answer:
    """
    Extract true or false answer from the response string, using the last occurrence.

    Args:
        response: The response string from the LLM.

    Returns:
        Answer: "true" or "false". Uses the last occurrence of true/false/yes/no.

    Raises:
        ValueError: If no recognizable answer is found.
    """
    response = response.lower()
    last_true = max(response.rfind("true"), response.rfind("yes"))
    last_false = max(response.rfind("false"), response.rfind("no"))

    if last_true == -1 and last_false == -1:
        raise ValueError("Answer not recognized")

    return "true" if last_true > last_false else "false"


def extract_caption_a_b_answer(response: str) -> Literal["A", "B"]:
    """
    Extract answer from the response string, using the last occurrence.

    Args:
        response: The response string from the LLM.

    Returns:
        Answer: "A" or "B". Uses the last occurrence of A/B.
    """
    last_a = response.rfind("A")
    last_b = response.rfind("B")

    if last_a == -1 and last_b == -1:
        raise ValueError("Answer not recognized")

    return "A" if last_a > last_b else "B"
