import json
from typing import Dict, Literal, Union

Answer = Literal["Yes", "No"]
Reasoning = Union[str, Dict[str, str]]


def extract_bool_answer(response: str) -> Answer:
    """
    Extract Yes/No answer from a JSON response string.

    Args:
        response: JSON string containing reasoning and answer

    Returns:
        "Yes" or "No"

    Raises:
        ValueError: If response is not valid JSON or missing required fields
    """
    try:
        parsed = json.loads(response)
        answer = parsed.get("answer")

        if not answer or answer not in ["Yes", "No"]:
            raise ValueError("Answer must be 'Yes' or 'No'")

        return answer

    except json.JSONDecodeError:
        raise ValueError("Response is not valid JSON")
