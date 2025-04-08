import re
from typing import Literal


def preprocess_dataframe():
    pass


def load_truthful_qa_dataset():
    pass



def extract_caption_a_b_c_answer(response: str) -> Literal["A", "B", "C"]:
    """
    Extract answer from the response string.

    First tries to find "Final Answer: A", "Final Answer: B", or "Final Answer: C" pattern.
    If not found, falls back to finding the last occurrence of A, B, or C.

    Args:
        response: The response string from the LLM.

    Returns:
        Literal["A", "B", "C"]: Answer "A", "B", or "C".
    """
    # Try to find "Final Answer: X" pattern
    match = re.search(r"Final Answer:\s*([ABC])", response)
    if match:
        return match.group(1)
    else:
        raise ValueError(
            "No valid answer found in the response. Please ensure the response contains 'Final Answer: A', 'Final Answer: B', or 'Final Answer: C'."
        )

def compare_truthful_qa_response(
    response: Literal["A", "B", "C"],
    answer: str,
) -> bool:
    """Compare the responses from the judge bench.

    Args:
        response: The response string from the LLM.
        answer: The correct answer to the question ("A"/"B"/"C").

    Returns:
        bool: True if the response matches the answer, False otherwise.
    """
    try:
        return response.upper() == answer.upper()
    except AttributeError:
        return False
