from typing import List, Dict
def evaluate_responses(
    responses: List[Dict],
    answer: str,
) -> bool:
    """Evaluate the responses from the debate.

    Args:
        responses: List of agent responses from the most recent round of debate.
        answer: The correct answer to the question.

    Returns:
        bool: True if all responses are the same and match the answer, False otherwise.
    """
    raw_responses = [response["response"] for response in responses]
    # Check if all responses are the same
    if len(set(raw_response["answer"] for raw_response in raw_responses)) == 1:
        # Check if the common answer matches the expected answer
        return responses[0]["answer"] == answer
    return False