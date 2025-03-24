from typing import Dict, List

NEW_LINE = "\n"

JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the passage",
    "answer": "true/false"
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
    "answer": "true/false"
}
"""

NON_JSON_FORMAT = """
Reasoning: your reasoning based on the passage
Final Answer: true/false
"""

NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your reasoning
Step 2: second step of your reasoning
Step 3: third step of your reasoning
...
Final Answer: true/false
"""


def build_bool_q_round_zero_prompt(
    question: str,
    passage: str,
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for round zero of boolean question debate.

    Args:
        question: The question to be answered
        passage: The passage to base the answer on
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format
        **kwargs: Additional arguments that will be ignored

    Returns:
        str: The formatted prompt
    """
    prompt = "You will be given a true or false question which is based on a passage. "
    if json_mode:
        prompt += "Answer in the following JSON format:" + NEW_LINE
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
    else:
        prompt += "Answer in the following format:" + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
    prompt += NEW_LINE
    prompt += "Question: " + question + NEW_LINE
    prompt += "Passage: " + passage

    return prompt


def build_bool_q_round_n_prompt(
    question: str,
    passage: str,
    responses: List[str | Dict],
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for subsequent rounds of boolean question debate.

    Args:
        question: The question to be answered
        passage: The passage to base the answer on
        responses: Previous responses from other models
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format
        **kwargs: Additional arguments that will be ignored

    Returns:
        str: The formatted prompt
    """
    prompt = (
        "Several other models have provided responses to a true or false question, below are their responses: "
        + NEW_LINE
    )

    for i, response in enumerate(responses, 1):
        prompt += f"Model {i}: {response}" + NEW_LINE

    prompt += NEW_LINE
    prompt += (
        "Consider these responses when answering the following true or false question."
        + NEW_LINE
    )
    if json_mode:
        prompt += "Answer in the following JSON format:" + NEW_LINE
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
    else:
        prompt += "Answer in the following format:" + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
    prompt += NEW_LINE
    prompt += "Question: " + question + NEW_LINE
    prompt += "Passage: " + passage

    return prompt
