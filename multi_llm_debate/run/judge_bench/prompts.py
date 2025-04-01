from typing import Dict, List

NEW_LINE = "\n"

JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the passage",
    "Final Answer": "A/B"
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
    "Final Answer": "A/B"
}
"""

NON_JSON_FORMAT = """
Reasoning: your reasoning based on the passage
Final Answer: A/B
"""

NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your reasoning
Step 2: second step of your reasoning
Step 3: third step of your reasoning
...
Final Answer: A/B
"""


def build_judge_bench_round_zero_prompt(
    question: str,
    response_a: str,
    response_b: str,
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for the initial round of judge evaluation.

    Args:
        question: The user question to be evaluated
        response_a: Response from Assistant A
        response_b: Response from Assistant B
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format

    Returns:
        str: The formatted prompt for judge evaluation
    """
    prompt = (
        "Please act as an impartial judge and evaluate the quality of the "
        "responses provided by two AI assistants to the user question "
        "displayed below. You should choose the assistant that follows the "
        "user's instructions and answers the user's question better. " + NEW_LINE
    )

    prompt += (
        "Avoid any position biases and ensure that the order in "
        "which the responses were presented does not influence your decision. "
        "Do not allow the length of the responses to influence your evaluation. "
        "Do not favor certain names of the assistants. Be as objective as "
        "possible. " + NEW_LINE
    )

    prompt += (
        "IMPORTANT: For your final answer, you MUST select either 'A' or 'B' "
        "as the better assistant. No other answers are acceptable. " + NEW_LINE
    )

    if json_mode:
        prompt += "Answer in the following JSON format:" + NEW_LINE
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value must be only 'A' or 'B'. "
            + "Do not include any other text after 'Final Answer: A' or 'Final Answer: B'."
            + NEW_LINE
        )
    else:
        prompt += "Answer in the following format:" + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer: ' MUST be placed at the end of your response, "
            + "and the value must be only 'A' or 'B'. "
            + "Do not include any other text after 'Final Answer: A' or 'Final Answer: B'."
            + NEW_LINE
        )

    prompt += NEW_LINE + "[User Question]" + NEW_LINE
    prompt += question + NEW_LINE + NEW_LINE

    prompt += "[The Start of Assistant A's Answer]" + NEW_LINE
    prompt += response_a + NEW_LINE
    prompt += "[The End of Assistant A's Answer]" + NEW_LINE + NEW_LINE

    prompt += "[The Start of Assistant B's Answer]" + NEW_LINE
    prompt += response_b + NEW_LINE
    prompt += "[The End of Assistant B's Answer]"

    prompt += NEW_LINE
    prompt += "Which assistant provided the better response? A or B?"

    prompt += NEW_LINE
    prompt += "Your answer:"

    return prompt


def build_judge_bench_round_n_prompt(
    question: str,
    response_a: str,
    response_b: str,
    responses: List[str | Dict],
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for subsequent rounds of judge evaluation.

    Args:
        question: The user question to be evaluated
        response_a: Response from Assistant A
        response_b: Response from Assistant B
        responses: Previous responses from judge models
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format

    Returns:
        str: The formatted prompt for judge evaluation
    """
    prompt = (
        "Several other judges have provided evaluations of two AI assistant "
        "responses to a user question. Below are their evaluations: " + NEW_LINE
    )

    for i, response in enumerate(responses, 1):
        prompt += f"Judge {i}: {response}" + NEW_LINE

    prompt += NEW_LINE
    prompt += (
        "Please act as an independent impartial judge and evaluate the quality "
        "of the responses provided by two AI assistants to the user question "
        "displayed below. Consider the previous judges' evaluations, but make "
        "your own assessment. You should choose the assistant that follows the "
        "user's instructions and answers the user's question better." + NEW_LINE
    )

    prompt += (
        "Avoid any position biases and ensure that the order in "
        "which the responses were presented does not influence your decision. "
        "Do not allow the length of the responses to influence your evaluation. "
        "Do not favor certain names of the assistants. Be as objective as "
        "possible. " + NEW_LINE
    )

    prompt += (
        "IMPORTANT: For your final answer, you MUST select either 'A' or 'B' "
        "as the better assistant. No other answers are acceptable. " + NEW_LINE
    )

    if json_mode:
        prompt += "Answer in the following JSON format:" + NEW_LINE
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value must be only 'A' or 'B'. "
            + "Do not include any other text after 'Final Answer: A' or 'Final Answer: B'."
            + NEW_LINE
        )
    else:
        prompt += "Answer in the following format:" + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer: ' MUST be placed at the end of your response, "
            + "and the value must be only 'A' or 'B'. "
            + "Do not include any other text after 'Final Answer: A' or 'Final Answer: B'."
            + NEW_LINE
        )

    prompt += NEW_LINE + "[User Question]" + NEW_LINE
    prompt += question + NEW_LINE + NEW_LINE

    prompt += "[The Start of Assistant A's Answer]" + NEW_LINE
    prompt += response_a + NEW_LINE
    prompt += "[The End of Assistant A's Answer]" + NEW_LINE + NEW_LINE

    prompt += "[The Start of Assistant B's Answer]" + NEW_LINE
    prompt += response_b + NEW_LINE
    prompt += "[The End of Assistant B's Answer]"

    prompt += NEW_LINE
    prompt += "Which assistant provided the better response? A or B?"

    prompt += NEW_LINE
    prompt += "Your answer:"

    return prompt
