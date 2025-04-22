from typing import Dict, List

NEW_LINE = "\n"
DIVIDER = "#" * 80
JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the passage",
    "Final Answer": "0/1"
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
    "Final Answer": "0/1"
}
"""

NON_JSON_FORMAT = """
Reasoning: your reasoning based on the passage
Final Answer: 0/1
"""

NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your reasoning
Step 2: second step of your reasoning
Step 3: third step of your reasoning
...
Final Answer: 0/1
"""


def build_big_bench_round_zero_prompt(
    input: str,
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for the initial round of judge evaluation.

    Args:
        input: The input string for the evaluation
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format

    Returns:
        str: The formatted prompt for judge evaluation
    """
    prompt = "As an assistant, your task is to serve as a response judge.\n" + NEW_LINE

    prompt += (
        "Determine whether the following statement is plausible or implausible."
    ) + NEW_LINE

    prompt += (
        "If the statement is plausible, you should rate it as 1.\n"
        "If the statement is implausible, you should rate it as 0.\n"
    )

    if json_mode:
        prompt += "You MUST answer in the following JSON format:" + NEW_LINE
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value must be only 0 or 1. "
            + "Do not include any other text after 'Final Answer: 0' or 'Final Answer: 1'."
            + NEW_LINE
        )
    else:
        prompt += "You MUST answer in the following format:" + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer: ' MUST be placed at the end of your response, "
            + "and the value must be only 0 or 1. "
            + "Do not include any other text after 'Final Answer: 0' or 'Final Answer: 1'."
            + NEW_LINE
        )
    prompt += DIVIDER + NEW_LINE
    prompt += "[Statement]" + NEW_LINE
    prompt += input + NEW_LINE
    prompt += DIVIDER + NEW_LINE
    prompt += "Your answer:" + NEW_LINE

    return prompt


def build_big_bench_round_n_prompt(
    input: str,
    responses: List[str | Dict],
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for subsequent rounds of judge evaluation.

    Args:
        input (str): The input string for the evaluation.
        responses (List[str | Dict]): Previous responses from judge models.
        use_cot (bool): Whether to use chain-of-thought prompting.
        json_mode (bool): Whether to return response in JSON format.

    Returns:
        str: The formatted prompt for judge evaluation.
    """
    prompt = "As an assistant, your task is to serve as a response judge.\n" + NEW_LINE

    prompt += (
        "Several other judges have provided evaluations. "
        "Review their assessments, but make your own independent evaluation.\n"
    ) + NEW_LINE

    prompt += (
        "Determine whether the following statement is plausible or implausible."
    ) + NEW_LINE

    prompt += (
        "If the statement is plausible, you should rate it as 1.\n"
        "If the statement is implausible, you should rate it as 0.\n"
    )

    if json_mode:
        prompt += "You MUST answer in the following JSON format:" + NEW_LINE
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value must be only 0 or 1. "
            + "Do not include any other text after 'Final Answer: 0' or 'Final Answer: 1'."
            + NEW_LINE
        )
    else:
        prompt += "You MUST answer in the following format:" + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer: ' MUST be placed at the end of your response, "
            + "and the value must be only 0 or 1. "
            + "Do not include any other text after 'Final Answer: 0' or 'Final Answer: 1'."
            + NEW_LINE
        )

    prompt += DIVIDER + NEW_LINE
    prompt += "Previous judge evaluations:" + NEW_LINE
    for i, judge_response in enumerate(responses, 1):
        prompt += f"Judge {i}: {judge_response}" + NEW_LINE
    prompt += DIVIDER + NEW_LINE
    prompt += "[Statement]" + NEW_LINE
    prompt += input + NEW_LINE
    prompt += DIVIDER + NEW_LINE
    prompt += "Your answer:" + NEW_LINE

    return prompt


if __name__ == "__main__":
    # Example usage
    knowledge = "The Earth revolves around the Sun."
    dialogue = (
        "User: Tell me about Earth's orbit.\nAssistant: I'll explain Earth's orbit."
    )
    response = "Earth orbits around the Sun once per year."
    responses = [
        "Judge 1: No hallucination detected. The response aligns with the provided knowledge.",
        "Judge 2: The response contains accurate information about Earth's orbit.",
    ]

    prompt = build_big_bench_round_n_prompt(knowledge, dialogue, response, responses)
    print(prompt)
