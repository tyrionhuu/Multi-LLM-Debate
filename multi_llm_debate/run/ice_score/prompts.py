from typing import Dict, List

NEW_LINE = "\n"
DIVIDER = "#" * 80
JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the passage",
    "Final Answer": "x"
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
    "Final Answer": "x"
}
"""

NON_JSON_FORMAT = """
Reasoning: your reasoning based on the passage
Final Answer: x
"""

NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your reasoning
Step 2: second step of your reasoning
Step 3: third step of your reasoning
...
Final Answer: x
"""


def build_ice_score_round_zero_prompt(
    question: str,
    response: str,
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for the initial round of judge evaluation.

    Args:
        question: The user question to be evaluated
        response: Response from the AI assistant
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format

    Returns:
        str: The formatted prompt for judge evaluation
    """
    # Convert question and response to strings if they are lists (for backward compatibility)
    if isinstance(question, list):
        question = NEW_LINE.join(question)
    if isinstance(response, list):
        response = NEW_LINE.join(response)

    prompt = (
        "You will be given the code snippet for a problem. "
        "Your task is to rate the code snippet only on one metric."
        "Please make sure you read and understand these instructions carefully."
        "Please keep this document open while reviewing, and refer to it as needed."
    ) + NEW_LINE

    prompt += (
        "Evaluation Criteria:\n"
        "Functional Correctness (0-4) - Execution-based quality of the code "
        "snippet combined with the problem.\n"
        "The correctness is measured by all possible unit tests, and the "
        "comparison of the reference code.\n"
        "The combination of the code snippet and the problem should pass all "
        "the possible tests based on your understanding of the reference code.\n"
        "The length of the code snippet cannot determine the correctness. You "
        "need to assess the logic line by line.\n"
        "- A score of 0 (failing all possible tests) means that the code "
        "snippet is totally incorrect and meaningless.\n"
        "- A score of 4 (passing all possible tests) means that the code "
        "snippet is totally correct and can handle all cases.\n"
    )

    prompt += (
        "Evaluation Steps:\n"
        "1. Read the problem carefully and identify required functionalities "
        "of the implementation.\n"
        "2. Read the code snippet and compare it to the problem. Check if the "
        "code snippet covers all required functionalities of the problem.\n"
        "3. Assign a score for functional correctness on a scale of 0 to 4, "
        "where 0 is the lowest and 4 is the highest based on the Evaluation "
        "Criteria.\n" + DIVIDER + NEW_LINE
    )

    if json_mode:
        prompt += (
            "You MUST answer in the following JSON format (x is an integer from 0 to 4):"
            + NEW_LINE
        )
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value must be only a number between 0 and 4. "
            + "Do not include any other text after 'Final Answer: 0' or 'Final Answer: 4'."
            + NEW_LINE
        )
    else:
        prompt += (
            "You MUST answer in the following format (x is an integer from 0 to 4):"
            + NEW_LINE
        )
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer: ' MUST be placed at the end of your response, "
            + "and the value must be only a number between 0 and 4. "
            + "Do not include any other text after 'Final Answer: 0' or 'Final Answer: 4'."
            + NEW_LINE
        )
    prompt += DIVIDER + NEW_LINE
    prompt += "[problem]" + NEW_LINE
    prompt += question + NEW_LINE
    prompt += "[The Start of the Code Snippet]" + NEW_LINE
    prompt += response + NEW_LINE
    prompt += "[The End of the Code Snippet]" + NEW_LINE
    prompt += NEW_LINE + "Your answer:" + NEW_LINE

    return prompt


def build_ice_score_round_n_prompt(
    question: str,
    response: str,
    responses: List[str | Dict],
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for subsequent rounds of judge evaluation.

    Args:
        question: The user question to be evaluated
        response: Response from the AI assistant
        responses: Previous responses from judge models
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format

    Returns:
        str: The formatted prompt for judge evaluation
    """
    # Convert question and response to strings if they are lists (for backward compatibility)
    if isinstance(question, list):
        question = NEW_LINE.join(question)
    if isinstance(response, list):
        response = NEW_LINE.join(response)

    prompt = (
        "Several other judges have provided evaluations of an AI assistant's code "
        "to a problem given by the user. "
        "Below are their evaluations: " + NEW_LINE
    )

    prompt += DIVIDER + NEW_LINE
    for i, response in enumerate(responses, 1):
        prompt += f"Judge {i}: {response}" + NEW_LINE

    prompt += DIVIDER + NEW_LINE
    prompt += (
        "Please act as an independent impartial judge and evaluate the quality "
        "of the code provided by an AI assistant to the user's problem. "
        "displayed below. Consider the previous judges' evaluations, but make "
        "your own assessment. " + NEW_LINE
    )

    prompt += (
        "Evaluation Criteria:\n"
        "Functional Correctness (0-4) - Execution-based quality of the code "
        "snippet combined with the problem.\n"
        "The correctness is measured by all possible unit tests, and the "
        "comparison of the reference code.\n"
        "The combination of the code snippet and the problem should pass all "
        "the possible tests based on your understanding of the reference code.\n"
        "The length of the code snippet cannot determine the correctness. You "
        "need to assess the logic line by line.\n"
        "- A score of 0 (failing all possible tests) means that the code "
        "snippet is totally incorrect and meaningless.\n"
        "- A score of 4 (passing all possible tests) means that the code "
        "snippet is totally correct and can handle all cases.\n"
    )

    prompt += (
        "Evaluation Steps:\n"
        "1. Read the problem carefully and identify required functionalities "
        "of the implementation.\n"
        "2. Read the code snippet and compare it to the problem. Check if the "
        "code snippet covers all required functionalities of the problem.\n"
        "3. Assign a score for functional correctness on a scale of 0 to 4, "
        "where 0 is the lowest and 4 is the highest based on the Evaluation "
        "Criteria.\n" + DIVIDER + NEW_LINE
    )
    if json_mode:
        prompt += "You MUST answer in the following JSON format:" + NEW_LINE
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value must be only a number between 0 and 4. "
            + "Do not include any other text after 'Final Answer: 0' or 'Final Answer: 4'."
            + NEW_LINE
        )
    else:
        prompt += "You MUST answer in the following format:" + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer: ' MUST be placed at the end of your response, "
            + "and the value must be only a number between 0 and 4. "
            + "Do not include any other text after 'Final Answer: 0' or 'Final Answer: 4'."
            + NEW_LINE
        )
    prompt += DIVIDER + NEW_LINE
    prompt += "[problem]" + NEW_LINE
    prompt += question + NEW_LINE
    prompt += "[The Start of the Code Snippet]" + NEW_LINE
    prompt += response + NEW_LINE
    prompt += "[The End of the Code Snippet]" + NEW_LINE
    prompt += NEW_LINE + "Your answer:" + NEW_LINE

    return prompt


if __name__ == "__main__":
    # Example usage
    question = "What is the sum of 2 and 3?"
    response = "The sum of 2 and 3 is 5."
    responses = [
        "Judge 1: The response is correct.",
        "Judge 2: The response is accurate.",
    ]

    prompt = build_ice_score_round_n_prompt(question, response, responses)
    print(prompt)
    # prompt = build_ice_score_round_zero_prompt(question, response)
    # print(prompt)
