from typing import List

NEW_LINE = "\n"
DIVIDER = "#" * 80

JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the passage",
    "Final Answer": "relevance - x; specificity - x; interestingness - x; understandability - x; overall - x"
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
    "Final Answer": "relevance - x; specificity - x; interestingness - x; understandability - x; overall - x"
}
"""
NON_JSON_FORMAT = """
Reasoning: your reasoning based on the passage
Final Answer: relevance - x; specificity - x; interestingness - x; understandability - x; overall - x
"""
NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your reasoning
Step 2: second step of your reasoning
Step 3: third step of your reasoning
...
Final Answer: relevance - x; specificity - x; interestingness - x; understandability - x; overall - x
"""


def build_comp_analysis_round_zero_prompt(
    context: str,
    response: str,
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for the initial round of comparison analysis.

    Args:
        context: The context to be evaluated
        response: The response to be evaluated
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format

    Returns:
        str: The formatted prompt for comparison analysis
    """
    prompt = (
        "Please act as an impartial judge and analyze the quality of the "
        "response provided based on the given context. "
        "Please use the CONTEXT and RESPONSE to rate the context relevance, "
        "specificity, interestingness, understandability, "
        "and overall quality of the response on a scale of 1 to 5"
    )

    if use_cot:
        prompt += NEW_LINE + "Please provide your reasoning step-by-step."

    prompt += NEW_LINE + DIVIDER + NEW_LINE
    prompt += (
        "Context: {context}"
        + NEW_LINE
        + "Response: {response}"
        + NEW_LINE
        + DIVIDER
        + NEW_LINE
    )

    if json_mode:
        prompt += (
            "You MUST return your response in the following JSON format: " + NEW_LINE
        )
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
    else:
        prompt += "You MUST return your response in the following format: " + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT

    prompt += DIVIDER + NEW_LINE + "Your response: "
    return prompt.format(context=context, response=response)


def build_comp_analysis_round_n_prompt(
    context: str,
    response: str,
    responses: List[str],
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for subsequent rounds of comparison analysis.

    Args:
        context: The context to be evaluated
        response: The response to be evaluated
        responses: List of previous responses for context
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format

    Returns:
        str: The formatted prompt for comparison analysis
    """
    prompt = (
        "Several other judges have provided evaluations of two AI assistant "
        "responses to a given context. Below are their evaluations: " + NEW_LINE
    )

    prompt += DIVIDER + NEW_LINE
    for i, _response in enumerate(responses, 1):
        prompt += f"Judge {i} response: {_response}" + NEW_LINE

    prompt += DIVIDER + NEW_LINE

    prompt += (
        "Please act as an impartial evaluator and analyze the quality of the "
        "response provided based on the given context and responses from other judges. "
        "Please use the CONTEXT, RESPONSE, and PREVIOUS RESPONSES to rate the "
        "context relevance, specificity, interestingness, understandability, "
        "and overall quality of the response on a scale of 1 to 5. "
    )
    prompt += NEW_LINE
    prompt += (
        "Avoid any position biases and ensure that the order in "
        "which the responses were presented does not influence your decision. "
        "Do not allow the length of the responses to influence your evaluation. "
        "Do not favor certain names of the assistants. Be as objective as "
        "possible. Be concise in your reasoning. " + NEW_LINE
    )

    if json_mode:
        prompt += (
            "You MUST return your response in the following JSON format: " + NEW_LINE
        )
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
    else:
        prompt += "You MUST return your response in the following format: " + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT

    prompt += DIVIDER + NEW_LINE

    prompt += "Context: {context}" + NEW_LINE
    prompt += "Response: {response}" + NEW_LINE
    prompt += DIVIDER + NEW_LINE
    prompt += "Your response: "
    return prompt.format(context=context, response=response)


if __name__ == "__main__":
    print(
        build_comp_analysis_round_n_prompt(
            context="This is a sample context for evaluation.",
            response="This is a sample response to be evaluated.",
            responses=[
                "Judge 1: The response was clear and relevant.",
                "Judge 2: The response lacked specificity but was interesting.",
                "Judge 3: The response was understandable but not very engaging.",
            ],
            use_cot=True,
            json_mode=False,
        )
    )
