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


def build_comp_analysis_round_zero_prompt(
    input: str,
    response: str,
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for the initial round of judge evaluation.

    Args:
        input: The input context for the evaluation
        response: Response from the AI assistant
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format

    Returns:
        str: The formatted prompt for judge evaluation
    """
    prompt = "As an assistant, your task is to serve as a response judge.\n" + NEW_LINE

    prompt += (
        "You will be given an input and a response from an AI assistant.\n"
        "Your task is to rate the response based on the input.\n"
        "You have to rate the response on a scale of 1 to 5, where:\n"
        "1 - very poor, 2 - poor, 3 - average, 4 - good, 5 - excellent.\n"
        "You should consider the context relevance, specificity, interestingness, understandability, and overall quality of the response.\n"
        "You should output your final answer in the format: 'Final Answer: x'.\n"
        "And x should be an integer between 1 and 5.\n"
    ) + NEW_LINE

    if json_mode:
        prompt += "You MUST output your response in JSON format.\n"
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            +"Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value x must be only an integer between 1 and 5.\n"
            + "Do not include any other text after 'Final Answer: x'."
            + NEW_LINE
        )
    else:
        prompt += "You MUST answer in the following format:" + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value x must be only an integer between 1 and 5.\n"
            + "Do not include any other text after 'Final Answer: x'."
            + NEW_LINE
        )
    prompt += DIVIDER + NEW_LINE

    prompt += "[Input]" + NEW_LINE
    prompt += input + NEW_LINE + DIVIDER + NEW_LINE
    prompt += "[Response]" + NEW_LINE
    prompt += response + NEW_LINE + DIVIDER + NEW_LINE

    prompt += "Your answer:" + NEW_LINE
    return prompt
def build_comp_analysis_round_n_prompt(
    input: str,
    response: str,
    responses: List[str | Dict],
    use_cot: bool = True,
    json_mode: bool = False,
) -> str:
    """Build prompt for subsequent rounds of judge evaluation.

    Args:
        input: The input context for the evaluation
        response: Response from the AI assistant
        responses: Previous responses from judge models
        use_cot: Whether to use chain-of-thought prompting
        json_mode: Whether to return response in JSON format

    Returns:
        str: The formatted prompt for judge evaluation
    """
    prompt = "As an assistant, your task is to serve as a response judge.\n" + NEW_LINE
    
    prompt += (
        "Several other judges have provided evaluations of an AI assistant's response. "
        "Review their assessments and provide your own independent evaluation.\n"
    ) + NEW_LINE
    
    prompt += (
        "You will be given an input and a response from an AI assistant.\n"
        "Your task is to rate the response based on the input.\n"
        "You have to rate the response on a scale of 1 to 5, where:\n"
        "1 - very poor, 2 - poor, 3 - average, 4 - good, 5 - excellent.\n"
        "You should consider the context relevance, specificity, interestingness, understandability, and overall quality of the response.\n"
        "You should output your final answer in the format: 'Final Answer: x'.\n"
        "And x should be an integer between 1 and 5.\n"
    ) + NEW_LINE

    if json_mode:
        prompt += "You MUST output your response in JSON format.\n"
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value x must be only an integer between 1 and 5.\n"
            + "Do not include any other text after 'Final Answer: x'."
            + NEW_LINE
        )
    else:
        prompt += "You MUST answer in the following format:" + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value x must be only an integer between 1 and 5.\n"
            + "Do not include any other text after 'Final Answer: x'."
            + NEW_LINE
        )
    prompt += DIVIDER + NEW_LINE
    
    prompt += "Previous judge responses:\n"
    for i, judge_response in enumerate(responses, 1):
        prompt += f"Judge {i} Response:" + NEW_LINE
        prompt += judge_response + NEW_LINE 
        
    prompt += DIVIDER + NEW_LINE
    prompt += "[Input]" + NEW_LINE
    
    prompt += input + NEW_LINE + DIVIDER + NEW_LINE
    prompt += "[Response]" + NEW_LINE
    
    prompt += response + NEW_LINE + DIVIDER + NEW_LINE
    prompt += "Your answer:" + NEW_LINE
    return prompt

if __name__ == "__main__":
    # Example usage
    input_text = "This is an example input for the judge evaluation."
    response_text = "This is an example response from the AI assistant."
    previous_responses = [
        "Judge 1: The response is relevant and specific.",
        "Judge 2: The response is interesting but lacks clarity.",
    ]
    prompt = build_comp_analysis_round_n_prompt(
        input=input_text,
        response=response_text,
        responses=previous_responses,
        use_cot=True,
        json_mode=True
    )
    print(prompt)