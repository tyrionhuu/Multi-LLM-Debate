from typing import Dict, List

NEW_LINE = "\n"
DIVIDER = "#" * 80
JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the passage",
    "Final Answer": "[x, y, ...]"
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
    "Final Answer": "[x, y, ...]"
}
"""

NON_JSON_FORMAT = """
Reasoning: your reasoning based on the passage
Final Answer: [x, y, ...]
"""

NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your reasoning
Step 2: second step of your reasoning
Step 3: third step of your reasoning
...
Final Answer: [x, y, ...]
"""


def build_prm800k_round_zero_prompt(
    question: str,
    steps: List[str],
    json_mode: bool = False,
    use_cot: bool = True,
) -> str:
    """Build prompt for the initial round of PRM800K evaluation.

    Args:
        question: The user question to be evaluated
        steps: List of reasoning steps to be included in the prompt

    Returns:
        str: The formatted prompt for PRM800K evaluation
    """
    prompt = (
        "As an assistant, your task is to serve as an impartial response judge.\n"
        + NEW_LINE
    )

    prompt += (
        "You will be given an input and a question and a step by step response from an AI assistant.\n"
        "Your task is to evaluate each step and give each step a rating.\n"
        "If you think a step is correct, give it a 1; if you think a step is wrong, give it a -1.\n"
        "No other values are allowed.\n"
        "You should output your final answer in the format: 'Final Answer: [x,y,...]'.\n"
        "And x,y,... should be integers, where each value is either 1 or -1, corresponding to the steps provided.\n"
    ) + NEW_LINE

    if json_mode:
        prompt += "You MUST output your response in JSON format.\n"
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value [x,y,...] must be a list of integers, where each value is either 1 or -1.\n"
            + "Do not include any other text after 'Final Answer: [x, y, ...]'."
            + NEW_LINE
        )
    else:
        prompt += "You MUST output your response in the following format:\n"
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value [x,y,...] must be a list of integers, where each value is either 1 or -1.\n"
            + "Do not include any other text after 'Final Answer: [x, y, ...]'."
            + NEW_LINE
        )
    prompt += DIVIDER + NEW_LINE

    prompt += "[Question]\n"
    prompt += question + NEW_LINE + DIVIDER + NEW_LINE
    prompt += "[Steps]\n"
    for i, step in enumerate(steps):
        prompt += f"{i + 1}. {step}\n"
    prompt += DIVIDER + NEW_LINE

    prompt += "Your answer:\n"
    return prompt
def build_prm800k_round_n_prompt(
    question: str,
    steps: List[str],
    responses: List[str | Dict],
    json_mode: bool = False,
    use_cot: bool = True,
) -> str:
    """Build prompt for subsequent rounds of PRM800K evaluation.

    Args:
        question: The user question to be evaluated
        steps: List of reasoning steps to be included in the prompt
        responses: Previous responses from judge models

    Returns:
        str: The formatted prompt for PRM800K evaluation
    """
    prompt = (
        "As an assistant, your task is to serve as an impartial response judge.\n"
        + NEW_LINE
    )
    
    prompt += (
        "Several other judges have provided evaluations of an AI assistant's response. "
        "Review their assessments and provide your own independent evaluation.\n"
    ) + NEW_LINE
    
    prompt += (
        "You will be given an input and a question and a step by step response from an AI assistant.\n"
        "Your task is to evaluate each step and give each step a rating.\n"
        "If you think a step is correct, give it a 1; if you think a step is wrong, give it a -1.\n"
        "No other values are allowed.\n"
        "You should output your final answer in the format: 'Final Answer: [x,y,...]'.\n"
        "And x,y,... should be integers, where each value is either 1 or -1, corresponding to the steps provided.\n"
    ) + NEW_LINE

    if json_mode:
        prompt += "You MUST output your response in JSON format.\n"
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value [x,y,...] must be a list of integers, where each value is either 1 or -1.\n"
            + "Do not include any other text after 'Final Answer: [x, y, ...]'."
            + NEW_LINE
        )
    else:
        prompt += "You MUST output your response in the following format:\n"
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value [x,y,...] must be a list of integers, where each value is either 1 or -1.\n"
            + "Do not include any other text after 'Final Answer: [x, y, ...]'."
            + NEW_LINE
        )
    prompt += DIVIDER + NEW_LINE
    
    prompt += "Previous judge responses:\n"
    for i, judge_response in enumerate(responses, 1):
        prompt += f"Judge {i} Response:" + NEW_LINE
        prompt += judge_response + NEW_LINE
        
    prompt += DIVIDER + NEW_LINE
    prompt += "[Question]\n"
    prompt += question + NEW_LINE + DIVIDER + NEW_LINE
    prompt += "[Steps]\n"
    for i, step in enumerate(steps):
        prompt += f"{i + 1}. {step}\n"
    prompt += DIVIDER + NEW_LINE
    prompt += "Your answer:\n"
    return prompt
if __name__ == "__main__":
    # Example usage
    question = "What is the capital of France?"
    steps = [
        "Identify the country in question.",
        "Recall the capital city of that country.",
        "Confirm the answer based on knowledge.",
    ]

    prompt = build_prm800k_round_n_prompt(question, steps, responses=[])
    print(prompt)  # This will print the formatted prompt for PRM800K evaluation.
