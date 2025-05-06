from typing import Dict, List

# Constants
NEW_LINE = "\n"
DIVIDER = "#" * 80

# Response format templates
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

SCORING_CRITERIA = """
Assign a single integer score from **1** to **5** based on the overall performance of the output. 
Each score level is described in detail below to guide the evaluation process.

1: The output fails to meet basic expectations. It is largely ineffective, significantly 
    flawed, and does not serve its intended purpose.
    
2: The output meets minimal standards but has considerable deficiencies. It partially 
    serves its purpose but requires substantial improvement.
    
3: The output adequately meets the basic requirements. It functions as intended but lacks 
    distinction and contains some areas needing enhancement.
    
4: The output effectively meets the expectations with minor areas for improvement. It is 
    well-executed and serves its purpose reliably.
    
5: The output surpasses expectations, demonstrating outstanding effectiveness, excellence, 
    and suitability. It is exemplary in fulfilling its intended purpose.
"""

def build_judge_anything_score_round_zero_prompt(
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
    prompt = (
        "Please act as an impartial judge and evaluate the quality of the "
        "response from the AI assistant to user's instruction. "
        "You need to provide a holistic assessment of the generated output by "
        "evaluating its general effectiveness, excellence, and suitability for "
        "the intended purpose. It reflects the cumulative performance of the "
        "output across various dimensions without delving into specific aspects, "
        "allowing for a comprehensive and integrated evaluation. "
        "You will be provided with a question, the response from the AI assistant, "
        "and an image. Your task is to score the response on a scale of 1 to 5, "
        f"based on the following criteria:{NEW_LINE}{SCORING_CRITERIA}"
    )

    if json_mode:
        prompt += (
            "You MUST answer in the following JSON format (x is an integer from 1 to 5):"
            + NEW_LINE
        )
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value must be only a number between 1 and 5. "
            + "Do not include any other text after 'Final Answer: x'."
            + NEW_LINE
        )
    else:
        prompt += (
            "You MUST answer in the following format (x is an integer from 1 to 5):"
            + NEW_LINE
        )
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer: ' MUST be placed at the end of your response, "
            + "and the value must be only a number between 1 and 5. "
            + "Do not include any other text after 'Final Answer: x'."
            + NEW_LINE
        )

    prompt += DIVIDER + NEW_LINE
    prompt += "[Instruction]" + NEW_LINE
    prompt += question + NEW_LINE
    prompt += "[The Start of Assistant's Response]" + NEW_LINE
    prompt += response + NEW_LINE
    prompt += "[The End of Assistant's Response]" + NEW_LINE
    prompt += NEW_LINE + "Your answer:" + NEW_LINE

    return prompt


def build_judge_anything_score_round_n_prompt(
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
    prompt = (
        "Several other judges have provided evaluations of an AI assistant's response "
        "to the instruction and an image given by the user. "
        "Below are their evaluations: " + NEW_LINE
    )

    prompt += DIVIDER + NEW_LINE
    for i, response in enumerate(responses, 1):
        prompt += f"Judge {i}: {response}" + NEW_LINE

    prompt += DIVIDER + NEW_LINE

    prompt += (
        "Please act as an impartial judge and evaluate the quality of the "
        "response from the AI assistant to user's instruction. "
        "You need to provide a holistic assessment of the generated output by "
        "evaluating its general effectiveness, excellence, and suitability for "
        "the intended purpose. It reflects the cumulative performance of the "
        "output across various dimensions without delving into specific aspects, "
        "allowing for a comprehensive and integrated evaluation. "
        "You will be provided with a question, the response from the AI assistant, "
        "and an image. Your task is to score the response on a scale of 1 to 5, "
        f"based on the following criteria:{NEW_LINE}{SCORING_CRITERIA}"
    )

    if json_mode:
        prompt += (
            "You MUST answer in the following JSON format (x is an integer from 1 to 5):"
            + NEW_LINE
        )
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer' MUST be placed at the end of your response, "
            + "and the value must be only a number between 1 and 5. "
            + "Do not include any other text after 'Final Answer: x'."
            + NEW_LINE
        )
    else:
        prompt += (
            "You MUST answer in the following format (x is an integer from 1 to 5):"
            + NEW_LINE
        )
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
        prompt += (
            NEW_LINE
            + "Note that the 'Final Answer: ' MUST be placed at the end of your response, "
            + "and the value must be only a number between 1 and 5. "
            + "Do not include any other text after 'Final Answer: x'."
            + NEW_LINE
        )
    prompt += DIVIDER + NEW_LINE
    prompt += "[Instruction]" + NEW_LINE
    prompt += question + NEW_LINE
    prompt += "[The Start of Assistant's Response]" + NEW_LINE
    prompt += response + NEW_LINE
    prompt += "[The End of Assistant's Response]" + NEW_LINE
    prompt += NEW_LINE + "Your answer:" + NEW_LINE
    return prompt
