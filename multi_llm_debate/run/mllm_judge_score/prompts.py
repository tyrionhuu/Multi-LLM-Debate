"""Module for generating prompts for multi-LLM judge scoring system."""

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
Poor (1): The response significantly deviates from the user's instruction and fails \
to address the query effectively. It shows a lack of relevance, accuracy, and \
comprehensiveness. Creativity and granularity are absent or poorly executed.

Fair (2): The response addresses the user's instruction partially, with evident \
shortcomings in relevance, accuracy, or comprehensiveness. It lacks depth in \
creativity and granularity, indicating a superficial understanding.

Average (3): The response adequately addresses the user's instruction, showing a \
fair level of relevance, accuracy, and comprehensiveness. It reflects basic \
creativity and granularity but may lack sophistication.

Good (4): The response is well-aligned with the user's instruction, demonstrating \
high relevance, accuracy, and comprehensiveness. Shows creativity and nuanced \
understanding with detailed granularity.

Excellent (5): The response perfectly adheres to the user's instruction, excelling \
in all criteria. Provides an insightful, detailed, and thorough answer with deep \
understanding.
"""


def build_mllm_judge_score_round_zero_prompt(
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
        " response from the AI assistant to user's instruction. "
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


def build_mllm_judge_score_round_n_prompt(
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
        "single response from the AI assistant to user's instruction. "
        "Consider the previous evaluations provided by other judges, "
        "but make your own independent assessment. "
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
