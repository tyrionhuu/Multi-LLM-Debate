from typing import Dict, List

NEW_LINE = "\n"

BOOL_JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the passage",
    "answer": "true/false"
}
"""

BOOL_JSON_FORMAT_COT = """
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

BOOL_NON_JSON_FORMAT = """
Reasoning: your reasoning based on the passage
Final Answer: true/false
"""

BOOL_NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your reasoning
Step 2: second step of your reasoning
Step 3: third step of your reasoning
...
Final Answer: true/false
"""


def build_bool_q_round_zero_prompt(
    question: str, passage: str, use_cot: bool = True, json_mode: bool = False
) -> str:
    prompt = "You will be given a true or false question which is based on a passage. "
    if json_mode:
        prompt += "Answer in the following JSON format:" + NEW_LINE
        prompt += BOOL_JSON_FORMAT_COT if use_cot else BOOL_JSON_FORMAT
    else:
        prompt += "Answer in the following format:" + NEW_LINE
        prompt += BOOL_NON_JSON_FORMAT_COT if use_cot else BOOL_NON_JSON_FORMAT
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
        prompt += BOOL_JSON_FORMAT_COT if use_cot else BOOL_JSON_FORMAT
    else:
        prompt += "Answer in the following format:" + NEW_LINE
        prompt += BOOL_NON_JSON_FORMAT_COT if use_cot else BOOL_NON_JSON_FORMAT
    prompt += NEW_LINE
    prompt += "Question: " + question + NEW_LINE
    prompt += "Passage: " + passage

    return prompt
