from typing import List
NEW_LINE = "\n"


def build_bool_q_round_zero_prompt(
    question: str, passage: str, use_cot: bool = True
) -> str:
    prompt = "You will be given a yes-no question which is based on a passage. "
    prompt += "Answer in the following format:" + NEW_LINE
    if use_cot:
        prompt += (
            "Reasoning: [Your step-by-step reasoning based on the passage]" + NEW_LINE
        )
    else:
        prompt += "Reasoning: [Your reasoning based on the passage]" + NEW_LINE
    prompt += "Final Answer: [Yes/No]" + NEW_LINE + NEW_LINE
    prompt += "Question: " + question + NEW_LINE
    prompt += "Passage: " + passage

    return prompt


def build_bool_q_round_n_prompt(
    question: str, passage: str, responses: List[str], use_cot: bool = True
) -> str:
    prompt = (
        "Several other models have provided responses to a yes-no question, below are their responses: "
        + NEW_LINE
    )

    for i, response in enumerate(responses, 1):
        prompt += f"Model {i}: {response}" + NEW_LINE

    prompt += NEW_LINE
    prompt += (
        "Consider these responses when answering the following yes-no question."
        + NEW_LINE
    )
    prompt += "Answer in the following format:" + NEW_LINE
    if use_cot:
        prompt += (
            "Reasoning: [Your step-by-step reasoning based on the passage and other responses]"
            + NEW_LINE
        )
    else:
        prompt += (
            "Reasoning: [Your reasoning based on the passage and other responses]"
            + NEW_LINE
        )
    prompt += "Final Answer: [Yes/No]" + NEW_LINE + NEW_LINE
    prompt += "Question: " + question + NEW_LINE
    prompt += "Passage: " + passage

    return prompt
