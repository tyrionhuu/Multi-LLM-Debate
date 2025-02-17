NEW_LINE = "\n"
def build_bool_round_zero_prompt(question: str, passage: str) -> str:
    prompt = "You will be given a yes-no question which is based on a passage. "
    prompt += "You should use the passage to help you answer the question. "
    prompt += "You should give a brief justification for your answer, and you must provide a final answer of either Yes or No. "
    prompt += NEW_LINE
    prompt += "Question: " + question
    prompt += NEW_LINE
    prompt += "Passage: " + passage
    
    return prompt

def build_bool_round_n_prompt(question: str, passage: str, responses: list[str]) -> str:
    prompt = "Several other models have provided responses to a yes-no question, below are their responses: "
    prompt += NEW_LINE
    
    for i, response in enumerate(responses, 1):
        prompt += f"Model {i}: {response}"
        prompt += NEW_LINE
    
    prompt += "You should consider these responses when answering the following yes-no question which is based on a passage. "
    prompt += "You should use the given responses and the passage to help you answer the question. "
    prompt += "You should give a brief justification for your answer, and you must provide a final answer of either Yes or No."
    prompt += NEW_LINE
    prompt += "Question: " + question
    prompt += NEW_LINE
    prompt += "Passage: " + passage
    
    return prompt