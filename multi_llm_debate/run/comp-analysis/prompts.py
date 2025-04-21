from typing import Dict, List

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
        "Please act as an impartial evaluator and analyze the quality of the "
        "response provided based on the given context. "
        "Your evaluation should consider relevance, specificity, interestingness, "
        "understandability, and overall quality."
    )
    
    if use_cot:
        prompt += NEW_LINE + "Please provide your reasoning step-by-step."
    
    prompt += NEW_LINE + DIVIDER + NEW_LINE
    prompt += "Context: {context}" + NEW_LINE + "Response: {response}" + NEW_LINE + DIVIDER + NEW_LINE
    
    if json_mode:
        prompt += "You MUST return your response in the following JSON format: " + NEW_LINE
        prompt += JSON_FORMAT_COT if use_cot else JSON_FORMAT
    else:
        prompt += "You MUST return your response in the following format: " + NEW_LINE
        prompt += NON_JSON_FORMAT_COT if use_cot else NON_JSON_FORMAT
    
    prompt += NEW_LINE + "Your response: "
    return prompt.format(context=context, response=response)