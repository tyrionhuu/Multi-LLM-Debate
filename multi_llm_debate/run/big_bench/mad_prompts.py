from typing import Dict, List

NEW_LINE = "\n"
DIVIDER = "#" * 80

# JSON format for Big Bench MAD responses
JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the debate",
    "Final Answer": "Response 1"
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
    "Final Answer": "Response 1"
}
"""

NON_JSON_FORMAT = """
Reasoning: your reasoning based on the debate
Final Answer: Response 1
"""

NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your reasoning
Step 2: second step of your reasoning
Step 3: third step of your reasoning
...
Final Answer: Response 1
"""


def build_big_bench_mad_player_meta_prompt(debate_topic: str) -> str:
    """Build meta prompt for Big Bench MAD debate players.
    
    Args:
        debate_topic: The debate topic with question and responses
        
    Returns:
        str: The formatted meta prompt for debate players
    """
    prompt = "As an assistant, your task is to serve as a debater in a structured debate.\n" + NEW_LINE

    prompt += (
        "You will evaluate whether a statement is plausible or implausible by debating between two responses. "
        "You should choose the response that correctly determines if the statement is plausible (Response 1) or implausible (Response 2).\n"
    ) + NEW_LINE

    prompt += (
        "Avoid any position biases and ensure that the order in which the responses were presented "
        "does not influence your decision. Do not allow the length of the responses to influence your evaluation. "
        "Be as objective as possible.\n"
    ) + NEW_LINE

    prompt += "The debate topic is:\n"
    prompt += debate_topic + NEW_LINE + NEW_LINE

    prompt += (
        "**Debate Instructions:**\n"
        "- Analyze both responses carefully and objectively\n"
        "- Consider whether each response correctly identifies plausibility\n"
        "- Provide clear reasoning for your position\n"
        "- Engage constructively with opposing arguments\n"
        "- Your final choice must be either 'Response 1 (Yes)' or 'Response 2 (No)'\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Plausibility Assessment: Does the response correctly evaluate if the statement is plausible?\n"
        "- Reasoning Quality: How well-reasoned is the response?\n"
        "- Evidence: Does the response provide appropriate evidence?\n"
        "- Logic: How logical and coherent is the reasoning?\n"
        "- Accuracy: Is the final determination correct?\n"
    )

    return prompt


def build_big_bench_mad_moderator_meta_prompt(debate_topic: str) -> str:
    """Build meta prompt for Big Bench MAD debate moderator.
    
    Args:
        debate_topic: The debate topic with question and responses
        
    Returns:
        str: The formatted meta prompt for debate moderator
    """
    prompt = "As an assistant, your task is to serve as a moderator in a structured debate.\n" + NEW_LINE

    prompt += (
        "You will evaluate a debate between two responses about whether a statement is plausible or implausible. "
        "You should choose the response that correctly determines if the statement is plausible (Response 1) or implausible (Response 2).\n"
    ) + NEW_LINE

    prompt += (
        "Avoid any position biases and ensure that the order in which the responses were presented "
        "does not influence your decision. Do not allow the length of the responses to influence your evaluation. "
        "Be as objective as possible.\n"
    ) + NEW_LINE

    prompt += "The debate topic is:\n"
    prompt += debate_topic + NEW_LINE + NEW_LINE

    prompt += (
        "**Moderator Instructions:**\n"
        "- Listen carefully to both sides of the debate\n"
        "- Evaluate the quality of arguments presented\n"
        "- Consider the evidence and reasoning provided\n"
        "- Make an objective assessment based on the debate\n"
        "- At the end of each round, you will evaluate both sides and decide which response is correct\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Plausibility Assessment: Which response correctly evaluates if the statement is plausible?\n"
        "- Argument Quality: How well-reasoned and supported are the arguments?\n"
        "- Evidence: How much relevant evidence is presented?\n"
        "- Logic: How logical and coherent is the reasoning?\n"
        "- Conclusion: Which response ultimately provides the correct determination?\n"
    )

    return prompt


def build_big_bench_mad_affirmative_prompt(debate_topic: str) -> str:
    """Build affirmative prompt for Big Bench MAD debate.
    
    Args:
        debate_topic: The debate topic with question and responses
        
    Returns:
        str: The formatted affirmative prompt
    """
    prompt = debate_topic + NEW_LINE + NEW_LINE

    prompt += "**Your Role:** You are the affirmative debater arguing for one of the two responses.\n" + NEW_LINE

    prompt += "**Task:** Analyze both responses carefully and argue for which one correctly determines if the statement is plausible.\n" + NEW_LINE

    prompt += (
        "**Instructions:**\n"
        "1. Read and understand the statement and both responses\n"
        "2. Evaluate each response based on:\n"
        "   - Plausibility assessment accuracy\n"
        "   - Reasoning quality and logic\n"
        "   - Evidence provided\n"
        "   - Clarity of determination\n"
        "   - Overall correctness\n"
        "3. Present your argument for which response is superior\n"
        "4. Provide specific reasons and evidence for your choice\n"
        "5. Address potential counterarguments\n"
    ) + NEW_LINE

    prompt += "**IMPORTANT:** You must choose between 'Response 1 (Yes)' and 'Response 2 (No)' and provide clear reasoning for your choice."

    return prompt


def build_big_bench_mad_negative_prompt(aff_ans: str) -> str:
    """Build negative prompt for Big Bench MAD debate.
    
    Args:
        aff_ans: The affirmative side's argument
        
    Returns:
        str: The formatted negative prompt
    """
    prompt = aff_ans + NEW_LINE + NEW_LINE

    prompt += "**Your Role:** You are the negative debater who disagrees with the affirmative side's position.\n" + NEW_LINE

    prompt += "**Task:** Provide your own analysis and reasoning for which response correctly determines if the statement is plausible.\n" + NEW_LINE

    prompt += (
        "**Instructions:**\n"
        "1. Review the affirmative side's argument carefully\n"
        "2. Consider their reasoning and evidence\n"
        "3. Present your own independent analysis of both responses\n"
        "4. Argue for your preferred choice with specific reasons\n"
        "5. Address any weaknesses in the affirmative argument\n"
        "6. Provide counter-evidence or alternative perspectives\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Plausibility Assessment: Which response correctly evaluates if the statement is plausible?\n"
        "- Reasoning Quality: How well-reasoned is each response?\n"
        "- Evidence: Does each response provide appropriate evidence?\n"
        "- Logic: How logical and coherent is the reasoning?\n"
        "- Accuracy: Is the final determination correct?\n"
    ) + NEW_LINE

    prompt += "**IMPORTANT:** You must choose between 'Response 1 (Yes)' and 'Response 2 (No)' and provide clear reasoning for your choice."

    return prompt


def build_big_bench_mad_moderator_prompt(aff_ans: str, neg_ans: str, round_name: str) -> str:
    """Build moderator prompt for Big Bench MAD debate.
    
    Args:
        aff_ans: The affirmative side's argument
        neg_ans: The negative side's argument
        round_name: The name of the current round
        
    Returns:
        str: The formatted moderator prompt
    """
    prompt = f"Now the {round_name} round of debate for both sides has ended.\n" + NEW_LINE

    prompt += "**Affirmative side arguing:**\n"
    prompt += aff_ans + NEW_LINE + NEW_LINE

    prompt += "**Negative side arguing:**\n"
    prompt += neg_ans + NEW_LINE + NEW_LINE

    prompt += "**Your Role:** As the moderator, evaluate both sides' arguments and determine which response correctly identifies if the statement is plausible.\n" + NEW_LINE

    prompt += (
        "**Evaluation Instructions:**\n"
        "1. Review both sides' arguments carefully\n"
        "2. Consider the quality of reasoning presented\n"
        "3. Evaluate the evidence and examples provided\n"
        "4. Assess the logical coherence of each position\n"
        "5. Determine which response ultimately provides the correct determination\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Plausibility Assessment: Which response correctly evaluates if the statement is plausible?\n"
        "- Argument Quality: How well-reasoned and supported are the arguments?\n"
        "- Evidence: How much relevant evidence is presented?\n"
        "- Logic: How logical and coherent is the reasoning?\n"
        "- Conclusion: Which response ultimately provides the correct determination?\n"
    ) + NEW_LINE

    prompt += "You MUST answer in the following JSON format:\n"
    prompt += JSON_FORMAT + NEW_LINE

    prompt += (
        "**Note:** The 'Final Answer' MUST be placed at the end of your response, "
        "and the value must be only 'Response 1' or 'Response 2'. "
        "Do not include any other text after the JSON response."
    )

    return prompt


def build_big_bench_mad_judge_prompt_1(aff_ans: str, neg_ans: str) -> str:
    """Build first judge prompt for Big Bench MAD debate.
    
    Args:
        aff_ans: The affirmative side's argument
        neg_ans: The negative side's argument
        
    Returns:
        str: The formatted first judge prompt
    """
    prompt = "**Affirmative side arguing:** " + aff_ans + NEW_LINE + NEW_LINE

    prompt += "**Negative side arguing:** " + neg_ans + NEW_LINE + NEW_LINE

    prompt += "**Task:** Summarize the key arguments and answer candidates presented in this debate.\n" + NEW_LINE

    prompt += (
        "**Instructions:**\n"
        "- Review both sides' arguments\n"
        "- Identify the main points of contention\n"
        "- Present the answer candidates without providing reasons\n"
        "- Focus on clarity and objectivity\n"
    ) + NEW_LINE

    prompt += "**Now, what answer candidates do we have? Present them without reasons.**"

    return prompt


def build_big_bench_mad_judge_prompt_2(debate_topic: str) -> str:
    """Build second judge prompt for Big Bench MAD debate.
    
    Args:
        debate_topic: The debate topic with question and responses
        
    Returns:
        str: The formatted second judge prompt
    """
    prompt = "**Therefore, " + debate_topic + "\n" + NEW_LINE

    prompt += "**Your Role:** As the final judge, you must make the ultimate decision based on the debate.\n" + NEW_LINE

    prompt += "**Task:** Summarize your reasons and give the final answer that you think is correct.\n" + NEW_LINE

    prompt += (
        "**Evaluation Instructions:**\n"
        "1. Review the entire debate and all arguments presented\n"
        "2. Consider the quality of reasoning from both sides\n"
        "3. Evaluate the evidence and examples provided\n"
        "4. Assess which response correctly determines if the statement is plausible\n"
        "5. Make your final determination\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Plausibility Assessment: Which response correctly evaluates if the statement is plausible?\n"
        "- Reasoning Quality: How well-reasoned is each response?\n"
        "- Evidence: Does each response provide appropriate evidence?\n"
        "- Logic: How logical and coherent is the reasoning?\n"
        "- Accuracy: Is the final determination correct?\n"
    ) + NEW_LINE

    prompt += "Please summarize your reasons and give the final answer that you think is correct.\n" + NEW_LINE

    prompt += "You MUST answer in the following JSON format:\n"
    prompt += JSON_FORMAT + NEW_LINE

    prompt += (
        "**Note:** The 'Final Answer' MUST be placed at the end of your response, "
        "and the value must be only 'Response 1' or 'Response 2'. "
        "Do not include any other text after the JSON response."
    )

    return prompt


def build_big_bench_mad_debate_prompt(oppo_ans: str) -> str:
    """Build debate prompt for Big Bench MAD debate.
    
    Args:
        oppo_ans: The opposing side's argument
        
    Returns:
        str: The formatted debate prompt
    """
    prompt = oppo_ans + NEW_LINE + NEW_LINE

    prompt += "**Your Role:** You are continuing the debate with the opposing side.\n" + NEW_LINE

    prompt += "**Task:** Respond to the opposing argument and provide your own perspective.\n" + NEW_LINE

    prompt += (
        "**Instructions:**\n"
        "1. Read and understand the opposing argument carefully\n"
        "2. Consider their reasoning and evidence\n"
        "3. Do you agree with their perspective? Why or why not?\n"
        "4. Provide your own analysis and reasoning\n"
        "5. Support your position with specific evidence or examples\n"
        "6. Address any weaknesses in the opposing argument\n"
    ) + NEW_LINE

    prompt += "**Please provide your reasons and answer.**"

    return prompt


def build_big_bench_mad_prompts(debate_topic: str) -> Dict[str, str]:
    """Build all Big Bench MAD prompts.
    
    Args:
        debate_topic: The debate topic with question and responses
        
    Returns:
        Dict containing all MAD prompts for Big Bench
    """
    return {
        "player_meta_prompt": build_big_bench_mad_player_meta_prompt(debate_topic),
        "moderator_meta_prompt": build_big_bench_mad_moderator_meta_prompt(debate_topic),
        "affirmative_prompt": build_big_bench_mad_affirmative_prompt(debate_topic),
        "negative_prompt": build_big_bench_mad_negative_prompt("##aff_ans##"),  # Placeholder
        "moderator_prompt": build_big_bench_mad_moderator_prompt("##aff_ans##", "##neg_ans##", "##round##"),  # Placeholders
        "judge_prompt_last1": build_big_bench_mad_judge_prompt_1("##aff_ans##", "##neg_ans##"),  # Placeholders
        "judge_prompt_last2": build_big_bench_mad_judge_prompt_2(debate_topic),
        "debate_prompt": build_big_bench_mad_debate_prompt("##oppo_ans##"),  # Placeholder
    } 