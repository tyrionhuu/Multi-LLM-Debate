from typing import Dict, List

NEW_LINE = "\n"
DIVIDER = "#" * 80

# JSON format for MLLM Judge Pair MAD responses
JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the debate",
    "Final Answer": "Response A"
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
    "Final Answer": "Response A"
}
"""

NON_JSON_FORMAT = """
Reasoning: your reasoning based on the debate
Final Answer: Response A
"""

NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your reasoning
Step 2: second step of your reasoning
Step 3: third step of your reasoning
...
Final Answer: Response A
"""


def build_mllm_judge_pair_mad_player_meta_prompt(debate_topic: str) -> str:
    """Build meta prompt for MLLM Judge Pair MAD debate players.

    Args:
        debate_topic: The debate topic with question and responses

    Returns:
        str: The formatted meta prompt for debate players
    """
    prompt = (
        "As an assistant, your task is to serve as a debater in a structured debate.\n"
        + NEW_LINE
    )

    prompt += (
        "You will evaluate the quality of responses provided by two AI assistants to a user question and an image. "
        "You should choose the assistant that follows the user's instructions and answers the user's question better.\n"
    ) + NEW_LINE

    prompt += (
        "Avoid any position biases and ensure that the order in which the responses were presented "
        "does not influence your decision. Do not allow the length of the responses to influence your evaluation. "
        "Do not favor certain names of the assistants. Be as objective as possible.\n"
    ) + NEW_LINE

    prompt += "The debate topic is:\n"
    prompt += debate_topic + NEW_LINE + NEW_LINE

    prompt += (
        "**Debate Instructions:**\n"
        "- Analyze both responses carefully and objectively\n"
        "- Consider factors such as accuracy, completeness, relevance, and helpfulness\n"
        "- Pay special attention to how well each response addresses the visual content\n"
        "- Provide clear reasoning for your position\n"
        "- Engage constructively with opposing arguments\n"
        "- Your final choice must be either 'Response A' or 'Response B'\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Visual Understanding: How well does the response interpret the image?\n"
        "- Accuracy: How factually correct is the response?\n"
        "- Completeness: Does the response address all aspects of the question and image?\n"
        "- Relevance: How well does the response relate to the question and visual content?\n"
        "- Helpfulness: How useful and actionable is the response?\n"
        "- Clarity: How clear and understandable is the response?\n"
    )

    return prompt


def build_mllm_judge_pair_mad_moderator_meta_prompt(debate_topic: str) -> str:
    """Build meta prompt for MLLM Judge Pair MAD debate moderator.

    Args:
        debate_topic: The debate topic with question and responses

    Returns:
        str: The formatted meta prompt for debate moderator
    """
    prompt = (
        "As an assistant, your task is to serve as a moderator in a structured debate.\n"
        + NEW_LINE
    )

    prompt += (
        "You will evaluate a debate between two AI assistant responses to a user question and an image. "
        "You should choose the assistant that follows the user's instructions and answers the user's question better.\n"
    ) + NEW_LINE

    prompt += (
        "Avoid any position biases and ensure that the order in which the responses were presented "
        "does not influence your decision. Do not allow the length of the responses to influence your evaluation. "
        "Do not favor certain names of the assistants. Be as objective as possible.\n"
    ) + NEW_LINE

    prompt += "The debate topic is:\n"
    prompt += debate_topic + NEW_LINE + NEW_LINE

    prompt += (
        "**Moderator Instructions:**\n"
        "- Listen carefully to both sides of the debate\n"
        "- Evaluate the quality of arguments presented\n"
        "- Consider how well each side addresses the visual content\n"
        "- Consider the evidence and reasoning provided\n"
        "- Make an objective assessment based on the debate\n"
        "- At the end of each round, you will evaluate both sides and decide which response is correct\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Visual Understanding: Which response better interprets the image?\n"
        "- Argument Quality: How well-reasoned and supported are the arguments?\n"
        "- Evidence: How much relevant evidence is presented?\n"
        "- Logic: How logical and coherent is the reasoning?\n"
        "- Fairness: Are both sides given equal consideration?\n"
        "- Conclusion: Which response ultimately provides the better answer?\n"
    )

    return prompt


def build_mllm_judge_pair_mad_affirmative_prompt(debate_topic: str) -> str:
    """Build affirmative prompt for MLLM Judge Pair MAD debate.

    Args:
        debate_topic: The debate topic with question and responses

    Returns:
        str: The formatted affirmative prompt
    """
    prompt = debate_topic + NEW_LINE + NEW_LINE

    prompt += (
        "**Your Role:** You are the affirmative debater arguing for one of the two responses.\n"
        + NEW_LINE
    )

    prompt += (
        "**Task:** Analyze both responses carefully and argue for which one is better.\n"
        + NEW_LINE
    )

    prompt += (
        "**Instructions:**\n"
        "1. Read and understand the question, image, and both responses\n"
        "2. Evaluate each response based on:\n"
        "   - Visual understanding and interpretation\n"
        "   - Accuracy and factual correctness\n"
        "   - Completeness in addressing the question and image\n"
        "   - Relevance to the topic and visual content\n"
        "   - Helpfulness and usefulness\n"
        "   - Clarity and understandability\n"
        "3. Present your argument for which response is superior\n"
        "4. Provide specific reasons and evidence for your choice\n"
        "5. Address potential counterarguments\n"
    ) + NEW_LINE

    prompt += "**IMPORTANT:** You must choose between 'Response A' and 'Response B' and provide clear reasoning for your choice."

    return prompt


def build_mllm_judge_pair_mad_negative_prompt(aff_ans: str) -> str:
    """Build negative prompt for MLLM Judge Pair MAD debate.

    Args:
        aff_ans: The affirmative side's argument

    Returns:
        str: The formatted negative prompt
    """
    prompt = aff_ans + NEW_LINE + NEW_LINE

    prompt += (
        "**Your Role:** You are the negative debater who disagrees with the affirmative side's position.\n"
        + NEW_LINE
    )

    prompt += (
        "**Task:** Provide your own analysis and reasoning for which response is better.\n"
        + NEW_LINE
    )

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
        "- Visual Understanding: How well does each response interpret the image?\n"
        "- Accuracy: How factually correct is each response?\n"
        "- Completeness: Does each response address all aspects of the question and image?\n"
        "- Relevance: How well does each response relate to the question and visual content?\n"
        "- Helpfulness: How useful and actionable is each response?\n"
        "- Clarity: How clear and understandable is each response?\n"
    ) + NEW_LINE

    prompt += "**IMPORTANT:** You must choose between 'Response A' and 'Response B' and provide clear reasoning for your choice."

    return prompt


def build_mllm_judge_pair_mad_moderator_prompt(
    aff_ans: str, neg_ans: str, round_name: str
) -> str:
    """Build moderator prompt for MLLM Judge Pair MAD debate.

    Args:
        aff_ans: The affirmative side's argument
        neg_ans: The negative side's argument
        round_name: The name of the current round

    Returns:
        str: The formatted moderator prompt
    """
    prompt = (
        f"Now the {round_name} round of debate for both sides has ended.\n" + NEW_LINE
    )

    prompt += "**Affirmative side arguing:**\n"
    prompt += aff_ans + NEW_LINE + NEW_LINE

    prompt += "**Negative side arguing:**\n"
    prompt += neg_ans + NEW_LINE + NEW_LINE

    prompt += (
        "**Your Role:** As the moderator, evaluate both sides' arguments and determine which response is better.\n"
        + NEW_LINE
    )

    prompt += (
        "**Evaluation Instructions:**\n"
        "1. Review both sides' arguments carefully\n"
        "2. Consider the quality of reasoning presented\n"
        "3. Evaluate how well each side addresses the visual content\n"
        "4. Evaluate the evidence and examples provided\n"
        "5. Assess the logical coherence of each position\n"
        "6. Determine which response ultimately provides the better answer\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Visual Understanding: Which response better interprets the image?\n"
        "- Argument Quality: How well-reasoned and supported are the arguments?\n"
        "- Evidence: How much relevant evidence is presented?\n"
        "- Logic: How logical and coherent is the reasoning?\n"
        "- Fairness: Are both sides given equal consideration?\n"
        "- Conclusion: Which response ultimately provides the better answer?\n"
    ) + NEW_LINE

    prompt += "You MUST answer in the following JSON format:\n"
    prompt += JSON_FORMAT + NEW_LINE

    prompt += (
        "**Note:** The 'Final Answer' MUST be placed at the end of your response, "
        "and the value must be only 'Response A' or 'Response B'. "
        "Do not include any other text after the JSON response."
    )

    return prompt


def build_mllm_judge_pair_mad_judge_prompt_1(aff_ans: str, neg_ans: str) -> str:
    """Build first judge prompt for MLLM Judge Pair MAD debate.

    Args:
        aff_ans: The affirmative side's argument
        neg_ans: The negative side's argument

    Returns:
        str: The formatted first judge prompt
    """
    prompt = "**Affirmative side arguing:** " + aff_ans + NEW_LINE + NEW_LINE

    prompt += "**Negative side arguing:** " + neg_ans + NEW_LINE + NEW_LINE

    prompt += (
        "**Task:** Summarize the key arguments and answer candidates presented in this debate.\n"
        + NEW_LINE
    )

    prompt += (
        "**Instructions:**\n"
        "- Review both sides' arguments\n"
        "- Identify the main points of contention\n"
        "- Present the answer candidates without providing reasons\n"
        "- Focus on clarity and objectivity\n"
    ) + NEW_LINE

    prompt += (
        "**Now, what answer candidates do we have? Present them without reasons.**"
    )

    return prompt


def build_mllm_judge_pair_mad_judge_prompt_2(debate_topic: str) -> str:
    """Build second judge prompt for MLLM Judge Pair MAD debate.

    Args:
        debate_topic: The debate topic with question and responses

    Returns:
        str: The formatted second judge prompt
    """
    prompt = "**Therefore, " + debate_topic + "\n" + NEW_LINE

    prompt += (
        "**Your Role:** As the final judge, you must make the ultimate decision based on the debate.\n"
        + NEW_LINE
    )

    prompt += (
        "**Task:** Summarize your reasons and give the final answer that you think is correct.\n"
        + NEW_LINE
    )

    prompt += (
        "**Evaluation Instructions:**\n"
        "1. Review the entire debate and all arguments presented\n"
        "2. Consider the quality of reasoning from both sides\n"
        "3. Evaluate how well each side addressed the visual content\n"
        "4. Evaluate the evidence and examples provided\n"
        "5. Assess which response better addresses the original question and image\n"
        "6. Make your final determination\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Visual Understanding: Which response better interprets the image?\n"
        "- Overall Accuracy: Which response is more factually correct?\n"
        "- Completeness: Which response better addresses all aspects of the question and image?\n"
        "- Relevance: Which response is more relevant to the topic and visual content?\n"
        "- Helpfulness: Which response is more useful and actionable?\n"
        "- Clarity: Which response is clearer and more understandable?\n"
    ) + NEW_LINE

    prompt += (
        "Please summarize your reasons and give the final answer that you think is correct.\n"
        + NEW_LINE
    )

    prompt += "You MUST answer in the following JSON format:\n"
    prompt += JSON_FORMAT + NEW_LINE

    prompt += (
        "**Note:** The 'Final Answer' MUST be placed at the end of your response, "
        "and the value must be only 'Response A' or 'Response B'. "
        "Do not include any other text after the JSON response."
    )

    return prompt


def build_mllm_judge_pair_mad_debate_prompt(oppo_ans: str) -> str:
    """Build debate prompt for MLLM Judge Pair MAD debate.

    Args:
        oppo_ans: The opposing side's argument

    Returns:
        str: The formatted debate prompt
    """
    prompt = oppo_ans + NEW_LINE + NEW_LINE

    prompt += (
        "**Your Role:** You are continuing the debate with the opposing side.\n"
        + NEW_LINE
    )

    prompt += (
        "**Task:** Respond to the opposing argument and provide your own perspective.\n"
        + NEW_LINE
    )

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


def build_mllm_judge_pair_mad_debater_prompt(debate_topic: str) -> str:
    """Build debater prompt for MLLM Judge Pair MAD debate.

    Args:
        debate_topic: The debate topic with question and responses

    Returns:
        str: The formatted debater prompt
    """
    prompt = f"**Debate Topic:** {debate_topic}\n" + NEW_LINE
    prompt += "**Debate History:** ##debate_history##\n" + NEW_LINE

    prompt += (
        "**Your Role:** You are ##debater_name## (Debater ##debater_number##). You are participating in a debate competition with multiple debaters.\n"
        + NEW_LINE
    )

    prompt += "**Your Assigned Position:** ##debater_position##\n" + NEW_LINE

    prompt += (
        "**Task:** Express your arguments based on the previous debate history.\n"
        + NEW_LINE
    )

    prompt += (
        "**Instructions:**\n"
        "1. Review the complete debate history carefully\n"
        "2. Consider all arguments presented by other debaters\n"
        "3. Defend your assigned position with strong arguments\n"
        "4. Challenge opposing arguments and build upon supporting ones\n"
        "5. Provide evidence and reasoning to support your position\n"
        "6. Engage constructively with the ongoing discussion\n"
    ) + NEW_LINE

    prompt += (
        "**Debate Context:**\n"
        "- You are one of multiple debaters in this competition\n"
        "- Each debater speaks in a fixed order\n"
        "- You must consider all previous arguments when forming your response\n"
        "- Your goal is to defend your assigned position effectively\n"
        "- Provide actual debate arguments, not just a final choice\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Relevance: How well does your argument relate to the topic?\n"
        "- Logic: How logical and coherent is your reasoning?\n"
        "- Evidence: How much relevant evidence do you provide?\n"
        "- Engagement: How well do you engage with previous arguments?\n"
        "- Clarity: How clear and understandable is your argument?\n"
        "- Position Defense: How well do you defend your assigned position?\n"
    ) + NEW_LINE

    prompt += (
        "**CRITICAL:** You MUST defend your assigned position regardless of your personal opinion. "
        "Even if you personally think the other response is better, you must argue for your assigned position. "
        "This is a debate competition where you are assigned a side to defend.\n"
    ) + NEW_LINE

    prompt += (
        "**DEBATE RULES:**\n"
        "- You are NOT a judge - you are a debater assigned to defend a specific position\n"
        "- You MUST argue for your assigned position, even if you disagree with it\n"
        "- Do NOT say 'I believe Response A is better' - say 'I am arguing that Response A is better'\n"
        "- Focus on finding strengths in your assigned response and weaknesses in the opposing response\n"
        "- This is a structured debate, not an evaluation\n"
        "- You are in a competitive debate - you want to WIN for your side\n"
        "- Challenge the other debater's arguments and defend your position aggressively\n"
        "- If the other debater argued for your position, you must still defend your assigned position and challenge their reasoning\n"
    ) + NEW_LINE

    prompt += (
        "**Important:** Provide a detailed debate argument defending your assigned position with reasoning and evidence. "
        "Do NOT just choose 'Response A' or 'Response B' - actually debate the topic with arguments and reasoning.\n"
    ) + NEW_LINE

    prompt += "**Please express your arguments based on the debate history provided, defending your assigned position.**"

    return prompt


def build_mllm_judge_pair_mad_judge_discriminative_prompt(debate_topic: str) -> str:
    """Build judge discriminative prompt for MLLM Judge Pair MAD debate.

    Args:
        debate_topic: The debate topic with question and responses

    Returns:
        str: The formatted judge discriminative prompt
    """
    prompt = "**Debate History:** ##debate_history##\n" + NEW_LINE

    prompt += "**Current Round:** ##current_round##\n" + NEW_LINE

    prompt += (
        "**Your Role:** You are a moderator in a debate competition. You must evaluate whether a correct solution has been obtained after the current iteration.\n"
        + NEW_LINE
    )

    prompt += (
        "**Task:** Determine if the correct solution can be obtained based on the current debate state.\n"
        + NEW_LINE
    )

    prompt += (
        "**Evaluation Instructions:**\n"
        "1. Review the complete debate history up to the current round\n"
        "2. Evaluate the quality and clarity of arguments presented\n"
        "3. Assess whether both sides have presented sufficient evidence and reasoning\n"
        "4. Determine if a clear, correct solution has emerged from the debate\n"
        "5. Consider the logical coherence and persuasiveness of arguments\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Argument Quality: Are the arguments well-reasoned and supported?\n"
        "- Evidence: Is there sufficient evidence presented by both sides?\n"
        "- Logic: Is the reasoning logical and coherent?\n"
        "- Clarity: Is there a clear winner or correct answer emerging?\n"
        "- Completeness: Have all aspects of the question been addressed?\n"
    ) + NEW_LINE

    prompt += (
        "**Decision Process:**\n"
        "- If a clear, correct solution has emerged → solution_obtained = True\n"
        "- If the debate needs to continue for more clarity → solution_obtained = False\n"
    ) + NEW_LINE

    prompt += (
        "**Output Format:**\n"
        "You must respond in the following JSON format:\n"
        "{\n"
        '  "solution_obtained": true/false,\n'
        '  "reasoning": "your detailed reasoning for the decision",\n'
        '  "Final Answer": "Response A" or "Response B" (only if solution_obtained is true)\n'
        "}\n"
    ) + NEW_LINE

    prompt += (
        "**Important:** If solution_obtained is true, you MUST provide a Final Answer of either 'Response A' or 'Response B'. "
        "If solution_obtained is false, you can omit the Final Answer field.\n"
    )

    return prompt


def build_mllm_judge_pair_mad_judge_extractive_prompt(debate_topic: str) -> str:
    """Build judge extractive prompt for MLLM Judge Pair MAD debate.

    Args:
        debate_topic: The debate topic with question and responses

    Returns:
        str: The formatted judge extractive prompt
    """
    prompt = "**Complete Debate History:** ##debate_history##\n" + NEW_LINE

    prompt += (
        "**Your Role:** You are a final judge in a debate competition. Your task is to extract the final answer from the complete debate history.\n"
        + NEW_LINE
    )

    prompt += (
        "**Task:** Based on the complete debate history, determine which response (Response A or Response B) is better and provide your final answer.\n"
        + NEW_LINE
    )

    prompt += (
        "**Instructions:**\n"
        "1. Review the complete debate history carefully\n"
        "2. Consider all arguments presented by all debaters\n"
        "3. Evaluate the strength and validity of each argument\n"
        "4. Determine which response is ultimately better\n"
        "5. Provide clear reasoning for your decision\n"
        "6. Give your final answer as either 'Response A' or 'Response B'\n"
    ) + NEW_LINE

    prompt += (
        "**Evaluation Criteria:**\n"
        "- Visual Understanding: Which response better interprets the image?\n"
        "- Overall Accuracy: Which response is more factually correct?\n"
        "- Completeness: Which response better addresses all aspects of the question and image?\n"
        "- Relevance: Which response is more relevant to the topic and visual content?\n"
        "- Helpfulness: Which response is more useful and actionable?\n"
        "- Clarity: Which response is clearer and more understandable?\n"
    ) + NEW_LINE

    prompt += (
        "**Output Format:**\n"
        "You must respond in the following JSON format:\n"
        "{\n"
        '  "reasoning": "your detailed reasoning for the final decision",\n'
        '  "Final Answer": "Response A" or "Response B"\n'
        "}\n"
    ) + NEW_LINE

    prompt += (
        "**Important:** Your Final Answer must be either 'Response A' or 'Response B'. "
        "Do not include any other text after the JSON response.\n"
    )

    return prompt


def build_mllm_judge_pair_mad_prompts(debate_topic: str) -> Dict[str, str]:
    """Build all MLLM Judge Pair MAD prompts.

    Args:
        debate_topic: The debate topic with question and responses

    Returns:
        Dict containing all MAD prompts for MLLM Judge Pair
    """
    return {
        "player_meta_prompt": build_mllm_judge_pair_mad_player_meta_prompt(
            debate_topic
        ),
        "moderator_meta_prompt": build_mllm_judge_pair_mad_moderator_meta_prompt(
            debate_topic
        ),
        "affirmative_prompt": build_mllm_judge_pair_mad_affirmative_prompt(
            debate_topic
        ),
        "negative_prompt": build_mllm_judge_pair_mad_negative_prompt(
            "##aff_ans##"
        ),  # Placeholder
        "moderator_prompt": build_mllm_judge_pair_mad_moderator_prompt(
            "##aff_ans##", "##neg_ans##", "##round##"
        ),  # Placeholders
        "judge_prompt_last1": build_mllm_judge_pair_mad_judge_prompt_1(
            "##aff_ans##", "##neg_ans##"
        ),  # Placeholders
        "judge_prompt_last2": build_mllm_judge_pair_mad_judge_prompt_2(debate_topic),
        "debate_prompt": build_mllm_judge_pair_mad_debate_prompt(
            "##oppo_ans##"
        ),  # Placeholder
        # N-debater framework prompts
        "judge_meta_prompt": build_mllm_judge_pair_mad_moderator_meta_prompt(
            debate_topic
        ),  # Reuse moderator meta prompt
        "debater_prompt": build_mllm_judge_pair_mad_debater_prompt(debate_topic),
        "judge_discriminative_prompt": build_mllm_judge_pair_mad_judge_discriminative_prompt(
            debate_topic
        ),
        "judge_extractive_prompt": build_mllm_judge_pair_mad_judge_extractive_prompt(
            debate_topic
        ),
    }
