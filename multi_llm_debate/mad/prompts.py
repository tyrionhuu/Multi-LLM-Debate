from typing import Dict, List

NEW_LINE = "\n"
DIVIDER = "#" * 80

# Detailed JSON format for MAD responses - following benchmark quality standards
JSON_FORMAT = """
{
    "reasoning": "your detailed reasoning based on the debate",
    "Final Answer": "Response 1"
}
"""

JSON_FORMAT_COT = """
{
    "reasoning": {
        "step_1": "first step of your detailed reasoning",
        "step_2": "second step of your detailed reasoning", 
        "step_3": "third step of your detailed reasoning",
        "...": "continue with as many steps as needed"
    },
    "Final Answer": "Response 1"
}
"""

NON_JSON_FORMAT = """
Reasoning: your detailed reasoning based on the debate
Final Answer: Response 1
"""

NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your detailed reasoning
Step 2: second step of your detailed reasoning
Step 3: third step of your detailed reasoning
...
Final Answer: Response 1
"""

# Player meta prompt - following benchmark quality standards
PLAYER_META_PROMPT = """**Your Role:** As an assistant, your task is to serve as a debater in a structured debate.

**Your Goal:** Find the correct answer through reasoned discussion between two responses labeled "Response 1" and "Response 2".

The debate topic is:
##debate_topic##

**Debate Instructions:**
- Analyze both responses carefully and objectively
- Consider factors such as accuracy, completeness, relevance, and helpfulness
- Provide clear reasoning for your position
- Engage constructively with opposing arguments
- Your final choice must be either "Response 1" or "Response 2"

**Evaluation Criteria:**
- Accuracy: How factually correct is the response?
- Completeness: Does the response address all aspects of the question?
- Relevance: How well does the response relate to the question?
- Helpfulness: How useful and actionable is the response?
- Clarity: How clear and understandable is the response?"""

# Judge meta prompt - following benchmark quality standards
JUDGE_META_PROMPT = """As an assistant, your task is to serve as a judge in a structured debate.

You will evaluate a debate between two responses labeled "Response 1" and "Response 2" and determine which response is better.

The debate topic is:
##debate_topic##

**Judge Instructions:**
- Listen carefully to both sides of the debate
- Evaluate the quality of arguments presented
- Consider the evidence and reasoning provided
- Make an objective assessment based on the debate
- At the end of the debate, you will evaluate both sides and decide which response is correct

**Evaluation Criteria:**
- Argument Quality: How well-reasoned and supported are the arguments?
- Evidence: How much relevant evidence is presented?
- Logic: How logical and coherent is the reasoning?
- Fairness: Are both sides given equal consideration?
- Conclusion: Which response ultimately provides the better answer?"""

# NEW DEBATE STRUCTURE PROMPTS

# Debater A initial prompt - gives the first answer
DEBATER_A_INITIAL_PROMPT = """##debate_topic##

**Your Role:** You are Debater A. You are the first to provide an answer to the question.

**Task:** Analyze the question carefully and provide your initial answer and reasoning.

**Instructions:**
1. Read and understand the question thoroughly
2. Consider all relevant factors and information
3. Provide a clear, well-reasoned answer
4. Support your answer with logical reasoning and evidence
5. Be confident but open to discussion

**IMPORTANT:** You must choose between "Response 1" and "Response 2" and provide clear reasoning for your choice.

**Evaluation Criteria:**
- Accuracy: How factually correct is your answer?
- Completeness: Does your answer address all aspects of the question?
- Relevance: How well does your answer relate to the question?
- Helpfulness: How useful and actionable is your answer?
- Clarity: How clear and understandable is your answer?"""

# Debater B disagree prompt - disagrees with Debater A's answer
DEBATER_B_DISAGREE_PROMPT = """##debater_a_answer##

**Your Role:** You are Debater B. You disagree with Debater A's position and must provide your own analysis.

**Task:** Review Debater A's answer and provide your disagreement with reasons.

**Instructions:**
1. Review Debater A's argument carefully
2. Identify weaknesses or flaws in their reasoning
3. Present your own independent analysis of the question
4. Argue for your preferred choice with specific reasons
5. Provide counter-evidence or alternative perspectives
6. Be respectful but firm in your disagreement

**Evaluation Criteria:**
- Accuracy: How factually correct is your analysis?
- Completeness: Does your analysis address all aspects of the question?
- Relevance: How well does your analysis relate to the question?
- Helpfulness: How useful and actionable is your analysis?
- Clarity: How clear and understandable is your analysis?

**IMPORTANT:** You must choose between "Response 1" and "Response 2" and provide clear reasoning for your choice."""

# Debater A rebuttal prompt - debates Debater B's points
DEBATER_A_REBUTTAL_PROMPT = """##debater_b_disagreement##

**Your Role:** You are Debater A. You must now respond to Debater B's disagreement and defend your position.

**Task:** Address Debater B's points and strengthen your original argument.

**Instructions:**
1. Read and understand Debater B's disagreement carefully
2. Consider their reasoning and evidence
3. Address any valid points they raised
4. Provide additional evidence or reasoning to support your position
5. Counter any weak or incorrect arguments they made
6. Reinforce the strength of your original position

**Please provide your rebuttal and reinforce your answer with additional reasoning.**"""

# NEW N-DEBATER PROMPT

# Debater prompt for N debaters - each debater expresses arguments based on history
DEBATER_PROMPT = """**Debate History:** ##debate_history##

**Your Role:** You are ##debater_name## (Debater ##debater_number##). You are participating in a debate competition with multiple debaters.

**Task:** Express your arguments based on the previous debate history.

**Instructions:**
1. Review the complete debate history carefully
2. Consider all arguments presented by other debaters
3. Express your own viewpoint and reasoning
4. Build upon or challenge previous arguments as appropriate
5. Provide evidence and reasoning to support your position
6. Engage constructively with the ongoing discussion

**Debate Context:**
- You are one of multiple debaters in this competition
- Each debater speaks in a fixed order
- You must consider all previous arguments when forming your response
- Your goal is to contribute meaningfully to the debate

**Evaluation Criteria:**
- Relevance: How well does your argument relate to the topic?
- Logic: How logical and coherent is your reasoning?
- Evidence: How much relevant evidence do you provide?
- Engagement: How well do you engage with previous arguments?
- Clarity: How clear and understandable is your argument?

**Please express your arguments based on the debate history provided.**"""

# Judge decision prompt - makes the final decision
JUDGE_DECISION_PROMPT = """**Debater A Initial Answer:** ##debater_a_answer##

**Debater B Disagreement:** ##debater_b_disagreement##

**Debater A Rebuttal:** ##debater_a_rebuttal##

**Your Role:** As the judge, you must make the ultimate decision based on the complete debate.

**Task:** Evaluate the entire debate and determine which response is correct.

**Evaluation Instructions:**
1. Review all arguments presented by both debaters
2. Consider the quality of reasoning from both sides
3. Evaluate the evidence and examples provided
4. Assess which response better addresses the original question
5. Make your final determination

**Evaluation Criteria:**
- Overall Accuracy: Which response is more factually correct?
- Completeness: Which response better addresses all aspects of the question?
- Relevance: Which response is more relevant to the topic?
- Helpfulness: Which response is more useful and actionable?
- Clarity: Which response is clearer and more understandable?
- Argument Strength: Which side presented stronger arguments and evidence?

Please summarize your reasons and give the final answer that you think is correct.

You MUST answer in the following JSON format:
{
    "reasoning": "your detailed reasoning based on the complete debate",
    "Final Answer": "Response 1"
}

**Note:** The 'Final Answer' MUST be placed at the end of your response, 
and the value must be only "Response 1" or "Response 2". 
Do not include any other text after the JSON response."""

# NEW ITERATIVE JUDGE PROMPTS

# Judge Discriminative Mode (Jd) - decides if correct solution is obtained
JUDGE_DISCRIMINATIVE_PROMPT = """**Debate History:** ##debate_history##

**Current Round:** ##current_round##

**Your Role:** You are a moderator in a debate competition. You must evaluate whether a correct solution has been obtained after the current iteration.

**Task:** Determine if the correct solution can be obtained based on the current debate state.

**Evaluation Instructions:**
1. Review the complete debate history up to the current round
2. Evaluate the quality and clarity of arguments presented
3. Assess whether both sides have presented sufficient evidence and reasoning
4. Determine if a clear, correct solution has emerged from the debate
5. Consider the logical coherence and persuasiveness of arguments

**Evaluation Criteria:**
- Argument Quality: Are the arguments well-reasoned and supported?
- Evidence: Is there sufficient evidence presented by both sides?
- Logic: Is the reasoning logical and coherent?
- Clarity: Is there a clear winner or correct answer emerging?
- Completeness: Have all aspects of the question been addressed?

**Decision Process:**
- If a clear, correct solution has emerged → solution_obtained = True
- If the debate needs to continue for more clarity → solution_obtained = False

You MUST answer in the following JSON format:
{
    "solution_obtained": true/false,
    "reasoning": "your detailed reasoning for the decision"
}

**Note:** 
- Set "solution_obtained" to true only if a clear, correct solution has emerged
- Set "solution_obtained" to false if the debate should continue
- Provide clear reasoning for your decision"""

# Judge Extractive Mode (Je) - extracts final solution from debate history
JUDGE_EXTRACTIVE_PROMPT = """**Complete Debate History:** ##debate_history##

**Your Role:** You are a moderator in a debate competition. You must extract the final solution based on the complete debate history.

**Task:** Evaluate the entire debate and determine which response is correct.

**Evaluation Instructions:**
1. Review the complete debate history from all iterations
2. Consider all arguments presented by both debaters
3. Evaluate the quality of reasoning from both sides
4. Assess which response better addresses the original question
5. Make your final determination based on the complete debate

**Evaluation Criteria:**
- Overall Accuracy: Which response is more factually correct?
- Completeness: Which response better addresses all aspects of the question?
- Relevance: Which response is more relevant to the topic?
- Helpfulness: Which response is more useful and actionable?
- Clarity: Which response is clearer and more understandable?
- Argument Strength: Which side presented stronger arguments and evidence throughout the debate?

**Final Decision Process:**
- Consider all iterations and arguments presented
- Weigh the evidence and reasoning from both sides
- Determine which response ultimately provides the better answer
- Provide clear reasoning for your final decision

Please summarize your reasons and give the final answer that you think is correct.

You MUST answer in the following JSON format:
{
    "reasoning": "your detailed reasoning based on the complete debate history",
    "Final Answer": "Response 1"
}

**Note:** The 'Final Answer' MUST be placed at the end of your response, 
and the value must be only "Response 1" or "Response 2". 
Do not include any other text after the JSON response."""

# LEGACY PROMPTS (for backward compatibility)

# Moderator meta prompt - following benchmark quality standards
MODERATOR_META_PROMPT = """As an assistant, your task is to serve as a moderator in a structured debate.

You will evaluate a debate between two responses labeled "Response 1" and "Response 2" and determine which response is better.

The debate topic is:
##debate_topic##

**Moderator Instructions:**
- Listen carefully to both sides of the debate
- Evaluate the quality of arguments presented
- Consider the evidence and reasoning provided
- Make an objective assessment based on the debate
- At the end of each round, you will evaluate both sides and decide which response is correct

**Evaluation Criteria:**
- Argument Quality: How well-reasoned and supported are the arguments?
- Evidence: How much relevant evidence is presented?
- Logic: How logical and coherent is the reasoning?
- Fairness: Are both sides given equal consideration?
- Conclusion: Which response ultimately provides the better answer?"""

# Affirmative prompt - following benchmark quality standards
AFFIRMATIVE_PROMPT = """##debate_topic##

**Your Role:** You are the affirmative debater arguing for one of the two responses.

**Task:** Analyze both responses carefully and argue for which one is better.

**Instructions:**
1. Read and understand the question and both responses
2. Evaluate each response based on:
   - Accuracy and factual correctness
   - Completeness in addressing the question
   - Relevance to the topic
   - Helpfulness and usefulness
   - Clarity and understandability
3. Present your argument for which response is superior
4. Provide specific reasons and evidence for your choice
5. Address potential counterarguments

**IMPORTANT:** You must choose between "Response 1" and "Response 2" and provide clear reasoning for your choice."""

# Negative prompt - following benchmark quality standards
NEGATIVE_PROMPT = """##aff_ans##

**Your Role:** You are the negative debater who disagrees with the affirmative side's position.

**Task:** Provide your own analysis and reasoning for which response is better.

**Instructions:**
1. Review the affirmative side's argument carefully
2. Consider their reasoning and evidence
3. Present your own independent analysis of both responses
4. Argue for your preferred choice with specific reasons
5. Address any weaknesses in the affirmative argument
6. Provide counter-evidence or alternative perspectives

**Evaluation Criteria:**
- Accuracy: How factually correct is each response?
- Completeness: Does each response address all aspects of the question?
- Relevance: How well does each response relate to the question?
- Helpfulness: How useful and actionable is each response?
- Clarity: How clear and understandable is each response?

**IMPORTANT:** You must choose between "Response 1" and "Response 2" and provide clear reasoning for your choice."""

# Moderator prompt - following benchmark quality standards
MODERATOR_PROMPT = """Now the ##round## round of debate for both sides has ended.

**Affirmative side arguing:**
##aff_ans##

**Negative side arguing:** 
##neg_ans##

**Your Role:** As the moderator, evaluate both sides' arguments and determine which response is better.

**Evaluation Instructions:**
1. Review both sides' arguments carefully
2. Consider the quality of reasoning presented
3. Evaluate the evidence and examples provided
4. Assess the logical coherence of each position
5. Determine which response ultimately provides the better answer

**Evaluation Criteria:**
- Argument Quality: How well-reasoned and supported are the arguments?
- Evidence: How much relevant evidence is presented?
- Logic: How logical and coherent is the reasoning?
- Fairness: Are both sides given equal consideration?
- Conclusion: Which response ultimately provides the better answer?

You MUST answer in the following JSON format:
{
    "reasoning": "your detailed reasoning based on the debate",
    "Final Answer": "Response 1"
}

**Note:** The 'Final Answer' MUST be placed at the end of your response, 
and the value must be only "Response 1" or "Response 2". 
Do not include any other text after the JSON response."""

# Judge prompts - following benchmark quality standards
JUDGE_PROMPT_1 = """**Affirmative side arguing:** ##aff_ans##

**Negative side arguing:** ##neg_ans##

**Task:** Summarize the key arguments and answer candidates presented in this debate.

**Instructions:**
- Review both sides' arguments
- Identify the main points of contention
- Present the answer candidates without providing reasons
- Focus on clarity and objectivity

**Now, what answer candidates do we have? Present them without reasons.**"""

JUDGE_PROMPT_2 = """**Therefore, ##debate_topic##

**Your Role:** As the final judge, you must make the ultimate decision based on the debate.

**Task:** Summarize your reasons and give the final answer that you think is correct.

**Evaluation Instructions:**
1. Review the entire debate and all arguments presented
2. Consider the quality of reasoning from both sides
3. Evaluate the evidence and examples provided
4. Assess which response better addresses the original question
5. Make your final determination

**Evaluation Criteria:**
- Overall Accuracy: Which response is more factually correct?
- Completeness: Which response better addresses all aspects of the question?
- Relevance: Which response is more relevant to the topic?
- Helpfulness: Which response is more useful and actionable?
- Clarity: Which response is clearer and more understandable?

Please summarize your reasons and give the final answer that you think is correct.

You MUST answer in the following JSON format:
{
    "reasoning": "your detailed reasoning based on the debate",
    "Final Answer": "Response 1"
}

**Note:** The 'Final Answer' MUST be placed at the end of your response, 
and the value must be only "Response 1" or "Response 2". 
Do not include any other text after the JSON response."""

# Debate prompt - following benchmark quality standards
DEBATE_PROMPT = """##oppo_ans##

**Your Role:** You are continuing the debate with the opposing side.

**Task:** Respond to the opposing argument and provide your own perspective.

**Instructions:**
1. Read and understand the opposing argument carefully
2. Consider their reasoning and evidence
3. Do you agree with their perspective? Why or why not?
4. Provide your own analysis and reasoning
5. Support your position with specific evidence or examples
6. Address any weaknesses in the opposing argument

**Please provide your reasons and answer.**"""


# Task-specific prompt builders
def build_mad_prompts_for_task(task_name: str) -> Dict[str, str]:
    """Build MAD prompts specific to a given task.

    Args:
        task_name: Name of the task (e.g., "judge_anything_pair", "big_bench", etc.)

    Returns:
        Dict containing task-specific prompts
    """

    if task_name in ["judge_anything_pair", "mllm_judge_pair", "judge_bench"]:
        # These tasks expect "A/B" format
        return {
            "player_meta_prompt": PLAYER_META_PROMPT.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            "judge_meta_prompt": JUDGE_META_PROMPT.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            "debater_prompt": DEBATER_PROMPT,
            "judge_discriminative_prompt": JUDGE_DISCRIMINATIVE_PROMPT,
            "judge_extractive_prompt": JUDGE_EXTRACTIVE_PROMPT.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            # Legacy prompts for backward compatibility
            "debater_a_initial_prompt": DEBATER_A_INITIAL_PROMPT.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            "debater_b_disagree_prompt": DEBATER_B_DISAGREE_PROMPT.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            "debater_a_rebuttal_prompt": DEBATER_A_REBUTTAL_PROMPT,
            "judge_decision_prompt": JUDGE_DECISION_PROMPT.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            "moderator_meta_prompt": MODERATOR_META_PROMPT.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            "affirmative_prompt": AFFIRMATIVE_PROMPT.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            "negative_prompt": NEGATIVE_PROMPT.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            "moderator_prompt": MODERATOR_PROMPT.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            "judge_prompt_last1": JUDGE_PROMPT_1,
            "judge_prompt_last2": JUDGE_PROMPT_2.replace(
                "Response 1", "Response A"
            ).replace("Response 2", "Response B"),
            "debate_prompt": DEBATE_PROMPT,
        }

    elif task_name == "big_bench":
        # Big Bench expects "0/1" format
        return {
            "player_meta_prompt": PLAYER_META_PROMPT.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "judge_meta_prompt": JUDGE_META_PROMPT.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "debater_prompt": DEBATER_PROMPT,
            "judge_discriminative_prompt": JUDGE_DISCRIMINATIVE_PROMPT,
            "judge_extractive_prompt": JUDGE_EXTRACTIVE_PROMPT.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "debater_a_initial_prompt": DEBATER_A_INITIAL_PROMPT.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "debater_b_disagree_prompt": DEBATER_B_DISAGREE_PROMPT.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "debater_a_rebuttal_prompt": DEBATER_A_REBUTTAL_PROMPT,
            # Legacy prompts for backward compatibility
            "judge_decision_prompt": JUDGE_DECISION_PROMPT.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "moderator_meta_prompt": MODERATOR_META_PROMPT.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "affirmative_prompt": AFFIRMATIVE_PROMPT.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "negative_prompt": NEGATIVE_PROMPT.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "moderator_prompt": MODERATOR_PROMPT.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "judge_prompt_last1": JUDGE_PROMPT_1,
            "judge_prompt_last2": JUDGE_PROMPT_2.replace(
                "Response 1", "Response 1 (Yes)"
            ).replace("Response 2", "Response 2 (No)"),
            "debate_prompt": DEBATE_PROMPT,
        }

    elif task_name == "truthful_qa":
        # TruthfulQA expects "A/B/C" format but converted to "Response 1/Response 2"
        return {
            "player_meta_prompt": PLAYER_META_PROMPT,
            "judge_meta_prompt": JUDGE_META_PROMPT,
            "debater_prompt": DEBATER_PROMPT,
            "judge_discriminative_prompt": JUDGE_DISCRIMINATIVE_PROMPT,
            "judge_extractive_prompt": JUDGE_EXTRACTIVE_PROMPT,
            "debater_a_initial_prompt": DEBATER_A_INITIAL_PROMPT,
            "debater_b_disagree_prompt": DEBATER_B_DISAGREE_PROMPT,
            "debater_a_rebuttal_prompt": DEBATER_A_REBUTTAL_PROMPT,
            "judge_decision_prompt": JUDGE_DECISION_PROMPT,
            # Legacy prompts for backward compatibility
            "moderator_meta_prompt": MODERATOR_META_PROMPT,
            "affirmative_prompt": AFFIRMATIVE_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "moderator_prompt": MODERATOR_PROMPT,
            "judge_prompt_last1": JUDGE_PROMPT_1,
            "judge_prompt_last2": JUDGE_PROMPT_2,
            "debate_prompt": DEBATE_PROMPT,
        }

    elif task_name == "llm_bar":
        # LLM Bar uses "Response 1/Response 2" format
        return {
            "player_meta_prompt": PLAYER_META_PROMPT,
            "judge_meta_prompt": JUDGE_META_PROMPT,
            "debater_prompt": DEBATER_PROMPT,
            "judge_discriminative_prompt": JUDGE_DISCRIMINATIVE_PROMPT,
            "judge_extractive_prompt": JUDGE_EXTRACTIVE_PROMPT,
            "debater_a_initial_prompt": DEBATER_A_INITIAL_PROMPT,
            "debater_b_disagree_prompt": DEBATER_B_DISAGREE_PROMPT,
            "debater_a_rebuttal_prompt": DEBATER_A_REBUTTAL_PROMPT,
            "judge_decision_prompt": JUDGE_DECISION_PROMPT,
            # Legacy prompts for backward compatibility
            "moderator_meta_prompt": MODERATOR_META_PROMPT,
            "affirmative_prompt": AFFIRMATIVE_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "moderator_prompt": MODERATOR_PROMPT,
            "judge_prompt_last1": JUDGE_PROMPT_1,
            "judge_prompt_last2": JUDGE_PROMPT_2,
            "debate_prompt": DEBATE_PROMPT,
        }

    else:
        # Default to "Response 1/Response 2" format
        return {
            "player_meta_prompt": PLAYER_META_PROMPT,
            "judge_meta_prompt": JUDGE_META_PROMPT,
            "debater_prompt": DEBATER_PROMPT,
            "judge_discriminative_prompt": JUDGE_DISCRIMINATIVE_PROMPT,
            "judge_extractive_prompt": JUDGE_EXTRACTIVE_PROMPT,
            "debater_a_initial_prompt": DEBATER_A_INITIAL_PROMPT,
            "debater_b_disagree_prompt": DEBATER_B_DISAGREE_PROMPT,
            "debater_a_rebuttal_prompt": DEBATER_A_REBUTTAL_PROMPT,
            "judge_decision_prompt": JUDGE_DECISION_PROMPT,
            # Legacy prompts for backward compatibility
            "moderator_meta_prompt": MODERATOR_META_PROMPT,
            "affirmative_prompt": AFFIRMATIVE_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "moderator_prompt": MODERATOR_PROMPT,
            "judge_prompt_last1": JUDGE_PROMPT_1,
            "judge_prompt_last2": JUDGE_PROMPT_2,
            "debate_prompt": DEBATE_PROMPT,
        }
