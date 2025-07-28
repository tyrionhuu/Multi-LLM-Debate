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
            "player_meta_prompt": PLAYER_META_PROMPT.replace("Response 1", "Response A").replace("Response 2", "Response B"),
            "moderator_meta_prompt": MODERATOR_META_PROMPT.replace("Response 1", "Response A").replace("Response 2", "Response B"),
            "affirmative_prompt": AFFIRMATIVE_PROMPT.replace("Response 1", "Response A").replace("Response 2", "Response B"),
            "negative_prompt": NEGATIVE_PROMPT.replace("Response 1", "Response A").replace("Response 2", "Response B"),
            "moderator_prompt": MODERATOR_PROMPT.replace("Response 1", "Response A").replace("Response 2", "Response B"),
            "judge_prompt_last1": JUDGE_PROMPT_1,
            "judge_prompt_last2": JUDGE_PROMPT_2.replace("Response 1", "Response A").replace("Response 2", "Response B"),
            "debate_prompt": DEBATE_PROMPT
        }
    
    elif task_name == "big_bench":
        # Big Bench expects "0/1" format
        return {
            "player_meta_prompt": PLAYER_META_PROMPT.replace("Response 1", "Response 1 (Yes)").replace("Response 2", "Response 2 (No)"),
            "moderator_meta_prompt": MODERATOR_META_PROMPT.replace("Response 1", "Response 1 (Yes)").replace("Response 2", "Response 2 (No)"),
            "affirmative_prompt": AFFIRMATIVE_PROMPT.replace("Response 1", "Response 1 (Yes)").replace("Response 2", "Response 2 (No)"),
            "negative_prompt": NEGATIVE_PROMPT.replace("Response 1", "Response 1 (Yes)").replace("Response 2", "Response 2 (No)"),
            "moderator_prompt": MODERATOR_PROMPT.replace("Response 1", "Response 1 (Yes)").replace("Response 2", "Response 2 (No)"),
            "judge_prompt_last1": JUDGE_PROMPT_1,
            "judge_prompt_last2": JUDGE_PROMPT_2.replace("Response 1", "Response 1 (Yes)").replace("Response 2", "Response 2 (No)"),
            "debate_prompt": DEBATE_PROMPT
        }
    
    elif task_name == "truthful_qa":
        # TruthfulQA expects "A/B/C" format but converted to "Response 1/Response 2"
        return {
            "player_meta_prompt": PLAYER_META_PROMPT,
            "moderator_meta_prompt": MODERATOR_META_PROMPT,
            "affirmative_prompt": AFFIRMATIVE_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "moderator_prompt": MODERATOR_PROMPT,
            "judge_prompt_last1": JUDGE_PROMPT_1,
            "judge_prompt_last2": JUDGE_PROMPT_2,
            "debate_prompt": DEBATE_PROMPT
        }
    
    elif task_name == "llm_bar":
        # LLM Bar uses "Response 1/Response 2" format
        return {
            "player_meta_prompt": PLAYER_META_PROMPT,
            "moderator_meta_prompt": MODERATOR_META_PROMPT,
            "affirmative_prompt": AFFIRMATIVE_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "moderator_prompt": MODERATOR_PROMPT,
            "judge_prompt_last1": JUDGE_PROMPT_1,
            "judge_prompt_last2": JUDGE_PROMPT_2,
            "debate_prompt": DEBATE_PROMPT
        }
    
    else:
        # Default to "Response 1/Response 2" format
        return {
            "player_meta_prompt": PLAYER_META_PROMPT,
            "moderator_meta_prompt": MODERATOR_META_PROMPT,
            "affirmative_prompt": AFFIRMATIVE_PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "moderator_prompt": MODERATOR_PROMPT,
            "judge_prompt_last1": JUDGE_PROMPT_1,
            "judge_prompt_last2": JUDGE_PROMPT_2,
            "debate_prompt": DEBATE_PROMPT
        }
