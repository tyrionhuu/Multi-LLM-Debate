from typing import Dict, List

NEW_LINE = "\n"
DIVIDER = "#" * 80

# JSON format for MAD responses
JSON_FORMAT = """
{
    "reasoning": "your reasoning based on the debate",
    "Final Answer": "Response 1/Response 2"
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
    "Final Answer": "Response 1/Response 2"
}
"""

NON_JSON_FORMAT = """
Reasoning: your reasoning based on the debate
Final Answer: Response 1/Response 2
"""

NON_JSON_FORMAT_COT = """
Reasoning:
Step 1: first step of your reasoning
Step 2: second step of your reasoning
Step 3: third step of your reasoning
...
Final Answer: Response 1/Response 2
"""

# Player meta prompt - follows your debate framework style
PLAYER_META_PROMPT = """As an assistant, your task is to serve as a debater in a structured debate.

Your goal is to find the correct answer through reasoned discussion between two responses labeled "Response 1" and "Response 2".

The debate topic is:
##debate_topic##

Remember: Your final choice must be either "Response 1" or "Response 2"."""

# Moderator meta prompt - follows your debate framework style  
MODERATOR_META_PROMPT = """As an assistant, your task is to serve as a moderator in a structured debate.

You will evaluate a debate between two responses labeled "Response 1" and "Response 2" and determine which response is better.

The debate topic is:
##debate_topic##

At the end of each round, you will evaluate both sides and decide which response is correct."""

# Affirmative prompt - follows your debate framework style
AFFIRMATIVE_PROMPT = """##debate_topic##

As a debater, analyze both responses carefully and argue for which one is better.

IMPORTANT: You must choose between "Response 1" and "Response 2"."""

# Negative prompt - follows your debate framework style
NEGATIVE_PROMPT = """##aff_ans##

You disagree with the affirmative side's answer. Provide your own analysis and reasoning.

IMPORTANT: You must choose between "Response 1" and "Response 2"."""

# Moderator prompt - follows your debate framework style with JSON format
MODERATOR_PROMPT = """Now the ##round## round of debate for both sides has ended.

Affirmative side arguing:
##aff_ans##

Negative side arguing: 
##neg_ans##

As the moderator, evaluate both sides' arguments and determine which response is better.

You MUST answer in the following JSON format:
{
    "reasoning": "your reasoning based on the debate",
    "Final Answer": "Response 1/Response 2"
}

Note that the 'Final Answer' MUST be placed at the end of your response, 
and the value must be only "Response 1" or "Response 2". 
Do not include any other text after the JSON response."""

# Judge prompts - following your debate framework style
JUDGE_PROMPT_1 = """Affirmative side arguing: ##aff_ans##

Negative side arguing: ##neg_ans##

Now, what answer candidates do we have? Present them without reasons."""

JUDGE_PROMPT_2 = """Therefore, ##debate_topic##

Please summarize your reasons and give the final answer that you think is correct.

You MUST answer in the following JSON format:
{
    "reasoning": "your reasoning based on the debate",
    "Final Answer": "Response 1/Response 2"
}

Note that the 'Final Answer' MUST be placed at the end of your response, 
and the value must be only "Response 1" or "Response 2". 
Do not include any other text after the JSON response."""

# Debate prompt - following your debate framework style
DEBATE_PROMPT = """##oppo_ans##

Do you agree with my perspective? Please provide your reasons and answer."""
