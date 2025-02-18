from ...debate.run_debate import run_debate
from ...debate.agents_ensemble import AgentsEnsemble
from ...llm.prompts import PromptBuilder
from ...llm.prompts import (
    build_bool_q_round_n_prompt,
    build_bool_q_round_zero_prompt,
)
from pathlib import Path
import pandas as pd
def run_bool_q_single_entry(
    entry: pd.Series,
    max_rounds: int = 10,
    base_dir: Path = Path("data" / "bool_q")
) -> None:
    """Run a single entry for the Boolean Question task.

    Args:
        question: The question to be answered.
        answer: The expected answer to the question.
        passage: The passage containing information related to the question.
        max_rounds: Maximum number of debate rounds.
    """
    # Check if the entry is valid
    if not isinstance(entry, pd.Series):
        raise ValueError("Entry must be a pandas Series.")
    
    if "question" not in entry or "answer" not in entry or "passage" not in entry or "id" not in entry:
        raise ValueError("Entry must contain 'question', 'answer', 'passage', and 'id'.")
    
    # Extract values from the entry
    question = entry["question"]
    answer = entry["answer"]
    passage = entry["passage"]
    id = entry["id"]
    
    output_dir = base_dir / id
    
    # Initialize prompt builder
    prompt_builder = PromptBuilder(
        round_zero_fn=build_bool_q_round_zero_prompt,
        round_n_fn=build_bool_q_round_n_prompt,
    )

    # Initialize agents
    agents_ensemble = AgentsEnsemble()
        
    # Run the debate
    run_debate(
        agents_ensemble,
        prompt_builder,
        max_rounds=max_rounds,
        output_dir=output_dir,
    )