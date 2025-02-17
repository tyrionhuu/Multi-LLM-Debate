from pathlib import Path

from .agents_ensemble import AgentsEnsemble
from .round_n import run_debate_round_n
from .round_zero import run_debate_round_zero


def run_debate(
    max_rounds: int,
    round_zero_prompt: str,
    round_n_prompt: str,
    agents_ensemble: AgentsEnsemble,
    output_dir: str | Path,
) -> None:
    """
    Run a full debate with multiple rounds using the given prompts and agents.
    
    Args:
        max_rounds (int): Maximum number of rounds to run
        round_zero_prompt (str): The initial prompt/question to start the debate
        round_n_prompt (str): The debate prompt including previous context
        agents_ensemble (AgentsEnsemble): Collection of LLM agents participating in the debate
        output_dir (str | Path): Directory path where debate responses will be saved
        
    Returns:
        None: Results are saved to file and logged
    """
    for i in range(max_rounds):
        if i == 0:
            run_debate_round_zero(round_zero_prompt, agents_ensemble, output_dir)
        else:
            run_debate_round_n(round_n_prompt, agents_ensemble, output_dir, i)