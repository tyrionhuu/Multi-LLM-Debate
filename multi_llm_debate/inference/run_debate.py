from pathlib import Path

from .agents_ensemble import AgentsEnsemble
from .round_n import run_debate_round_n
from .round_zero import run_debate_round_zero


def run_debate(
    round_zero_prompt: str,
    round_n_prompt: str,
    agents_ensemble: AgentsEnsemble,
    data_dir: str | Path,
) -> None:
    """
    Run a debate with the given prompts and agents.

    Args:
        round_zero_prompt (str): The initial prompt/question to start the debate
        round_n_prompt (str): The prompt for subsequent rounds of the debate
        agents_ensemble (AgentsEnsemble): Collection of LLM agents participating in the debate
        data_dir (str | Path): Directory path where debate responses will be saved

    Returns:
        None: Results are saved to file and logged
    """
    run_debate_round_zero(round_zero_prompt, agents_ensemble, data_dir)
    for i in range(1, agents_ensemble.num_agents):
        run_debate_round_n(round_n_prompt, agents_ensemble, data_dir, i)
