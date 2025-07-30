import json
import os
import random
from typing import Any, Dict, List, Optional

from .agent import Agent

random.seed(0)

# Default names for N debaters
DEFAULT_DEBATER_NAMES = [
    "Debater 1",
    "Debater 2",
    "Debater 3",
    "Debater 4",
    "Debater 5",
    "Debater 6",
    "Debater 7",
    "Debater 8",
    "Debater 9",
    "Debater 10",
]


class DebatePlayer(Agent):
    def __init__(
        self,
        model_name: str,
        name: str,
        temperature: float,
        provider: str = "ollama",
        sleep_time: float = 0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Create a player in the debate

        Args:
            model_name(str): model name
            name (str): name of this player
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            provider (str): LLM provider (e.g., "ollama", "openai")
            sleep_time (float): sleep because of rate limits
            base_url (Optional[str]): Base URL for the API calls
            api_key (Optional[str]): API key for the agent
        """
        super(DebatePlayer, self).__init__(
            model_name, name, temperature, provider, sleep_time, base_url, api_key
        )


class Debate:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        num_debaters: int = 2,  # Default to 2 debaters for practical use
        provider: str = "ollama",
        config: Optional[Dict[str, Any]] = None,
        max_round: int = 10,  # Increased default from 3 to 10
        sleep_time: float = 0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,  # Add verbose mode
    ) -> None:
        """Create a debate

        Args:
            model_name (str): model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_debaters (int): number of debaters (N debaters)
            provider (str): LLM provider (e.g., "ollama", "openai")
            config (Optional[Dict[str, Any]]): Configuration for the debate
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
            base_url (Optional[str]): Base URL for the API calls
            api_key (Optional[str]): API key for the agent
        """

        self.model_name = model_name
        self.temperature = temperature
        self.num_debaters = num_debaters
        self.provider = provider
        self.config = config or {}
        self.max_round = max_round
        self.sleep_time = sleep_time
        self.base_url = base_url
        self.api_key = api_key
        self.verbose = verbose  # Store verbose setting

        self.init_prompt()

        # creat&init agents
        self.creat_agents()
        self.init_agents()

    def init_prompt(self):
        """Initialize prompts by replacing placeholders with actual content."""
        if not self.config:
            return

        def prompt_replace(key):
            if (
                key in self.config
                and isinstance(self.config[key], str)
                and "##debate_topic##" in self.config[key]
            ):
                self.config[key] = self.config[key].replace(
                    "##debate_topic##", self.config.get("debate_topic", "")
                )

        prompt_replace("player_meta_prompt")
        prompt_replace("judge_meta_prompt")
        prompt_replace("debater_prompt")
        prompt_replace("judge_discriminative_prompt")
        prompt_replace("judge_extractive_prompt")

    def creat_agents(self):
        """Create debate players - N debaters + 1 judge."""
        # Create N debaters
        debater_names = DEFAULT_DEBATER_NAMES[: self.num_debaters]
        self.debaters = [
            DebatePlayer(
                model_name=self.model_name,
                name=name,
                temperature=self.temperature,
                provider=self.provider,
                sleep_time=self.sleep_time,
                base_url=self.base_url,
                api_key=self.api_key,
            )
            for name in debater_names
        ]

        # Create judge
        self.judge = DebatePlayer(
            model_name=self.model_name,
            name="Judge",
            temperature=self.temperature,
            provider=self.provider,
            sleep_time=self.sleep_time,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # All players (debaters + judge)
        self.players = self.debaters + [self.judge]

    def init_agents(self):
        """Initialize agents with their meta prompts."""
        # Set meta prompts for all debaters
        if "player_meta_prompt" in self.config:
            for debater in self.debaters:
                debater.set_meta_prompt(self.config["player_meta_prompt"])

        # Set meta prompt for judge
        if "judge_meta_prompt" in self.config:
            self.judge.set_meta_prompt(self.config["judge_meta_prompt"])

    def run_iterative_debate(self):
        """Run the iterative debate process with N debaters and two judge modes:
        1. Discriminative Mode (Jd): Judge decides if correct solution is obtained
        2. Extractive Mode (Je): Judge extracts final solution from debate history
        """

        debate_history = []
        current_round = 0

        # Iterative debate process
        for iteration in range(self.max_round):
            current_round = iteration + 1
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {current_round}")
                print(f"{'='*60}")

            iteration_responses = []

            # Each debater speaks one by one in fixed order
            for debater_idx, debater in enumerate(self.debaters):
                if self.verbose:
                    print(f"\n{debater.name} speaking...")
                    print("-" * 40)

                # Build debate history context for this debater
                history_context = self._build_debate_history_context(debate_history)

                # Create prompt for this debater based on debate history
                if "debater_prompt" in self.config:
                    # Assign different positions to debaters for better debate
                    if debater_idx == 0:
                        # First debater argues for plausibility (1)
                        debater_position = "You are arguing that the statement is PLAUSIBLE (1). Defend the position that the statement could be true."
                    else:
                        # Second debater argues against plausibility (0)
                        debater_position = "You are arguing that the statement is IMPLAUSIBLE (0). Defend the position that the statement is unlikely to be true."

                    debater_prompt = (
                        self.config["debater_prompt"]
                        .replace("##debate_history##", history_context)
                        .replace("##debater_name##", debater.name)
                        .replace("##debater_number##", str(debater_idx + 1))
                        .replace("##debater_position##", debater_position)
                    )

                    debater.add_event(debater_prompt)
                    # Temporarily reduce logging verbosity for cleaner output
                    import logging

                    original_level = logging.getLogger("multi_llm_debate.llm.llm").level
                    logging.getLogger("multi_llm_debate.llm.llm").setLevel(
                        logging.WARNING
                    )

                    debater_response = debater.ask(
                        json_mode=False
                    )  # Debaters should provide natural language responses

                    # Restore original logging level
                    logging.getLogger("multi_llm_debate.llm.llm").setLevel(
                        original_level
                    )
                    debater.add_memory(debater_response, verbose=self.verbose)

                    iteration_responses.append(
                        {
                            "debater_name": debater.name,
                            "debater_number": debater_idx + 1,
                            "response": debater_response,
                        }
                    )

                    if self.verbose:
                        print(f"----- {debater.name} -----")
                        print(debater_response)
                        print("-" * 40)

            # Add iteration to debate history
            debate_history.append(
                {"round": current_round, "responses": iteration_responses}
            )

            # Judge Discriminative Mode (Jd) - Decide if correct solution obtained
            if "judge_discriminative_prompt" in self.config:
                # Build debate history context
                history_context = self._build_debate_history_context(debate_history)

                discriminative_prompt = (
                    self.config["judge_discriminative_prompt"]
                    .replace("##debate_history##", history_context)
                    .replace("##current_round##", str(current_round))
                )

                self.judge.add_event(discriminative_prompt)
                # Temporarily reduce logging verbosity for cleaner output
                import logging

                original_level = logging.getLogger("multi_llm_debate.llm.llm").level
                logging.getLogger("multi_llm_debate.llm.llm").setLevel(logging.WARNING)

                self.judge_discriminative_decision = self.judge.ask(
                    json_mode=True
                )  # Judge needs JSON for structured decisions

                # Restore original logging level
                logging.getLogger("multi_llm_debate.llm.llm").setLevel(original_level)
                self.judge.add_memory(self.judge_discriminative_decision, verbose=self.verbose)

                try:
                    # Parse JSON response properly
                    if isinstance(self.judge_discriminative_decision, str):
                        import json

                        self.judge_discriminative_decision = json.loads(
                            self.judge_discriminative_decision
                        )
                except (ValueError, SyntaxError, NameError, json.JSONDecodeError):
                    # Fallback to unified format
                    self.judge_discriminative_decision = {
                        "solution_obtained": False,
                        "reasoning": "Unable to parse judge response",
                    }

                # Check if correct solution is obtained
                solution_obtained = self.judge_discriminative_decision.get(
                    "solution_obtained", False
                )

                if self.verbose:
                    print(
                        f"\nJudge Discriminative Decision (Iteration {current_round}):"
                    )
                    print("-" * 40)
                    print(f"Solution Obtained: {solution_obtained}")
                    print(
                        f"Reasoning: {self.judge_discriminative_decision.get('reasoning', 'No reasoning provided')}"
                    )
                    print("-" * 40)

                if solution_obtained:
                    if self.verbose:
                        print(
                            f"✓ Correct solution obtained in iteration {current_round} - debate concluded successfully"
                        )
                    # Solution found - debate is over, no need for Extractive Mode
                    self.config["success"] = True
                    self.config["iterations_used"] = current_round
                    self.config["solution_found_in_discriminative"] = True
                    # Use the discriminative decision as the final answer
                    if isinstance(self.judge_discriminative_decision, dict):
                        self.config.update(self.judge_discriminative_decision)
                    break
                else:
                    if self.verbose:
                        print(
                            f"⚠ No clear solution in iteration {current_round} - continuing to next iteration"
                        )
                    # Continue to next iteration
                    continue

        # Judge Extractive Mode (Je) - Only used when no solution found within iteration limit
        if (
            not self.config.get("success", False)
            and "judge_extractive_prompt" in self.config
        ):
            if self.verbose:
                print(f"\n{'='*60}")
                print("JUDGE EXTRACTIVE MODE")
                print(f"{'='*60}")
                print(
                    "No solution found within iteration limit - extracting final solution from complete debate history..."
                )

            # Build complete debate history context
            complete_history_context = self._build_debate_history_context(
                debate_history
            )

            extractive_prompt = self.config["judge_extractive_prompt"].replace(
                "##debate_history##", complete_history_context
            )

            self.judge.add_event(extractive_prompt)
            # Temporarily reduce logging verbosity for cleaner output
            import logging

            original_level = logging.getLogger("multi_llm_debate.llm.llm").level
            logging.getLogger("multi_llm_debate.llm.llm").setLevel(logging.WARNING)

            self.judge_extractive_decision = self.judge.ask(
                json_mode=True
            )  # Judge needs JSON for structured decisions

            # Restore original logging level
            logging.getLogger("multi_llm_debate.llm.llm").setLevel(original_level)
            self.judge.add_memory(self.judge_extractive_decision, verbose=self.verbose)

            try:
                # Parse JSON response properly
                if isinstance(self.judge_extractive_decision, str):
                    import json

                    self.judge_extractive_decision = json.loads(
                        self.judge_extractive_decision
                    )
            except (ValueError, SyntaxError, NameError, json.JSONDecodeError):
                # Fallback to unified format
                self.judge_extractive_decision = {
                    "reasoning": "Unable to parse judge response",
                    "Final Answer": "Response 1",
                }

            # Set success flag and update config with final decision from Extractive Mode
            if (
                isinstance(self.judge_extractive_decision, dict)
                and self.judge_extractive_decision.get("Final Answer", "") != ""
            ):
                self.config.update(self.judge_extractive_decision)
                self.config["success"] = True
                self.config["iterations_used"] = current_round
                self.config["solution_found_in_extractive"] = True
            else:
                self.config["success"] = False
                self.config["iterations_used"] = current_round
        else:
            # No Extractive Mode needed - solution was found in Discriminative Mode
            pass

        # Save debate history to config for output
        self.config["debate_history"] = debate_history
        
        if self.verbose:
            self.print_answer()
        return self.config

    def _build_debate_history_context(self, debate_history: list) -> str:
        """Build context string from debate history for judge evaluation.

        Args:
            debate_history: List of debate rounds and responses

        Returns:
            Context string for judge evaluation
        """
        context_parts = []

        for entry in debate_history:
            round_num = entry.get("round", 0)
            context_parts.append(f"Round {round_num}:")

            responses = entry.get("responses", [])
            for response in responses:
                debater_name = response.get("debater_name", "Unknown")
                debater_response = response.get("response", "")
                context_parts.append(f"  {debater_name}: {debater_response}")

        return "\n".join(context_parts)

    def run(self):
        """Run the complete debate process following the new structure:
        N debaters speak one by one in fixed order, then judge decides
        """

        # Use iterative debate by default
        return self.run_iterative_debate()

    def print_answer(self):
        """Print the final debate results."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("FINAL DEBATE RESULTS")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("DEBATE RESULTS")
            print("=" * 80)

        print(f"\nNumber of Debaters: {self.num_debaters}")
        print(f"Iterations Used: {self.config.get('iterations_used', 0)}")

        if hasattr(self, "judge_discriminative_decision"):
            if self.verbose:
                print(f"\nJudge Discriminative Decision:")
                print("-" * 40)
                if isinstance(self.judge_discriminative_decision, dict):
                    print(
                        f"Solution Obtained: {self.judge_discriminative_decision.get('solution_obtained', False)}"
                    )
                    print(
                        f"Reasoning: {self.judge_discriminative_decision.get('reasoning', 'No reasoning provided')}"
                    )
                else:
                    print(f"Raw Decision: {self.judge_discriminative_decision}")
                print("-" * 40)

        # Show which mode found the solution
        if self.config.get("solution_found_in_discriminative", False):
            if self.verbose:
                print(
                    f"\nSolution found in Discriminative Mode (Iteration {self.config.get('iterations_used', 0)})"
                )
        elif self.config.get("solution_found_in_extractive", False):
            if self.verbose:
                print(
                    f"\nSolution found in Extractive Mode (after {self.config.get('iterations_used', 0)} iterations)"
                )

        if hasattr(self, "judge_extractive_decision") and self.config.get(
            "solution_found_in_extractive", False
        ):
            if self.verbose:
                print(f"\nJudge Extractive Decision:")
                print("-" * 40)
                if isinstance(self.judge_extractive_decision, dict):
                    print(
                        f"Final Answer: {self.judge_extractive_decision.get('Final Answer', 'Unknown')}"
                    )
                    print(
                        f"Reasoning: {self.judge_extractive_decision.get('reasoning', 'No reasoning provided')}"
                    )
                else:
                    print(f"Raw Decision: {self.judge_extractive_decision}")
                print("-" * 40)

        if self.config.get("success", False):
            if self.verbose:
                print(f"\nFinal Answer: {self.config.get('Final Answer', 'Unknown')}")
                print(f"Reasoning: {self.config.get('reasoning', 'No reasoning provided')}")
        else:
            if self.verbose:
                print(f"\nNo clear decision reached")

    def broadcast(self, msg: str):
        """Broadcast a message to all players.
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcast a message to all other players.

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: DebatePlayer):
        """Ask a player to respond and broadcast their answer."""
        ans = player.ask()
        player.add_memory(ans, verbose=self.verbose)
        self.speak(player.name, ans)


if __name__ == "__main__":
    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 1)[0]

    while True:
        debate_topic = ""
        while debate_topic == "":
            debate_topic = input("\nEnter your debate topic: ")

        # Load default config or create one
        config_path = f"{MAD_path}/code/utils/config4all.json"
        if os.path.exists(config_path):
            config = json.load(open(config_path, "r"))
        else:
            # Use default prompts from prompts.py
            from .prompts import (
                DEBATER_PROMPT,
                JUDGE_DISCRIMINATIVE_PROMPT,
                JUDGE_EXTRACTIVE_PROMPT,
                JUDGE_META_PROMPT,
                PLAYER_META_PROMPT,
            )

            config = {
                "debate_topic": debate_topic,
                "player_meta_prompt": PLAYER_META_PROMPT,
                "judge_meta_prompt": JUDGE_META_PROMPT,
                "debater_prompt": DEBATER_PROMPT,
                "judge_discriminative_prompt": JUDGE_DISCRIMINATIVE_PROMPT,
                "judge_extractive_prompt": JUDGE_EXTRACTIVE_PROMPT,
            }

        config["debate_topic"] = debate_topic

        debate = Debate(
            num_debaters=2,
            provider="google",
            config=config,
            temperature=0,
            sleep_time=0,
        )
        debate.run()
