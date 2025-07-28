import json
import os
import random
from typing import Any, Dict, Optional

from .agent import Agent

random.seed(0)

NAME_LIST = [
    "Affirmative side",
    "Negative side",
    "Moderator",
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
        num_players: int = 3,
        provider: str = "ollama",
        config: Optional[Dict[str, Any]] = None,
        max_round: int = 3,
        sleep_time: float = 0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Create a debate

        Args:
            model_name (str): model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_players (int): num of players
            provider (str): LLM provider (e.g., "ollama", "openai")
            config (Optional[Dict[str, Any]]): Configuration for the debate
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
            base_url (Optional[str]): Base URL for the API calls
            api_key (Optional[str]): API key for the agent
        """

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.provider = provider
        self.config = config or {}
        self.max_round = max_round
        self.sleep_time = sleep_time
        self.base_url = base_url
        self.api_key = api_key

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
        prompt_replace("moderator_meta_prompt")
        prompt_replace("affirmative_prompt")
        prompt_replace("judge_prompt_last2")

    def creat_agents(self):
        """Create debate players."""
        # creates players
        self.players = [
            DebatePlayer(
                model_name=self.model_name,
                name=name,
                temperature=self.temperature,
                provider=self.provider,
                sleep_time=self.sleep_time,
                base_url=self.base_url,
                api_key=self.api_key,
            )
            for name in NAME_LIST
        ]
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

    def init_agents(self):
        """Initialize agents with their meta prompts and first round."""
        # start: set meta prompt
        if "player_meta_prompt" in self.config:
            self.affirmative.set_meta_prompt(self.config["player_meta_prompt"])
            self.negative.set_meta_prompt(self.config["player_meta_prompt"])
        if "moderator_meta_prompt" in self.config:
            self.moderator.set_meta_prompt(self.config["moderator_meta_prompt"])

        # start: first round debate, state opinions
        print("===== Debate Round-1 =====\n")

        if "affirmative_prompt" in self.config:
            self.affirmative.add_event(self.config["affirmative_prompt"])
            self.aff_ans = self.affirmative.ask()
            self.affirmative.add_memory(self.aff_ans)
            self.config["base_answer"] = self.aff_ans

        # Convert JSON response to string for prompt replacement
        aff_ans_str = (
            str(self.aff_ans)
            if isinstance(self.aff_ans, (dict, list))
            else self.aff_ans
        )

        if "negative_prompt" in self.config:
            neg_prompt = self.config["negative_prompt"].replace(
                "##aff_ans##", aff_ans_str
            )
            self.negative.add_event(neg_prompt)
            self.neg_ans = self.negative.ask()
            self.negative.add_memory(self.neg_ans)

        if "moderator_prompt" in self.config:
            # Convert responses to strings for prompt replacement
            neg_ans_str = (
                str(self.neg_ans)
                if isinstance(self.neg_ans, (dict, list))
                else self.neg_ans
            )

            mod_prompt = (
                self.config["moderator_prompt"]
                .replace("##aff_ans##", aff_ans_str)
                .replace("##neg_ans##", neg_ans_str)
                .replace("##round##", "first")
            )
            self.moderator.add_event(mod_prompt)
            self.mod_ans = self.moderator.ask()
            self.moderator.add_memory(self.mod_ans)
            try:
                # Only eval if it's a string, otherwise assume it's already parsed
                if isinstance(self.mod_ans, str):
                    self.mod_ans = eval(self.mod_ans)
            except (ValueError, SyntaxError, NameError):
                self.mod_ans = {
                    "debate_answer": "",
                    "Whether there is a preference": "No",
                }

    def round_dct(self, num: int):
        """Convert round number to text representation."""
        dct = {
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
            7: "seventh",
            8: "eighth",
            9: "ninth",
            10: "tenth",
        }
        return dct.get(num, f"{num}th")

    def print_answer(self):
        """Print the final debate results."""
        print("\n\n===== Debate Done! =====")
        print("\n----- Debate Topic -----")
        print(self.config.get("debate_topic", ""))
        print("\n----- Base Answer -----")
        print(self.config.get("base_answer", ""))
        print("\n----- Debate Answer -----")
        print(self.config.get("debate_answer", ""))
        print("\n----- Debate Reason -----")
        print(self.config.get("Reason", ""))

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
        player.add_memory(ans)
        self.speak(player.name, ans)

    def run(self):
        """Run the complete debate process."""
        for round in range(self.max_round - 1):
            if (
                isinstance(self.mod_ans, dict)
                and self.mod_ans.get("debate_answer", "") != ""
            ):
                break
            else:
                print(f"===== Debate Round-{round+2} =====\n")

                if "debate_prompt" in self.config:
                    # Convert responses to strings for prompt replacement
                    neg_ans_str = (
                        str(self.neg_ans)
                        if isinstance(self.neg_ans, (dict, list))
                        else self.neg_ans
                    )

                    self.affirmative.add_event(
                        self.config["debate_prompt"].replace(
                            "##oppo_ans##", neg_ans_str
                        )
                    )
                    self.aff_ans = self.affirmative.ask()
                    self.affirmative.add_memory(self.aff_ans)

                    # Convert responses to strings for prompt replacement
                    aff_ans_str = (
                        str(self.aff_ans)
                        if isinstance(self.aff_ans, (dict, list))
                        else self.aff_ans
                    )

                    self.negative.add_event(
                        self.config["debate_prompt"].replace(
                            "##oppo_ans##", aff_ans_str
                        )
                    )
                    self.neg_ans = self.negative.ask()
                    self.negative.add_memory(self.neg_ans)

                if "moderator_prompt" in self.config:
                    # Convert responses to strings for prompt replacement
                    aff_ans_str = (
                        str(self.aff_ans)
                        if isinstance(self.aff_ans, (dict, list))
                        else self.aff_ans
                    )
                    neg_ans_str = (
                        str(self.neg_ans)
                        if isinstance(self.neg_ans, (dict, list))
                        else self.neg_ans
                    )

                    mod_prompt = (
                        self.config["moderator_prompt"]
                        .replace("##aff_ans##", aff_ans_str)
                        .replace("##neg_ans##", neg_ans_str)
                        .replace("##round##", self.round_dct(round + 2))
                    )
                    self.moderator.add_event(mod_prompt)
                    self.mod_ans = self.moderator.ask()
                    self.moderator.add_memory(self.mod_ans)
                    try:
                        # Only eval if it's a string, otherwise assume it's already parsed
                        if isinstance(self.mod_ans, str):
                            self.mod_ans = eval(self.mod_ans)
                    except (ValueError, SyntaxError, NameError):
                        self.mod_ans = {
                            "debate_answer": "",
                            "Whether there is a preference": "No",
                        }

        if (
            isinstance(self.mod_ans, dict)
            and self.mod_ans.get("debate_answer", "") != ""
        ):
            self.config.update(self.mod_ans)
            self.config["success"] = True

        # ultimate deadly technique.
        else:
            judge_player = DebatePlayer(
                model_name=self.model_name,
                name="Judge",
                temperature=self.temperature,
                provider=self.provider,
                sleep_time=self.sleep_time,
                base_url=self.base_url,
                api_key=self.api_key,
            )

            if len(self.affirmative.memory_lst) > 2:
                aff_ans = self.affirmative.memory_lst[2]["content"]
            else:
                aff_ans = ""

            if len(self.negative.memory_lst) > 2:
                neg_ans = self.negative.memory_lst[2]["content"]
            else:
                neg_ans = ""

            if "moderator_meta_prompt" in self.config:
                judge_player.set_meta_prompt(self.config["moderator_meta_prompt"])

            # extract answer candidates
            if "judge_prompt_last1" in self.config:
                judge_prompt = (
                    self.config["judge_prompt_last1"]
                    .replace("##aff_ans##", aff_ans)
                    .replace("##neg_ans##", neg_ans)
                )
                judge_player.add_event(judge_prompt)
                ans = judge_player.ask()
                judge_player.add_memory(ans)

            # select one from the candidates
            if "judge_prompt_last2" in self.config:
                judge_player.add_event(self.config["judge_prompt_last2"])
                ans = judge_player.ask()
                judge_player.add_memory(ans)

                try:
                    # Only eval if it's a string, otherwise assume it's already parsed
                    if isinstance(ans, str):
                        ans = eval(ans)
                except (ValueError, SyntaxError, NameError):
                    ans = {"debate_answer": "", "Reason": ""}

                if ans.get("debate_answer", "") != "":
                    self.config["success"] = True
                self.config.update(ans)
                self.players.append(judge_player)

        self.print_answer()
        return self.config


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
                AFFIRMATIVE_PROMPT,
                DEBATE_PROMPT,
                JUDGE_PROMPT_1,
                JUDGE_PROMPT_2,
                MODERATOR_META_PROMPT,
                MODERATOR_PROMPT,
                NEGATIVE_PROMPT,
                PLAYER_META_PROMPT,
            )

            config = {
                "debate_topic": debate_topic,
                "player_meta_prompt": PLAYER_META_PROMPT,
                "moderator_meta_prompt": MODERATOR_META_PROMPT,
                "affirmative_prompt": AFFIRMATIVE_PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "moderator_prompt": MODERATOR_PROMPT,
                "judge_prompt_last1": JUDGE_PROMPT_1,
                "judge_prompt_last2": JUDGE_PROMPT_2,
                "debate_prompt": DEBATE_PROMPT,
            }

        config["debate_topic"] = debate_topic

        debate = Debate(
            num_players=3, provider="ollama", config=config, temperature=0, sleep_time=0
        )
        debate.run()
