from multi_llm_debate.llm.prompts import (
    BOOL_JSON_FORMAT,
    BOOL_JSON_FORMAT_COT,
    BOOL_NON_JSON_FORMAT,
    BOOL_NON_JSON_FORMAT_COT,
    build_bool_q_round_n_prompt,
    build_bool_q_round_zero_prompt,
)
from multi_llm_debate.llm.prompt_builder import PromptBuilder

def test_build_bool_q_round_zero_prompt_with_cot() -> None:
    question = "Is the sky blue?"
    passage = "The sky appears blue due to Rayleigh scattering."

    prompt = build_bool_q_round_zero_prompt(
        question, passage, use_cot=True, json_mode=True
    )

    assert "Question: Is the sky blue?" in prompt
    assert "Passage: The sky appears blue due to Rayleigh scattering." in prompt
    assert BOOL_JSON_FORMAT_COT in prompt


def test_build_bool_q_round_zero_prompt_without_cot() -> None:
    question = "Is the sky blue?"
    passage = "The sky appears blue due to Rayleigh scattering."

    prompt = build_bool_q_round_zero_prompt(
        question, passage, use_cot=False, json_mode=True
    )

    assert "Question: Is the sky blue?" in prompt
    assert "Passage: The sky appears blue due to Rayleigh scattering." in prompt
    assert BOOL_JSON_FORMAT in prompt


def test_build_bool_q_round_n_prompt_with_cot() -> None:
    question = "Is the sky blue?"
    passage = "The sky appears blue due to Rayleigh scattering."
    responses = ["Response 1", "Response 2"]

    prompt = build_bool_q_round_n_prompt(
        question, passage, responses, use_cot=True, json_mode=True
    )

    assert "Model 1: Response 1" in prompt
    assert "Model 2: Response 2" in prompt
    assert "Question: Is the sky blue?" in prompt
    assert "Passage: The sky appears blue due to Rayleigh scattering." in prompt
    assert BOOL_JSON_FORMAT_COT in prompt


def test_build_bool_q_round_n_prompt_without_cot() -> None:
    question = "Is the sky blue?"
    passage = "The sky appears blue due to Rayleigh scattering."
    responses = ["Response 1", "Response 2"]

    prompt = build_bool_q_round_n_prompt(
        question, passage, responses, use_cot=False, json_mode=True
    )

    assert "Model 1: Response 1" in prompt
    assert "Model 2: Response 2" in prompt
    assert "Question: Is the sky blue?" in prompt
    assert "Passage: The sky appears blue due to Rayleigh scattering." in prompt
    assert BOOL_JSON_FORMAT in prompt


def test_prompt_builder() -> None:
    params = {
        "question": "Is the sky blue?",
        "passage": "The sky appears blue due to Rayleigh scattering.",
        "use_cot": True,
        "json_mode": True,
    }

    builder = PromptBuilder(
        round_zero_fn=build_bool_q_round_zero_prompt,
        round_n_fn=build_bool_q_round_n_prompt,
        prompt_params=params,
    )

    round_zero = builder.build_round_zero()
    assert BOOL_JSON_FORMAT_COT in round_zero
    assert "Question: Is the sky blue?" in round_zero

    responses = ["Response 1", "Response 2"]
    round_n = builder.build_round_n(responses)
    assert "Model 1: Response 1" in round_n
    assert "Model 2: Response 2" in round_n
    assert BOOL_JSON_FORMAT_COT in round_n


def test_build_bool_q_round_zero_prompt_json_mode() -> None:
    """Test round zero prompt with json_mode=True."""
    question = "Is the sky blue?"
    passage = "The sky appears blue due to Rayleigh scattering."

    prompt = build_bool_q_round_zero_prompt(
        question, passage, use_cot=True, json_mode=True
    )

    assert "JSON format" in prompt
    assert BOOL_JSON_FORMAT_COT in prompt

    prompt = build_bool_q_round_zero_prompt(
        question, passage, use_cot=False, json_mode=True
    )

    assert "JSON format" in prompt
    assert BOOL_JSON_FORMAT in prompt


def test_build_bool_q_round_zero_prompt_non_json_mode() -> None:
    """Test round zero prompt with json_mode=False."""
    question = "Is the sky blue?"
    passage = "The sky appears blue due to Rayleigh scattering."

    prompt = build_bool_q_round_zero_prompt(
        question, passage, use_cot=True, json_mode=False
    )

    assert "following format" in prompt
    assert BOOL_NON_JSON_FORMAT_COT in prompt

    prompt = build_bool_q_round_zero_prompt(
        question, passage, use_cot=False, json_mode=False
    )

    assert "following format" in prompt
    assert BOOL_NON_JSON_FORMAT in prompt


def test_build_bool_q_round_n_prompt_json_mode() -> None:
    """Test round n prompt with json_mode=True."""
    question = "Is the sky blue?"
    passage = "The sky appears blue due to Rayleigh scattering."
    responses = ["Response 1", "Response 2"]

    prompt = build_bool_q_round_n_prompt(
        question, passage, responses, use_cot=True, json_mode=True
    )

    assert "JSON format" in prompt
    assert BOOL_JSON_FORMAT_COT in prompt

    prompt = build_bool_q_round_n_prompt(
        question, passage, responses, use_cot=False, json_mode=True
    )

    assert "JSON format" in prompt
    assert BOOL_JSON_FORMAT in prompt


def test_build_bool_q_round_n_prompt_non_json_mode() -> None:
    """Test round n prompt with json_mode=False."""
    question = "Is the sky blue?"
    passage = "The sky appears blue due to Rayleigh scattering."
    responses = ["Response 1", "Response 2"]

    prompt = build_bool_q_round_n_prompt(
        question, passage, responses, use_cot=True, json_mode=False
    )

    assert "following format" in prompt
    assert BOOL_NON_JSON_FORMAT_COT in prompt

    prompt = build_bool_q_round_n_prompt(
        question, passage, responses, use_cot=False, json_mode=False
    )

    assert "following format" in prompt
    assert BOOL_NON_JSON_FORMAT in prompt

def test_prompt_builder_with_json_mode() -> None:
    """Test PromptBuilder with json_mode parameter."""
    params = {
        "question": "Is the sky blue?",
        "passage": "The sky appears blue due to Rayleigh scattering.",
        "use_cot": True,
        "json_mode": True,
    }

    builder = PromptBuilder(
        round_zero_fn=build_bool_q_round_zero_prompt,
        round_n_fn=build_bool_q_round_n_prompt,
        prompt_params=params,
    )

    round_zero = builder.build_round_zero()
    assert "JSON format" in round_zero
    assert BOOL_JSON_FORMAT_COT in round_zero

    responses = ["Response 1", "Response 2"]
    round_n = builder.build_round_n(responses)
    assert "JSON format" in round_n
    assert BOOL_JSON_FORMAT_COT in round_n
