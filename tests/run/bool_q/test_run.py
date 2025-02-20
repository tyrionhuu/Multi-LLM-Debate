from multi_llm_debate.run.bool_q.run import _build_config_desc
from multi_llm_debate.utils.model_config import ModelConfig


def test_build_config_desc_with_llama3():
    """Test config description builder with llama3 configuration."""
    # Test configuration
    test_config = [
        ModelConfig(provider="ollama", name="llama3", quantity=3),
        ModelConfig(provider="ollama", name="llama2", quantity=3),
        ModelConfig(provider="ollama", name="Mistral", quantity=3),
    ]

    # Test with CoT enabled
    desc_cot = _build_config_desc(test_config, use_cot=True, max_rounds=10)
    assert desc_cot == "9 models (llama3×3, llama2×3, Mistral×3) | CoT | Max rounds: 10"

    # Test with CoT disabled
    desc_no_cot = _build_config_desc(test_config, use_cot=False, max_rounds=10)
    assert (
        desc_no_cot
        == "9 models (llama3×3, llama2×3, Mistral×3) | No CoT | Max rounds: 10"
    )


def test_build_config_desc_edge_cases():
    """Test config description builder with edge cases."""
    # Test with empty config
    desc_empty = _build_config_desc(None, use_cot=True, max_rounds=5)
    assert desc_empty == "1 models (default) | CoT | Max rounds: 5"

    # Test with empty list
    desc_empty_list = _build_config_desc([], use_cot=True, max_rounds=5)
    assert desc_empty_list == "1 models (default) | CoT | Max rounds: 5"
