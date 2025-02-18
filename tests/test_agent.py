from unittest.mock import patch

import pytest

from multi_llm_debate.debate.agent import Agent


@pytest.fixture
def test_agent():
    """Fixture providing a test Agent instance."""
    return Agent(agent_id=1, model="llama2:3.1", provider="Ollama")


def test_agent_initialization():
    """Test Agent instance initialization with basic attributes."""
    agent = Agent(agent_id=1, model="llama2:3.1", provider="Ollama")
    assert agent.agent_id == 1
    assert agent.model == "llama2:3.1"
    assert agent.provider == "Ollama"


def test_agent_string_representation(test_agent):
    """Test string and repr methods of Agent."""
    assert str(test_agent) == "Agent 1 (llama2:3.1)"
    assert repr(test_agent) == "Agent 1 (llama2:3.1)"


@pytest.mark.parametrize(
    "mock_response,expected_content",
    [
        (
            '{"key": "value"}',
            {"key": "value"},
        ),
        (
            {"key": "value"},
            {"key": "value"},
        ),
        (
            "invalid json",
            {"raw_content": "invalid json"},
        ),
        (
            123,
            {"raw_content": "123"},
        ),
    ],
)
def test_agent_respond(test_agent, mock_response, expected_content):
    """Test agent response handling with different response types."""
    with patch(
        "multi_llm_debate.debate.agent.call_model",
        return_value=mock_response,
    ):
        response = test_agent.respond("test prompt")
        expected = {
            "agent_id": 1,
            "model": "llama2:3.1",
            "response": expected_content,
        }
        assert response == expected


def test_agent_respond_call_parameters(test_agent):
    """Test that respond method calls LLM with correct parameters."""
    with patch("multi_llm_debate.debate.agent.call_model") as mock_call:
        test_agent.respond("test prompt")
        mock_call.assert_called_once_with(
            model_name="llama3.1:latest",
            provider="ollama",
            prompt="test prompt",
        )


@pytest.mark.integration
def test_real_llm_call():
    """Test actual LLM API call integration using Ollama.

    This test requires:
    - Ollama to be running locally
    - llama3.1:latest model to be pulled

    Run with: pytest -v -m integration
    """
    agent = Agent(agent_id=1, model="llama3.1:latest", provider="ollama")

    # Test with a simple instruction that works well with Llama
    prompt = "Complete this sequence: 1, 2, 3, ..."
    response = agent.respond(prompt)

    assert isinstance(response, dict)
    # Llama might return the response in different formats
    response_str = str(response)
    assert response_str != "", "Response should not be empty"

    # Test JSON understanding
    prompt = (
        "You are a helpful AI assistant. "
        "Return a JSON object with your name. "
        "Format: {'name': 'Llama'}"
    )
    response = agent.respond(prompt)
    response_str = str(response)
    assert "llama" in response_str.lower(), "Expected 'llama' in response"
