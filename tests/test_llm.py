from unittest.mock import MagicMock, patch

import pytest

from multi_llm_debate.llm.llm import (
    call_model,
    generate_api_messages,
    generate_with_api,
    generate_with_ollama,
)


# Test message generation
def test_generate_api_messages_text_only():
    prompt = "Test prompt"
    messages = generate_api_messages(prompt)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == prompt


# Test model generation with mocks
@patch("multi_llm_debate.llm.llm.ollama")
def test_generate_with_ollama(mock_ollama):
    mock_ollama.generate.return_value = {"response": "Test response"}

    response = generate_with_ollama(
        model_name="test-model", prompt="test prompt", temperature=0.1, max_tokens=100
    )

    assert response == "Test response"
    mock_ollama.generate.assert_called_once()


@patch("multi_llm_debate.llm.llm.OpenAI")
def test_generate_with_api(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        "Test response"
    )

    response = generate_with_api(
        model_name="test-model", prompt="test prompt", temperature=0.1, max_tokens=100
    )

    assert response == "Test response"
    mock_client.chat.completions.create.assert_called_once()


# Test high-level interface
@patch("multi_llm_debate.llm.llm.generate_with_ollama")
def test_call_model_ollama(mock_generate):
    mock_generate.return_value = "Test response"

    response = call_model(
        model_name="test-model", provider="ollama", prompt="test prompt"
    )

    assert response == "Test response"
    mock_generate.assert_called_once()


@patch("multi_llm_debate.llm.llm.generate_with_api")
def test_call_model_api(mock_generate):
    mock_generate.return_value = "Test response"

    response = call_model(model_name="test-model", provider="api", prompt="test prompt")

    assert response == "Test response"
    mock_generate.assert_called_once()


def test_call_model_invalid_provider():
    with pytest.raises(ValueError):
        call_model(provider="invalid")
