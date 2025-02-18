import pytest
from sentence_transformers import SentenceTransformer

from multi_llm_debate.interventions.quality_pruning import quality_pruning


@pytest.fixture
def sentence_model():
    """Fixture to provide a sentence transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def test_quality_pruning_valid_input(sentence_model):
    """Test quality pruning with valid inputs."""
    task = "Explain quantum computing"
    responses = [
        "Quantum computing uses quantum bits.",
        "The weather is sunny today.",
        "Quantum computers leverage superposition.",
        "I like to play basketball.",
    ]
    selected = quality_pruning(responses, task, 2, sentence_model)

    assert len(selected) == 2
    # Verify quantum-related responses are selected
    assert all("quantum" in response.lower() for response in selected)


def test_quality_pruning_small_input(sentence_model):
    """Test when input list is smaller than requested amount."""
    responses = ["Response 1", "Response 2"]
    selected = quality_pruning(responses, "Task", 3, sentence_model)

    assert len(selected) == 2
    assert selected == responses


def test_quality_pruning_no_model():
    """Test error handling when no model is provided."""
    with pytest.raises(ValueError):
        quality_pruning(["Response 1"], "Task", 1, None)


def test_quality_pruning_ordering(sentence_model):
    """Test if outputs maintain correct ordering based on similarity."""
    task = "Tell me about dogs"
    responses = [
        "Cats are independent pets.",
        "Dogs are loyal companions.",
        "Dogs make great pets.",
        "The sky is blue today.",
    ]
    selected = quality_pruning(responses, task, 3, sentence_model)

    # First two selections should be about dogs
    assert sum("dog" in response.lower() for response in selected[:2]) == 2
