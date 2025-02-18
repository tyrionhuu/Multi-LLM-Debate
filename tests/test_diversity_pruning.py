import pytest
from sentence_transformers import SentenceTransformer
from multi_llm_debate.interventions.diversity_pruning import diversity_pruning


@pytest.fixture
def sentence_transformer():
    """Fixture providing a sentence transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def sample_responses():
    """Fixture providing sample responses for testing."""
    return [
        "The sky is blue.",
        "The sky is blue and clear.",  # Similar to first
        "Cats are cute pets.",  # Different topic
        "Dogs make great companions.",  # Different topic
        "Python is a programming language.",  # Different topic
    ]


def test_diversity_pruning_basic(sentence_transformer, sample_responses):
    """Test basic functionality of diversity pruning."""
    selected = diversity_pruning(
        responses=sample_responses,
        selected_amount=3,
        model=sentence_transformer,
    )
    
    assert len(selected) == 3
    assert all(resp in sample_responses for resp in selected)
    # Similar responses should not both be selected
    assert not (
        "The sky is blue." in selected 
        and "The sky is blue and clear." in selected
    )


def test_diversity_pruning_all_responses(sentence_transformer, sample_responses):
    """Test when selected amount equals total responses."""
    selected = diversity_pruning(
        responses=sample_responses,
        selected_amount=len(sample_responses),
        model=sentence_transformer,
    )
    assert len(selected) == len(sample_responses)
    assert set(selected) == set(sample_responses)


def test_diversity_pruning_single_response(sentence_transformer):
    """Test with a single response."""
    responses = ["Single response"]
    selected = diversity_pruning(
        responses=responses,
        selected_amount=1,
        model=sentence_transformer,
    )
    assert len(selected) == 1
    assert selected[0] == responses[0]


def test_diversity_pruning_no_model():
    """Test that error is raised when no model is provided."""
    with pytest.raises(ValueError):
        diversity_pruning(
            responses=["test"],
            selected_amount=1,
            model=None,
        )


def test_diversity_pruning_selected_greater_than_responses(sentence_transformer):
    """Test when selected amount is greater than available responses."""
    responses = ["Response 1", "Response 2"]
    selected = diversity_pruning(
        responses=responses,
        selected_amount=3,
        model=sentence_transformer,
    )
    assert len(selected) == 2
    assert set(selected) == set(responses)
