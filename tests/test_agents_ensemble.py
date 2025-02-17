from unittest.mock import patch

import pytest

from multi_llm_debate.inference.agent import Agent
from multi_llm_debate.inference.agents_ensemble import AgentsEnsemble


@pytest.fixture
def mock_agents():
    """Fixture providing a list of mock agents."""
    return [
        Agent(agent_id=1, model="model1", provider="provider1"),
        Agent(agent_id=2, model="model2", provider="provider2"),
        Agent(agent_id=3, model="model3", provider="provider3"),
    ]


@pytest.fixture
def ensemble(mock_agents):
    """Fixture providing an AgentsEnsemble instance."""
    return AgentsEnsemble(agents=mock_agents)


def test_ensemble_initialization(mock_agents):
    """Test AgentsEnsemble initialization with agents."""
    ensemble = AgentsEnsemble(agents=mock_agents)
    assert len(ensemble.agents) == 3
    assert all(isinstance(agent, Agent) for agent in ensemble.agents)


def test_empty_ensemble_initialization():
    """Test AgentsEnsemble initialization with no agents."""
    with pytest.raises(ValueError):
        AgentsEnsemble(agents=[])


@pytest.mark.parametrize(
    "responses,expected",
    [
        (
            [
                {"choice": "A"},
                {"choice": "A"},
                {"choice": "B"},
            ],
            "A",
        ),
        (
            [
                {"choice": "A"},
                {"choice": "B"},
                {"choice": "C"},
            ],
            None,  # No majority
        ),
    ],
)
def test_ensemble_responses(ensemble, responses, expected):
    """Test response handling with different response patterns."""
    with patch.object(Agent, "respond") as mock_respond:
        mock_respond.side_effect = responses
        result = ensemble.get_responses("What is your choice?")
        assert len(result) == len(responses)


@pytest.mark.integration
def test_ensemble_integration():
    """Test actual ensemble integration with real LLM calls.

    Requires:
    - Multiple Ollama models to be available
    """
    agents = [
        Agent(agent_id=1, model="llama3.1:latest", provider="ollama"),
        Agent(agent_id=2, model="llama3.1:latest", provider="ollama"),
    ]
    ensemble = AgentsEnsemble(auto_init=False)
    for agent in agents:
        ensemble.add_agent(agent)

    prompt = (
        "You are helping with a simple choice. "
        "Answer with a JSON object containing a 'choice' key "
        "with either 'yes' or 'no' as the value. "
        "Question: Is Python a programming language?"
    )

    responses = ensemble.get_responses(prompt)
    assert len(responses) == 2
    for response in responses:
        assert isinstance(response, dict), "Response should be a dictionary"
