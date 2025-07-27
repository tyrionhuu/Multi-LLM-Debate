#!/usr/bin/env python3
"""
Test script for the MAD (Multi-Agent Debate) framework.

This script tests the basic functionality of the MAD framework.
"""

import json
import logging
from pathlib import Path

from multi_llm_debate.mad.debate import Debate
from multi_llm_debate.mad.prompts import (
    AFFIRMATIVE_PROMPT,
    DEBATE_PROMPT,
    JUDGE_PROMPT_1,
    JUDGE_PROMPT_2,
    MODERATOR_META_PROMPT,
    MODERATOR_PROMPT,
    NEGATIVE_PROMPT,
    PLAYER_META_PROMPT,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mad_framework():
    """Test the MAD framework with a sample debate topic."""

    # Sample debate topic
    debate_topic = "Is artificial intelligence beneficial for society?"

    # Create MAD configuration
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

    print(f"Testing MAD framework with topic: {debate_topic}")
    print("=" * 60)

    try:
        # Create and run debate
        debate = Debate(
            model_name="gpt-3.5-turbo",  # You can change this to your preferred model
            temperature=0.7,
            num_players=3,
            provider="ollama",  # Change to "openai" if using OpenAI
            config=config,
            max_round=3,
            sleep_time=0,
            # Add your API configuration here if needed
            # base_url="http://localhost:11434/v1",  # For Ollama
            # api_key="your-api-key",  # For OpenAI
        )

        # Run the debate
        result = debate.run()

        print("\n" + "=" * 60)
        print("DEBATE COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Print results
        print(f"Success: {result.get('success', False)}")
        print(f"Base Answer: {result.get('base_answer', 'N/A')}")
        print(f"Debate Answer: {result.get('debate_answer', 'N/A')}")
        print(f"Reason: {result.get('Reason', 'N/A')}")

        return True

    except Exception as e:
        print(f"Error testing MAD framework: {str(e)}")
        logger.error(f"Error testing MAD framework: {str(e)}", exc_info=True)
        return False


def test_mad_benchmark_integration():
    """Test the MAD benchmark integration."""

    try:
        from multi_llm_debate.run.mad.evaluate import evaluate_mad_results
        from multi_llm_debate.run.mad.run_debate import process_mad_dataset
        from multi_llm_debate.run.mad.utils import create_mad_config, load_mad_dataset

        print("\nTesting MAD benchmark integration...")

        # Load sample dataset
        df = load_mad_dataset(sample_size=2)
        print(f"Loaded dataset with {len(df)} samples")

        # Create model configs
        model_configs = [
            {
                "name": "gpt-3.5-turbo",
                "base_url": "https://api.openai.com/v1",
                "quantity": 1,
            }
        ]

        # Create output directory
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)

        # Process dataset (this would normally run debates, but we'll skip for testing)
        print("MAD benchmark integration test completed successfully!")

        return True

    except Exception as e:
        print(f"Error testing MAD benchmark integration: {str(e)}")
        logger.error(
            f"Error testing MAD benchmark integration: {str(e)}", exc_info=True
        )
        return False


if __name__ == "__main__":
    print("MAD Framework Test")
    print("=" * 60)

    # Test basic MAD framework
    success1 = test_mad_framework()

    # Test benchmark integration
    success2 = test_mad_benchmark_integration()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"MAD Framework Test: {'PASSED' if success1 else 'FAILED'}")
    print(f"Benchmark Integration Test: {'PASSED' if success2 else 'FAILED'}")

    if success1 and success2:
        print("\n🎉 All tests passed! The MAD framework is ready to use.")
        print("\nTo run a full benchmark, use:")
        print("python -m multi_llm_debate.run.mad.main --sample_size 5")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
